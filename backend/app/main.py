from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import asyncio
import logging
import os

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.core.config import settings
from app.core.database import engine, Base
from app.routes import auth, measurements, brands, products, webhooks, batch

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rate limiter setup - uses client IP address for tracking
# In production, consider using Redis backend for distributed rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{settings.RATE_LIMIT_PER_MINUTE}/minute"],
    storage_uri=settings.REDIS_URL if settings.ENVIRONMENT == "production" else "memory://",
)

# Create database tables
Base.metadata.create_all(bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI app.
    Pre-loads ML models on startup to avoid cold-start delays.
    """
    logger.info("Starting up - Pre-loading ML models...")

    # Pre-load models in a thread pool to avoid blocking
    loop = asyncio.get_event_loop()

    def preload_models():
        try:
            # Import and cache models
            from app.routes.measurements import get_cached_pose_detector, get_cached_processor
            get_cached_pose_detector()
            get_cached_processor(use_ml_ratios=True)
            logger.info("ML models pre-loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to pre-load ML models: {e}")

    await loop.run_in_executor(None, preload_models)

    yield  # App is running

    # Cleanup on shutdown
    logger.info("Shutting down...")

# Create FastAPI app with lifespan for model preloading
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="AI-powered body measurement API for e-commerce size recommendations",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add rate limiter to app state and exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Add rate limiting middleware (added first, runs last)
app.add_middleware(SlowAPIMiddleware)

# Custom middleware to handle client disconnection gracefully
@app.middleware("http")
async def handle_client_disconnect(request: Request, call_next):
    """
    Middleware to handle client disconnection gracefully.
    This prevents the server from processing requests when the client has left.
    """
    try:
        response = await call_next(request)
        return response
    except asyncio.CancelledError:
        logger.info(f"Request cancelled (client disconnected): {request.url.path}")
        return JSONResponse(
            status_code=499,  # Client Closed Request
            content={"detail": "Client disconnected"}
        )
    except Exception as e:
        logger.error(f"Error processing request {request.url.path}: {str(e)}")
        raise

# Configure CORS with explicit methods and headers
# Added LAST so it runs FIRST and handles preflight OPTIONS requests properly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development - restrict in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],  # Allow all headers
)

# Include routers
app.include_router(
    auth.router,
    prefix=f"{settings.API_V1_PREFIX}/auth",
    tags=["Authentication"],
)

app.include_router(
    measurements.router,
    prefix=f"{settings.API_V1_PREFIX}/measurements",
    tags=["Measurements"],
)

app.include_router(
    brands.router,
    prefix=f"{settings.API_V1_PREFIX}/brands",
    tags=["Brands"],
)

app.include_router(
    products.router,
    prefix=f"{settings.API_V1_PREFIX}",
    tags=["Products"],
)

app.include_router(
    webhooks.router,
    prefix=f"{settings.API_V1_PREFIX}/webhooks",
    tags=["Webhooks"],
)

app.include_router(
    batch.router,
    prefix=f"{settings.API_V1_PREFIX}/batch",
    tags=["Batch Processing"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "FitWhisperer API",
        "version": settings.VERSION,
        "docs": "/docs",
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/stats")
async def get_system_stats():
    """
    Get system statistics including cache and model performance.

    Returns:
    - cache_stats: Hit/miss rates for measurement and recommendation caches
    - model_status: Which ML models are loaded
    - rate_limit_info: Current rate limit configuration
    """
    try:
        from app.core.cache import get_all_cache_stats
        cache_stats = get_all_cache_stats()
    except Exception:
        cache_stats = {"error": "Cache module not available"}

    try:
        from app.ml.model_manager import get_model_stats
        model_stats = get_model_stats()
    except Exception:
        model_stats = {"error": "Model manager not available"}

    return {
        "status": "healthy",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "cache_stats": cache_stats,
        "model_stats": model_stats,
        "rate_limit": {
            "requests_per_minute": settings.RATE_LIMIT_PER_MINUTE,
            "burst_limit": settings.RATE_LIMIT_BURST,
        },
        "debug_mode": settings.DEBUG,
    }


if __name__ == "__main__":
    import uvicorn

    # Get number of workers from environment or use default
    # Note: workers > 1 requires running without reload
    num_workers = int(os.environ.get("UVICORN_WORKERS", "1"))

    # In debug/development mode, use single worker with reload
    # In production mode, use multiple workers without reload
    if settings.DEBUG:
        logger.info("Starting in DEVELOPMENT mode (single worker with reload)")
        uvicorn.run(
            "app.main:app",  # Use string reference for reload
            host="0.0.0.0",
            port=8000,
            reload=True,
            timeout_keep_alive=30,
        )
    else:
        logger.info(f"Starting in PRODUCTION mode ({num_workers} workers)")
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            workers=num_workers,
            timeout_keep_alive=30,
            access_log=True,
        )

    # ===========================================================================
    # PRODUCTION DEPLOYMENT OPTIONS:
    # ===========================================================================
    #
    # Option 1: Run with multiple uvicorn workers (recommended for moderate load)
    #   UVICORN_WORKERS=4 python -m app.main
    #
    # Option 2: Use Gunicorn with uvicorn workers (recommended for production)
    #   gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
    #
    # Option 3: Use environment variable
    #   export UVICORN_WORKERS=4
    #   python -m app.main
    #
    # ===========================================================================
