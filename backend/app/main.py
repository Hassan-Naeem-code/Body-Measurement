from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import asyncio
import logging
import os

from app.core.config import settings
from app.core.database import engine, Base
from app.routes import auth, measurements, brands, products, webhooks, batch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
