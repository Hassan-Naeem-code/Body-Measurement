from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import logging

from app.core.config import settings
from app.core.database import engine, Base
from app.routes import auth, measurements, brands, products

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create database tables
Base.metadata.create_all(bind=engine)

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="AI-powered body measurement API for e-commerce size recommendations",
    docs_url="/docs",
    redoc_url="/redoc",
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


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Body Measurement API",
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

    # Configure uvicorn with proper timeout settings
    # timeout_keep_alive: How long to keep idle connections open
    # timeout_notify: Time to wait for worker to complete request before killing
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        timeout_keep_alive=30,  # Keep-alive timeout in seconds
        # For production, use multiple workers:
        # workers=4  # Use gunicorn for production: gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
    )
