from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.database import engine, Base
from app.routes import auth, measurements, brands, products

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
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=settings.DEBUG)
