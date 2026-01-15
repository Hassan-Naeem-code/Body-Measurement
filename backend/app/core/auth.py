"""
Authentication utilities and dependency injection
"""

from fastapi import Depends, HTTPException, status, Header, Query
from fastapi.security import HTTPBearer, APIKeyHeader
from starlette.requests import Request
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from typing import Optional

from app.core.config import settings
from app.core.database import get_db
from app.models.brand import Brand

security = HTTPBearer()

# API Key header scheme - preferred method
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_brand_by_api_key(
    api_key: str,
    db: Session
) -> Brand:
    """Get brand from API key - shared helper function"""
    brand = db.query(Brand).filter(Brand.api_key == api_key).first()
    if not brand:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    if not brand.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive",
        )
    return brand


def get_current_brand_by_api_key(
    x_api_key: Optional[str] = Depends(api_key_header),
    api_key: Optional[str] = Query(None, description="API key (deprecated - use X-API-Key header)", deprecated=True),
    db: Session = Depends(get_db)
) -> Brand:
    """
    Dependency to get the current authenticated brand from API key.

    Accepts API key from:
    1. X-API-Key header (preferred, secure)
    2. api_key query parameter (deprecated, for backwards compatibility)

    The query parameter is deprecated and will log a warning.
    """
    # Prefer header over query param
    key = x_api_key or api_key

    if not key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Use X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Log deprecation warning if using query param
    if not x_api_key and api_key:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            "API key passed via query parameter is deprecated. "
            "Use X-API-Key header instead for better security."
        )

    return get_brand_by_api_key(key, db)


def get_current_brand(
    request: Request,
    db: Session = Depends(get_db)
) -> Brand:
    """
    Dependency to get the current authenticated brand from JWT token
    """
    # Get token from Authorization header
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = auth_header[7:]  # Remove "Bearer " prefix
    
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        brand_id: str = payload.get("sub")
        if brand_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get brand from database
    brand = db.query(Brand).filter(Brand.id == brand_id).first()
    if brand is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Brand not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not brand.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Brand account is inactive",
        )
    
    return brand
