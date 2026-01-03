from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import create_access_token, verify_password, get_password_hash, generate_api_key
from app.models import Brand
from app.schemas import BrandCreate, BrandLogin, BrandWithToken, BrandResponse

router = APIRouter()


@router.post("/register", response_model=BrandWithToken, status_code=status.HTTP_201_CREATED)
async def register_brand(brand_data: BrandCreate, db: Session = Depends(get_db)):
    """
    Register a new brand account
    """
    # Check if email already exists
    existing_brand = db.query(Brand).filter(Brand.email == brand_data.email).first()
    if existing_brand:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Create new brand
    hashed_password = get_password_hash(brand_data.password)
    api_key = generate_api_key()

    new_brand = Brand(
        name=brand_data.name,
        email=brand_data.email,
        hashed_password=hashed_password,
        api_key=api_key,
    )

    db.add(new_brand)
    db.commit()
    db.refresh(new_brand)

    # Create access token
    access_token = create_access_token(data={"sub": str(new_brand.id)})

    return BrandWithToken(
        access_token=access_token,
        brand=BrandResponse.model_validate(new_brand),
    )


@router.post("/login", response_model=BrandWithToken)
async def login_brand(credentials: BrandLogin, db: Session = Depends(get_db)):
    """
    Login with email and password
    """
    # Find brand by email
    brand = db.query(Brand).filter(Brand.email == credentials.email).first()

    if not brand or not verify_password(credentials.password, brand.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    if not brand.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive",
        )

    # Create access token
    access_token = create_access_token(data={"sub": str(brand.id)})

    return BrandWithToken(
        access_token=access_token,
        brand=BrandResponse.model_validate(brand),
    )
