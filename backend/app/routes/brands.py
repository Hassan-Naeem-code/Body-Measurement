from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timedelta
from typing import List

from app.core.database import get_db
from app.models import Brand, Measurement, Product, SubscriptionTier
from app.schemas import (
    BrandResponse,
    UsageStats,
    ProductCreate,
    ProductResponse,
    Analytics,
    PopularProduct,
    RevenueImpact,
)

router = APIRouter()


def get_brand_by_api_key(api_key: str, db: Session) -> Brand:
    """Dependency to get brand from API key"""
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


@router.get("/me", response_model=BrandResponse)
async def get_brand_profile(
    api_key: str = Query(..., description="API key"),
    db: Session = Depends(get_db),
):
    """Get brand profile information"""
    brand = get_brand_by_api_key(api_key, db)
    return BrandResponse.model_validate(brand)


@router.get("/usage", response_model=UsageStats)
async def get_usage_stats(
    api_key: str = Query(..., description="API key"),
    db: Session = Depends(get_db),
):
    """Get API usage statistics"""
    brand = get_brand_by_api_key(api_key, db)

    # Total requests
    total_requests = db.query(func.count(Measurement.id)).filter(
        Measurement.brand_id == brand.id
    ).scalar() or 0

    # Requests today
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    requests_today = db.query(func.count(Measurement.id)).filter(
        Measurement.brand_id == brand.id,
        Measurement.created_at >= today_start,
    ).scalar() or 0

    # Requests this month
    month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    requests_this_month = db.query(func.count(Measurement.id)).filter(
        Measurement.brand_id == brand.id,
        Measurement.created_at >= month_start,
    ).scalar() or 0

    # Average processing time
    avg_time = db.query(func.avg(Measurement.processing_time_ms)).filter(
        Measurement.brand_id == brand.id
    ).scalar() or 0

    # Plan limits
    plan_limits = {
        SubscriptionTier.FREE: 1000,
        SubscriptionTier.STARTER: 10000,
        SubscriptionTier.PROFESSIONAL: 50000,
        SubscriptionTier.ENTERPRISE: 1000000,
    }

    return UsageStats(
        total_requests=total_requests,
        requests_today=requests_today,
        requests_this_month=requests_this_month,
        average_processing_time=float(avg_time),
        plan_limit=plan_limits.get(brand.subscription_tier, 1000),
    )


@router.get("/analytics", response_model=Analytics)
async def get_analytics(
    api_key: str = Query(..., description="API key"),
    db: Session = Depends(get_db),
):
    """Get analytics and ROI metrics"""
    brand = get_brand_by_api_key(api_key, db)

    # Total measurements
    total_measurements = db.query(func.count(Measurement.id)).filter(
        Measurement.brand_id == brand.id
    ).scalar() or 0

    # Size distribution
    measurements = db.query(Measurement.recommended_size).filter(
        Measurement.brand_id == brand.id
    ).all()

    size_distribution = {}
    for m in measurements:
        if m.recommended_size:
            size_distribution[m.recommended_size] = size_distribution.get(m.recommended_size, 0) + 1

    # Average confidence
    all_measurements = db.query(Measurement).filter(
        Measurement.brand_id == brand.id
    ).all()

    total_confidence = 0
    count = 0
    for m in all_measurements:
        if m.confidence_scores:
            for score in m.confidence_scores.values():
                total_confidence += score
                count += 1

    average_confidence = total_confidence / count if count > 0 else 0

    # Popular products (mock data for now)
    products = db.query(Product).filter(Product.brand_id == brand.id).all()
    popular_products = [
        PopularProduct(product_name=p.name, measurement_count=0)
        for p in products[:5]
    ]

    # Revenue impact estimation
    estimated_conversions = int(total_measurements * 0.15)  # 15% conversion increase
    estimated_returns_prevented = int(total_measurements * 0.08)  # 8% return reduction
    roi_percentage = 250.0  # Average 250% ROI

    revenue_impact = RevenueImpact(
        estimated_conversions=estimated_conversions,
        estimated_returns_prevented=estimated_returns_prevented,
        roi_percentage=roi_percentage,
    )

    return Analytics(
        total_measurements=total_measurements,
        size_distribution=size_distribution,
        average_confidence=average_confidence,
        popular_products=popular_products,
        revenue_impact=revenue_impact,
    )


@router.post("/products", response_model=ProductResponse, status_code=status.HTTP_201_CREATED)
async def create_product(
    product_data: ProductCreate,
    api_key: str = Query(..., description="API key"),
    db: Session = Depends(get_db),
):
    """Add a new product with size chart"""
    brand = get_brand_by_api_key(api_key, db)

    new_product = Product(
        brand_id=brand.id,
        name=product_data.name,
        category=product_data.category,
        size_chart=product_data.size_chart,
    )

    db.add(new_product)
    db.commit()
    db.refresh(new_product)

    return ProductResponse.model_validate(new_product)


@router.get("/products", response_model=List[ProductResponse])
async def get_products(
    api_key: str = Query(..., description="API key"),
    db: Session = Depends(get_db),
):
    """Get all products for the brand"""
    brand = get_brand_by_api_key(api_key, db)

    products = db.query(Product).filter(Product.brand_id == brand.id).all()

    return [ProductResponse.model_validate(p) for p in products]
