from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timedelta
from typing import List

from app.core.database import get_db
from app.core.auth import get_current_brand_by_api_key
from app.models import Brand, Measurement, Product, SubscriptionTier
from app.schemas import (
    BrandResponse,
    UsageStats,
    ProductCreate,
    ProductResponse,
    Analytics,
    PopularProduct,
    RevenueImpact,
    DailyMeasurementCount,
    MeasurementTrend,
    SizeDistributionOverTime,
    MeasurementHistoryItem,
    MeasurementHistoryResponse,
)

router = APIRouter()


@router.get("/me", response_model=BrandResponse)
async def get_brand_profile(
    brand: Brand = Depends(get_current_brand_by_api_key),
):
    """
    Get brand profile information

    **Authentication**: Use X-API-Key header (query param deprecated)
    """
    return BrandResponse.model_validate(brand)


@router.get("/usage", response_model=UsageStats)
async def get_usage_stats(
    brand: Brand = Depends(get_current_brand_by_api_key),
    db: Session = Depends(get_db),
):
    """
    Get API usage statistics

    **Authentication**: Use X-API-Key header (query param deprecated)
    """

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
    brand: Brand = Depends(get_current_brand_by_api_key),
    db: Session = Depends(get_db),
):
    """
    Get analytics and ROI metrics

    **Authentication**: Use X-API-Key header (query param deprecated)
    """

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
    brand: Brand = Depends(get_current_brand_by_api_key),
    db: Session = Depends(get_db),
):
    """
    Add a new product with size chart

    **Authentication**: Use X-API-Key header (query param deprecated)
    """

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
    brand: Brand = Depends(get_current_brand_by_api_key),
    db: Session = Depends(get_db),
):
    """
    Get all products for the brand

    **Authentication**: Use X-API-Key header (query param deprecated)
    """

    products = db.query(Product).filter(Product.brand_id == brand.id).all()

    return [ProductResponse.model_validate(p) for p in products]


@router.get("/history", response_model=MeasurementHistoryResponse)
async def get_measurement_history(
    days: int = Query(30, description="Number of days to look back", ge=1, le=365),
    brand: Brand = Depends(get_current_brand_by_api_key),
    db: Session = Depends(get_db),
):
    """
    Get measurement history with daily aggregations and trends

    **Authentication**: Use X-API-Key header (query param deprecated)
    """

    # Calculate date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    previous_start = start_date - timedelta(days=days)

    # Get all measurements in the period
    measurements = db.query(Measurement).filter(
        Measurement.brand_id == brand.id,
        Measurement.created_at >= start_date,
        Measurement.created_at <= end_date,
    ).order_by(Measurement.created_at.desc()).all()

    # Get previous period measurements for trend calculation
    previous_measurements = db.query(Measurement).filter(
        Measurement.brand_id == brand.id,
        Measurement.created_at >= previous_start,
        Measurement.created_at < start_date,
    ).all()

    # Aggregate by day
    daily_data = {}
    for m in measurements:
        date_key = m.created_at.strftime("%Y-%m-%d")
        if date_key not in daily_data:
            daily_data[date_key] = {
                "count": 0,
                "total_confidence": 0,
                "confidence_count": 0,
                "total_time": 0,
                "size_distribution": {}
            }

        daily_data[date_key]["count"] += 1
        daily_data[date_key]["total_time"] += m.processing_time_ms

        # Aggregate confidence scores
        if m.confidence_scores:
            for score in m.confidence_scores.values():
                daily_data[date_key]["total_confidence"] += score
                daily_data[date_key]["confidence_count"] += 1

        # Track size distribution
        if m.recommended_size:
            size = m.recommended_size
            daily_data[date_key]["size_distribution"][size] = \
                daily_data[date_key]["size_distribution"].get(size, 0) + 1

    # Build daily counts list
    daily_counts = []
    size_distribution_over_time = []

    # Fill in all days (even empty ones)
    current = start_date
    while current <= end_date:
        date_key = current.strftime("%Y-%m-%d")
        if date_key in daily_data:
            data = daily_data[date_key]
            avg_conf = data["total_confidence"] / data["confidence_count"] if data["confidence_count"] > 0 else 0
            avg_time = data["total_time"] / data["count"] if data["count"] > 0 else 0
            daily_counts.append(DailyMeasurementCount(
                date=date_key,
                count=data["count"],
                avg_confidence=avg_conf,
                avg_processing_time=avg_time,
            ))
            size_distribution_over_time.append(SizeDistributionOverTime(
                date=date_key,
                size_distribution=data["size_distribution"],
            ))
        else:
            daily_counts.append(DailyMeasurementCount(
                date=date_key,
                count=0,
                avg_confidence=0,
                avg_processing_time=0,
            ))
        current += timedelta(days=1)

    # Calculate trends
    def calculate_trend(current_val: float, previous_val: float) -> tuple:
        if previous_val == 0:
            return (0.0, "stable") if current_val == 0 else (100.0, "up")
        change = ((current_val - previous_val) / previous_val) * 100
        trend = "up" if change > 5 else ("down" if change < -5 else "stable")
        return (change, trend)

    current_count = len(measurements)
    previous_count = len(previous_measurements)
    count_change, count_trend = calculate_trend(current_count, previous_count)

    # Calculate average confidence for both periods
    def calc_avg_confidence(measure_list):
        total, count = 0, 0
        for m in measure_list:
            if m.confidence_scores:
                for score in m.confidence_scores.values():
                    total += score
                    count += 1
        return total / count if count > 0 else 0

    current_avg_conf = calc_avg_confidence(measurements)
    previous_avg_conf = calc_avg_confidence(previous_measurements)
    conf_change, conf_trend = calculate_trend(current_avg_conf * 100, previous_avg_conf * 100)

    # Calculate average processing time
    def calc_avg_time(measure_list):
        if not measure_list:
            return 0
        return sum(m.processing_time_ms for m in measure_list) / len(measure_list)

    current_avg_time = calc_avg_time(measurements)
    previous_avg_time = calc_avg_time(previous_measurements)
    time_change, time_trend = calculate_trend(current_avg_time, previous_avg_time)
    # For processing time, down is good
    time_trend = "up" if time_change > 5 else ("down" if time_change < -5 else "stable")

    trends = [
        MeasurementTrend(
            metric="Measurements",
            current_value=float(current_count),
            previous_value=float(previous_count),
            change_percentage=count_change,
            trend=count_trend,
        ),
        MeasurementTrend(
            metric="Avg Confidence",
            current_value=current_avg_conf * 100,
            previous_value=previous_avg_conf * 100,
            change_percentage=conf_change,
            trend=conf_trend,
        ),
        MeasurementTrend(
            metric="Avg Processing Time",
            current_value=current_avg_time,
            previous_value=previous_avg_time,
            change_percentage=time_change,
            trend=time_trend,
        ),
    ]

    # Recent measurements (last 20)
    recent_measurements = [
        MeasurementHistoryItem(
            id=str(m.id),
            created_at=m.created_at,
            shoulder_width=m.shoulder_width,
            chest_width=m.chest_width,
            waist_width=m.waist_width,
            hip_width=m.hip_width,
            inseam=m.inseam,
            arm_length=m.arm_length,
            recommended_size=m.recommended_size,
            confidence_scores=m.confidence_scores or {},
            processing_time_ms=m.processing_time_ms,
        )
        for m in measurements[:20]
    ]

    return MeasurementHistoryResponse(
        daily_counts=daily_counts,
        trends=trends,
        size_distribution_over_time=size_distribution_over_time,
        recent_measurements=recent_measurements,
        total_count=current_count,
        period_start=start_date,
        period_end=end_date,
    )
