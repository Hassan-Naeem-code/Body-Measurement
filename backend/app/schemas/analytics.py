from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime


class PopularProduct(BaseModel):
    product_name: str
    measurement_count: int


class RevenueImpact(BaseModel):
    estimated_conversions: int
    estimated_returns_prevented: int
    roi_percentage: float


class Analytics(BaseModel):
    total_measurements: int
    size_distribution: Dict[str, int]
    average_confidence: float
    popular_products: List[PopularProduct]
    revenue_impact: RevenueImpact


# History Dashboard Schemas
class DailyMeasurementCount(BaseModel):
    date: str  # ISO date string
    count: int
    avg_confidence: float
    avg_processing_time: float


class MeasurementTrend(BaseModel):
    metric: str
    current_value: float
    previous_value: float
    change_percentage: float
    trend: str  # "up", "down", "stable"


class SizeDistributionOverTime(BaseModel):
    date: str
    size_distribution: Dict[str, int]


class MeasurementHistoryItem(BaseModel):
    id: str
    created_at: datetime
    shoulder_width: float
    chest_width: float
    waist_width: float
    hip_width: float
    inseam: float
    arm_length: float
    recommended_size: Optional[str]
    confidence_scores: Dict[str, float]
    processing_time_ms: float


class MeasurementHistoryResponse(BaseModel):
    daily_counts: List[DailyMeasurementCount]
    trends: List[MeasurementTrend]
    size_distribution_over_time: List[SizeDistributionOverTime]
    recent_measurements: List[MeasurementHistoryItem]
    total_count: int
    period_start: datetime
    period_end: datetime
