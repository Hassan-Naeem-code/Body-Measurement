from pydantic import BaseModel
from typing import Dict, List


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
