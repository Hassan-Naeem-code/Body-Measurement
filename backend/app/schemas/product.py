from pydantic import BaseModel
from typing import Dict
from uuid import UUID
from datetime import datetime


class SizeChartMeasurement(BaseModel):
    chest: float = None
    waist: float = None
    hip: float = None
    inseam: float = None


class ProductBase(BaseModel):
    name: str
    category: str
    size_chart: Dict[str, Dict[str, float]]


class ProductCreate(ProductBase):
    pass


class ProductResponse(ProductBase):
    id: UUID
    brand_id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
