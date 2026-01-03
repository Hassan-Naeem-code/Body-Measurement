from pydantic import BaseModel
from typing import Dict
from uuid import UUID
from datetime import datetime


class MeasurementResponse(BaseModel):
    shoulder_width: float
    chest_width: float
    waist_width: float
    hip_width: float
    inseam: float
    arm_length: float
    confidence_scores: Dict[str, float]
    recommended_size: str
    size_probabilities: Dict[str, float]
    processing_time_ms: float

    class Config:
        from_attributes = True


class MeasurementRecord(BaseModel):
    id: UUID
    brand_id: UUID
    shoulder_width: float
    chest_width: float
    waist_width: float
    hip_width: float
    inseam: float
    arm_length: float
    confidence_scores: Dict[str, float]
    recommended_size: str
    size_probabilities: Dict[str, float]
    processing_time_ms: float
    created_at: datetime

    class Config:
        from_attributes = True
