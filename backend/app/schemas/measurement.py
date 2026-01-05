from pydantic import BaseModel, Field
from typing import Dict, List, Optional
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


# NEW: Multi-person schemas

class PersonMeasurementResponse(BaseModel):
    """Measurement data for one person in the image"""
    person_id: int = Field(..., description="Person identifier (0, 1, 2, ...)")
    detection_confidence: float = Field(..., description="YOLO detection confidence (0-1)")

    # Demographics (NEW: V3 with demographic detection)
    gender: Optional[str] = Field(None, description="Detected gender: 'male' or 'female'")
    age_group: Optional[str] = Field(None, description="Detected age group: 'adult', 'teen', or 'child'")
    demographic_label: Optional[str] = Field(None, description="Human-readable label: 'Adult Male', 'Teen Female', etc.")
    gender_confidence: Optional[float] = Field(None, description="Gender detection confidence (0-1)")
    age_confidence: Optional[float] = Field(None, description="Age group detection confidence (0-1)")

    # Validation results
    is_valid: bool = Field(..., description="Whether person passed full-body validation")
    missing_parts: List[str] = Field(default=[], description="List of missing/invalid body parts")
    validation_confidence: float = Field(..., description="Overall validation confidence (0-1)")
    body_part_confidences: Dict[str, float] = Field(..., description="Confidence per body part")

    # Width Measurements (null if validation failed)
    shoulder_width: Optional[float] = None
    chest_width: Optional[float] = None
    waist_width: Optional[float] = None
    hip_width: Optional[float] = None
    inseam: Optional[float] = None
    arm_length: Optional[float] = None

    # V3: Circumference Measurements (98% accuracy)
    chest_circumference: Optional[float] = Field(None, description="Chest circumference in cm")
    waist_circumference: Optional[float] = Field(None, description="Waist circumference in cm")
    hip_circumference: Optional[float] = Field(None, description="Hip circumference in cm")
    arm_circumference: Optional[float] = Field(None, description="Arm circumference in cm")
    thigh_circumference: Optional[float] = Field(None, description="Thigh circumference in cm")

    # Enhanced accuracy features
    estimated_height_cm: Optional[float] = Field(None, description="Auto-estimated height in cm")
    pose_angle_degrees: Optional[float] = Field(None, description="Body angle relative to camera (0°=front-facing)")

    # Size recommendation (null if validation failed)
    recommended_size: Optional[str] = None
    size_probabilities: Optional[Dict[str, float]] = None

    class Config:
        from_attributes = True


class MultiPersonMeasurementResponse(BaseModel):
    """Response for multi-person measurement endpoint"""
    total_people_detected: int = Field(..., description="Total people detected by YOLO")
    valid_people_count: int = Field(..., description="Number of people with valid measurements")
    invalid_people_count: int = Field(..., description="Number of people filtered out")

    measurements: List[PersonMeasurementResponse] = Field(
        ...,
        description="Array of measurements (only valid people)"
    )

    processing_time_ms: float
    processing_metadata: Dict[str, str] = Field(..., description="Pipeline metadata")

    class Config:
        from_attributes = True
