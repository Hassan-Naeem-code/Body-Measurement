from sqlalchemy import Column, String, Float, DateTime, Text
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

from app.core.database import Base


class GroundTruth(Base):
    __tablename__ = "ground_truth"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Image reference
    image_path = Column(String, nullable=False)

    # Actual tape measurements (cm)
    actual_chest_circumference = Column(Float, nullable=True)
    actual_waist_circumference = Column(Float, nullable=True)
    actual_hip_circumference = Column(Float, nullable=True)
    actual_shoulder_width = Column(Float, nullable=True)
    actual_inseam = Column(Float, nullable=True)
    actual_arm_length = Column(Float, nullable=True)
    actual_height = Column(Float, nullable=True)

    # Demographics
    gender = Column(String, nullable=True)
    age_group = Column(String, nullable=True)
    body_type = Column(String, nullable=True)

    # Image metadata
    pose_type = Column(String, nullable=True)  # front, side, angled
    image_quality = Column(String, nullable=True)  # good, average, poor

    # Measurement metadata
    measured_by = Column(String, nullable=True)
    measurement_date = Column(String, nullable=True)
    notes = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
