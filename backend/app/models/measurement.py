from sqlalchemy import Column, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.core.database import Base


class Measurement(Base):
    __tablename__ = "measurements"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    brand_id = Column(UUID(as_uuid=True), ForeignKey("brands.id"), nullable=False)
    product_id = Column(UUID(as_uuid=True), ForeignKey("products.id", ondelete="SET NULL"), nullable=True)

    # Body measurements in cm
    shoulder_width = Column(Float, nullable=False)
    chest_width = Column(Float, nullable=False)
    waist_width = Column(Float, nullable=False)
    hip_width = Column(Float, nullable=False)
    inseam = Column(Float, nullable=False)
    arm_length = Column(Float, nullable=False)

    # Confidence scores (0-1)
    confidence_scores = Column(JSON, nullable=False)

    # Size recommendation
    recommended_size = Column(String, nullable=True)
    size_probabilities = Column(JSON, nullable=True)

    # Processing metadata
    processing_time_ms = Column(Float, nullable=False)
    image_hash = Column(String, nullable=True)

    # ML enhancement tracking (NEW)
    used_ml_ratios = Column(JSON, nullable=True)  # Stores: {used: bool, method: str, confidence: float, body_shape: str, bmi: float}

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    brand = relationship("Brand", back_populates="measurements")
    product = relationship("Product", back_populates="measurements")
