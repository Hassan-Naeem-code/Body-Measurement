from sqlalchemy import Column, String, DateTime, ForeignKey, JSON, Boolean, Text, Float, Integer, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import uuid

from app.core.database import Base


class Product(Base):
    """Enhanced product model with demographics and detailed metadata"""
    __tablename__ = "products"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    brand_id = Column(UUID(as_uuid=True), ForeignKey("brands.id", ondelete="CASCADE"), nullable=False)

    # Basic product info
    name = Column(String(255), nullable=False)
    sku = Column(String(100), nullable=True)
    category = Column(String(100), nullable=False)  # 'tops', 'bottoms', 'dresses', 'outerwear'
    subcategory = Column(String(100), nullable=True)  # 't-shirt', 'jeans', 'jacket', etc.

    # Demographic targeting (NEW - matches our demographic detection!)
    gender = Column(String(20), nullable=True)  # 'male', 'female', 'unisex'
    age_group = Column(String(20), nullable=True)  # 'adult', 'teen', 'child'

    # Additional info
    description = Column(Text, nullable=True)
    image_url = Column(String(500), nullable=True)
    is_active = Column(Boolean, default=True)

    # Legacy size chart (keep for backward compatibility)
    size_chart = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    brand = relationship("Brand", back_populates="products")
    size_charts = relationship("SizeChart", back_populates="product", cascade="all, delete-orphan")
    measurements = relationship("Measurement", back_populates="product")

    def __repr__(self):
        return f"<Product {self.name} ({self.category})>"


class SizeChart(Base):
    """Detailed size chart with min/max ranges for each measurement"""
    __tablename__ = "size_charts"
    __table_args__ = (
        UniqueConstraint('product_id', 'size_name', name='unique_product_size'),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    product_id = Column(UUID(as_uuid=True), ForeignKey("products.id", ondelete="CASCADE"), nullable=False)

    # Size name
    size_name = Column(String(20), nullable=False)  # 'XS', 'S', 'M', 'L', 'XL', etc.

    # Circumference measurement ranges (in cm) - matches our CircumferenceMeasurements!
    chest_min = Column(Float, nullable=True)
    chest_max = Column(Float, nullable=True)
    waist_min = Column(Float, nullable=True)
    waist_max = Column(Float, nullable=True)
    hip_min = Column(Float, nullable=True)
    hip_max = Column(Float, nullable=True)

    # Additional measurements
    height_min = Column(Float, nullable=True)
    height_max = Column(Float, nullable=True)
    inseam_min = Column(Float, nullable=True)
    inseam_max = Column(Float, nullable=True)
    shoulder_width_min = Column(Float, nullable=True)
    shoulder_width_max = Column(Float, nullable=True)
    arm_length_min = Column(Float, nullable=True)
    arm_length_max = Column(Float, nullable=True)

    # Weight range (optional)
    weight_min = Column(Float, nullable=True)
    weight_max = Column(Float, nullable=True)

    # Fit type (NEW - for fit preferences!)
    fit_type = Column(String(20), default='regular')  # 'tight', 'regular', 'loose'

    # Display order
    display_order = Column(Integer, default=0)

    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationship
    product = relationship("Product", back_populates="size_charts")

    def __repr__(self):
        return f"<SizeChart {self.size_name} for {self.product_id}>"

    def matches_measurements(self, chest=None, waist=None, hip=None, height=None) -> float:
        """
        Calculate how well measurements fit this size
        Returns a score: lower = better fit (0 = perfect)
        """
        score = 0.0
        count = 0

        # Check each measurement if provided
        if chest is not None and self.chest_min and self.chest_max:
            if self.chest_min <= chest <= self.chest_max:
                score += 0  # Perfect fit
            else:
                # Distance from range
                distance = min(abs(chest - self.chest_min), abs(chest - self.chest_max))
                score += distance * 2.0  # Weight: 2x (chest is important)
            count += 2.0

        if waist is not None and self.waist_min and self.waist_max:
            if self.waist_min <= waist <= self.waist_max:
                score += 0
            else:
                distance = min(abs(waist - self.waist_min), abs(waist - self.waist_max))
                score += distance * 1.5  # Weight: 1.5x
            count += 1.5

        if hip is not None and self.hip_min and self.hip_max:
            if self.hip_min <= hip <= self.hip_max:
                score += 0
            else:
                distance = min(abs(hip - self.hip_min), abs(hip - self.hip_max))
                score += distance * 1.5  # Weight: 1.5x
            count += 1.5

        if height is not None and self.height_min and self.height_max:
            if self.height_min <= height <= self.height_max:
                score += 0
            else:
                distance = min(abs(height - self.height_min), abs(height - self.height_max))
                score += distance * 0.5  # Weight: 0.5x (height less important)
            count += 0.5

        # Return average weighted distance
        return score / max(count, 1)
