"""
Product and Size Chart Schemas - Enhanced for Product-Specific Sizing
"""

from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict
from uuid import UUID
from datetime import datetime


# ===== Size Chart Schemas =====

class SizeChartBase(BaseModel):
    """Base schema for size chart"""
    size_name: str = Field(..., description="Size name (XS, S, M, L, XL, etc.)")

    # Circumference ranges (in cm)
    chest_min: Optional[float] = Field(None, ge=0, le=300, description="Minimum chest circumference in cm")
    chest_max: Optional[float] = Field(None, ge=0, le=300, description="Maximum chest circumference in cm")
    waist_min: Optional[float] = Field(None, ge=0, le=300, description="Minimum waist circumference in cm")
    waist_max: Optional[float] = Field(None, ge=0, le=300, description="Maximum waist circumference in cm")
    hip_min: Optional[float] = Field(None, ge=0, le=300, description="Minimum hip circumference in cm")
    hip_max: Optional[float] = Field(None, ge=0, le=300, description="Maximum hip circumference in cm")

    # Additional measurements
    height_min: Optional[float] = Field(None, ge=0, le=300, description="Minimum height in cm")
    height_max: Optional[float] = Field(None, ge=0, le=300, description="Maximum height in cm")
    inseam_min: Optional[float] = Field(None, ge=0, le=150, description="Minimum inseam in cm")
    inseam_max: Optional[float] = Field(None, ge=0, le=150, description="Maximum inseam in cm")
    shoulder_width_min: Optional[float] = Field(None, ge=0, le=100, description="Minimum shoulder width in cm")
    shoulder_width_max: Optional[float] = Field(None, ge=0, le=100, description="Maximum shoulder width in cm")
    arm_length_min: Optional[float] = Field(None, ge=0, le=120, description="Minimum arm length in cm")
    arm_length_max: Optional[float] = Field(None, ge=0, le=120, description="Maximum arm length in cm")

    # Weight range (optional)
    weight_min: Optional[float] = Field(None, ge=0, le=500, description="Minimum weight in kg")
    weight_max: Optional[float] = Field(None, ge=0, le=500, description="Maximum weight in kg")

    # Fit type
    fit_type: str = Field(default="regular", description="Fit type: tight, regular, loose")
    display_order: int = Field(default=0, ge=0, description="Display order for sorting")

    @model_validator(mode='after')
    def validate_min_max_ranges(self):
        """Validate that min values are less than or equal to max values"""
        measurement_pairs = [
            ('chest_min', 'chest_max', 'Chest'),
            ('waist_min', 'waist_max', 'Waist'),
            ('hip_min', 'hip_max', 'Hip'),
            ('height_min', 'height_max', 'Height'),
            ('inseam_min', 'inseam_max', 'Inseam'),
            ('shoulder_width_min', 'shoulder_width_max', 'Shoulder width'),
            ('arm_length_min', 'arm_length_max', 'Arm length'),
            ('weight_min', 'weight_max', 'Weight'),
        ]

        errors = []
        for min_field, max_field, name in measurement_pairs:
            min_val = getattr(self, min_field)
            max_val = getattr(self, max_field)
            if min_val is not None and max_val is not None:
                if min_val > max_val:
                    errors.append(f"{name} minimum ({min_val}) cannot be greater than maximum ({max_val})")

        if errors:
            raise ValueError('; '.join(errors))

        # Validate fit_type
        if self.fit_type not in ('tight', 'regular', 'loose'):
            raise ValueError("fit_type must be 'tight', 'regular', or 'loose'")

        return self


class SizeChartCreate(SizeChartBase):
    """Schema for creating a size chart"""
    pass


class SizeChartUpdate(BaseModel):
    """Schema for updating a size chart (all fields optional)"""
    size_name: Optional[str] = None
    chest_min: Optional[float] = None
    chest_max: Optional[float] = None
    waist_min: Optional[float] = None
    waist_max: Optional[float] = None
    hip_min: Optional[float] = None
    hip_max: Optional[float] = None
    height_min: Optional[float] = None
    height_max: Optional[float] = None
    inseam_min: Optional[float] = None
    inseam_max: Optional[float] = None
    shoulder_width_min: Optional[float] = None
    shoulder_width_max: Optional[float] = None
    arm_length_min: Optional[float] = None
    arm_length_max: Optional[float] = None
    weight_min: Optional[float] = None
    weight_max: Optional[float] = None
    fit_type: Optional[str] = None
    display_order: Optional[int] = None


class SizeChartResponse(SizeChartBase):
    """Schema for size chart response"""
    id: UUID
    product_id: UUID
    created_at: datetime

    class Config:
        from_attributes = True


# ===== Product Schemas =====

class ProductBase(BaseModel):
    """Base schema for product"""
    name: str = Field(..., min_length=1, max_length=255, description="Product name")
    sku: Optional[str] = Field(None, max_length=100, description="Product SKU")
    category: str = Field(..., description="Product category (tops, bottoms, dresses, outerwear)")
    subcategory: Optional[str] = Field(None, description="Product subcategory (t-shirt, jeans, etc.)")

    # Demographics (matches our AI detection!)
    gender: Optional[str] = Field(None, description="Target gender: male, female, unisex")
    age_group: Optional[str] = Field(None, description="Target age group: adult, teen, child")

    # Additional info
    description: Optional[str] = Field(None, description="Product description")
    image_url: Optional[str] = Field(None, max_length=500, description="Product image URL")
    is_active: bool = Field(default=True, description="Whether product is active")


class ProductCreate(ProductBase):
    """Schema for creating a product"""
    # Support for simple size chart format (backward compatible)
    size_chart: Optional[Dict] = Field(None, description="Legacy size chart format as JSON")


class ProductUpdate(BaseModel):
    """Schema for updating a product (all fields optional)"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    sku: Optional[str] = Field(None, max_length=100)
    category: Optional[str] = None
    subcategory: Optional[str] = None
    gender: Optional[str] = None
    age_group: Optional[str] = None
    description: Optional[str] = None
    image_url: Optional[str] = Field(None, max_length=500)
    is_active: Optional[bool] = None


class ProductResponse(ProductBase):
    """Schema for product response"""
    id: UUID
    brand_id: UUID
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ProductWithSizeCharts(ProductResponse):
    """Schema for product with its size charts"""
    size_charts: List[SizeChartResponse] = Field(default=[], description="Size charts for this product")

    class Config:
        from_attributes = True


class ProductListResponse(BaseModel):
    """Schema for paginated product list"""
    total: int
    products: List[ProductResponse]


# ===== Size Recommendation Request =====

class SizeRecommendationRequest(BaseModel):
    """Request for size recommendation with product"""
    product_id: UUID = Field(..., description="Product ID to get size recommendation for")

    # User measurements
    chest_circumference: Optional[float] = Field(None, description="Chest circumference in cm")
    waist_circumference: Optional[float] = Field(None, description="Waist circumference in cm")
    hip_circumference: Optional[float] = Field(None, description="Hip circumference in cm")
    height: Optional[float] = Field(None, description="Height in cm")

    # Fit preference (for Feature #2!)
    fit_preference: Optional[str] = Field(default="regular", description="Fit preference: tight, regular, loose")


class SizeRecommendationResponse(BaseModel):
    """Response for size recommendation"""
    recommended_size: str = Field(..., description="Recommended size")
    confidence: float = Field(..., description="Confidence score (0-1)")
    size_scores: Dict[str, float] = Field(..., description="Fit scores for all sizes (lower = better fit)")
    fit_quality: str = Field(..., description="Fit quality: perfect, good, acceptable, poor")
    alternative_sizes: Optional[List[str]] = Field(None, description="Alternative sizes to consider")

    # Helpful info
    product_name: str
    product_category: str
    fit_type: str
