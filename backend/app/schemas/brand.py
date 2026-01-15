from pydantic import BaseModel, EmailStr, field_validator
from datetime import datetime
from typing import Optional
from uuid import UUID
import re

from app.models.brand import SubscriptionTier


class BrandBase(BaseModel):
    name: str
    email: EmailStr

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        """Validate brand name is not empty or whitespace"""
        if not v or not v.strip():
            raise ValueError('Brand name cannot be empty')
        if len(v.strip()) < 2:
            raise ValueError('Brand name must be at least 2 characters')
        if len(v) > 100:
            raise ValueError('Brand name cannot exceed 100 characters')
        return v.strip()


class BrandCreate(BrandBase):
    password: str

    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        """
        Validate password strength:
        - At least 8 characters
        - At least one uppercase letter
        - At least one lowercase letter
        - At least one digit
        - At least one special character
        """
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if len(v) > 128:
            raise ValueError('Password cannot exceed 128 characters')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>_\-+=\[\]\\;\'`~]', v):
            raise ValueError('Password must contain at least one special character (!@#$%^&*(),.?":{}|<>)')
        return v


class BrandLogin(BaseModel):
    email: EmailStr
    password: str


class BrandResponse(BrandBase):
    id: UUID
    api_key: str
    is_active: bool
    subscription_tier: SubscriptionTier
    created_at: datetime

    class Config:
        from_attributes = True


class BrandWithToken(BaseModel):
    access_token: str
    token_type: str = "bearer"
    brand: BrandResponse


class UsageStats(BaseModel):
    total_requests: int
    requests_today: int
    requests_this_month: int
    average_processing_time: float
    plan_limit: int
