from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional
from uuid import UUID

from app.models.brand import SubscriptionTier


class BrandBase(BaseModel):
    name: str
    email: EmailStr


class BrandCreate(BrandBase):
    password: str


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
