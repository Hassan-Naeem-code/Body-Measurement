from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional
from datetime import datetime


# Event types that can trigger webhooks
WEBHOOK_EVENTS = [
    "measurement.completed",
    "measurement.failed",
    "batch.completed",
    "batch.failed",
]


class WebhookCreate(BaseModel):
    url: HttpUrl = Field(..., description="The URL to send webhook notifications to")
    secret: Optional[str] = Field(None, description="Secret for HMAC signature verification")
    events: List[str] = Field(
        default=["measurement.completed"],
        description="List of event types to subscribe to"
    )
    description: Optional[str] = Field(None, description="Optional description for this webhook")


class WebhookUpdate(BaseModel):
    url: Optional[HttpUrl] = None
    secret: Optional[str] = None
    events: Optional[List[str]] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None


class WebhookResponse(BaseModel):
    id: str
    url: str
    events: List[str]
    is_active: bool
    description: Optional[str]
    total_deliveries: int
    successful_deliveries: int
    failed_deliveries: int
    last_delivery_at: Optional[datetime]
    last_delivery_status: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class WebhookDeliveryResponse(BaseModel):
    id: str
    webhook_id: str
    event_type: str
    status: str
    response_status: Optional[int]
    error_message: Optional[str]
    created_at: datetime
    delivered_at: Optional[datetime]
    duration_ms: Optional[int]
    attempt_number: int

    class Config:
        from_attributes = True


class WebhookTestResponse(BaseModel):
    success: bool
    status_code: Optional[int]
    response_time_ms: int
    error: Optional[str]


# Webhook payload schemas
class MeasurementWebhookPayload(BaseModel):
    event: str = "measurement.completed"
    timestamp: datetime
    measurement_id: str
    brand_id: str
    total_people_detected: int
    valid_people_count: int
    processing_time_ms: float
    measurements: List[dict]  # PersonMeasurement data


class BatchWebhookPayload(BaseModel):
    event: str = "batch.completed"
    timestamp: datetime
    batch_id: str
    brand_id: str
    total_images: int
    successful_images: int
    failed_images: int
    total_processing_time_ms: float
