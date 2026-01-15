from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey, JSON, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.core.database import Base


class Webhook(Base):
    """Webhook configuration for brands to receive notifications"""
    __tablename__ = "webhooks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    brand_id = Column(UUID(as_uuid=True), ForeignKey("brands.id"), nullable=False)

    # Webhook configuration
    url = Column(String, nullable=False)
    secret = Column(String, nullable=True)  # For signature verification

    # Event types to subscribe to
    events = Column(JSON, nullable=False, default=["measurement.completed"])

    # Status
    is_active = Column(Boolean, default=True)

    # Metadata
    description = Column(String, nullable=True)

    # Statistics
    total_deliveries = Column(Integer, default=0)
    successful_deliveries = Column(Integer, default=0)
    failed_deliveries = Column(Integer, default=0)
    last_delivery_at = Column(DateTime, nullable=True)
    last_delivery_status = Column(String, nullable=True)  # "success", "failed"
    last_error = Column(String, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    brand = relationship("Brand", back_populates="webhooks")


class WebhookDelivery(Base):
    """Log of webhook delivery attempts"""
    __tablename__ = "webhook_deliveries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    webhook_id = Column(UUID(as_uuid=True), ForeignKey("webhooks.id"), nullable=False)

    # Event details
    event_type = Column(String, nullable=False)
    payload = Column(JSON, nullable=False)

    # Delivery status
    status = Column(String, nullable=False)  # "pending", "success", "failed"
    response_status = Column(Integer, nullable=True)  # HTTP status code
    response_body = Column(String, nullable=True)
    error_message = Column(String, nullable=True)

    # Timing
    created_at = Column(DateTime, default=datetime.utcnow)
    delivered_at = Column(DateTime, nullable=True)
    duration_ms = Column(Integer, nullable=True)

    # Retry information
    attempt_number = Column(Integer, default=1)
    next_retry_at = Column(DateTime, nullable=True)

    # Relationships
    webhook = relationship("Webhook")
