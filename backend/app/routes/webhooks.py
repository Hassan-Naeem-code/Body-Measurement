from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from typing import List
import secrets

from app.core.database import get_db
from app.models import Brand, Webhook
from app.schemas.webhook import (
    WebhookCreate,
    WebhookUpdate,
    WebhookResponse,
    WebhookDeliveryResponse,
    WebhookTestResponse,
    WEBHOOK_EVENTS,
)
from app.services.webhook_service import test_webhook, get_webhook_deliveries

router = APIRouter()


def get_brand_by_api_key(api_key: str, db: Session) -> Brand:
    """Dependency to get brand from API key"""
    brand = db.query(Brand).filter(Brand.api_key == api_key).first()
    if not brand:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    if not brand.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive",
        )
    return brand


@router.get("/events", response_model=List[str])
async def list_webhook_events():
    """List all available webhook event types"""
    return WEBHOOK_EVENTS


@router.get("", response_model=List[WebhookResponse])
async def list_webhooks(
    api_key: str = Query(..., description="API key"),
    db: Session = Depends(get_db),
):
    """List all webhooks for the brand"""
    brand = get_brand_by_api_key(api_key, db)

    webhooks = db.query(Webhook).filter(Webhook.brand_id == brand.id).all()

    return [
        WebhookResponse(
            id=str(w.id),
            url=str(w.url),
            events=w.events or [],
            is_active=w.is_active,
            description=w.description,
            total_deliveries=w.total_deliveries,
            successful_deliveries=w.successful_deliveries,
            failed_deliveries=w.failed_deliveries,
            last_delivery_at=w.last_delivery_at,
            last_delivery_status=w.last_delivery_status,
            created_at=w.created_at,
            updated_at=w.updated_at,
        )
        for w in webhooks
    ]


@router.post("", response_model=WebhookResponse, status_code=status.HTTP_201_CREATED)
async def create_webhook(
    webhook_data: WebhookCreate,
    api_key: str = Query(..., description="API key"),
    db: Session = Depends(get_db),
):
    """Create a new webhook"""
    brand = get_brand_by_api_key(api_key, db)

    # Validate events
    for event in webhook_data.events:
        if event not in WEBHOOK_EVENTS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid event type: {event}. Valid events are: {WEBHOOK_EVENTS}",
            )

    # Generate a random secret if not provided
    secret = webhook_data.secret or secrets.token_hex(32)

    webhook = Webhook(
        brand_id=brand.id,
        url=str(webhook_data.url),
        secret=secret,
        events=webhook_data.events,
        description=webhook_data.description,
    )

    db.add(webhook)
    db.commit()
    db.refresh(webhook)

    return WebhookResponse(
        id=str(webhook.id),
        url=str(webhook.url),
        events=webhook.events or [],
        is_active=webhook.is_active,
        description=webhook.description,
        total_deliveries=webhook.total_deliveries,
        successful_deliveries=webhook.successful_deliveries,
        failed_deliveries=webhook.failed_deliveries,
        last_delivery_at=webhook.last_delivery_at,
        last_delivery_status=webhook.last_delivery_status,
        created_at=webhook.created_at,
        updated_at=webhook.updated_at,
    )


@router.get("/{webhook_id}", response_model=WebhookResponse)
async def get_webhook(
    webhook_id: str,
    api_key: str = Query(..., description="API key"),
    db: Session = Depends(get_db),
):
    """Get a specific webhook"""
    brand = get_brand_by_api_key(api_key, db)

    webhook = db.query(Webhook).filter(
        Webhook.id == webhook_id,
        Webhook.brand_id == brand.id,
    ).first()

    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Webhook not found",
        )

    return WebhookResponse(
        id=str(webhook.id),
        url=str(webhook.url),
        events=webhook.events or [],
        is_active=webhook.is_active,
        description=webhook.description,
        total_deliveries=webhook.total_deliveries,
        successful_deliveries=webhook.successful_deliveries,
        failed_deliveries=webhook.failed_deliveries,
        last_delivery_at=webhook.last_delivery_at,
        last_delivery_status=webhook.last_delivery_status,
        created_at=webhook.created_at,
        updated_at=webhook.updated_at,
    )


@router.patch("/{webhook_id}", response_model=WebhookResponse)
async def update_webhook(
    webhook_id: str,
    webhook_data: WebhookUpdate,
    api_key: str = Query(..., description="API key"),
    db: Session = Depends(get_db),
):
    """Update a webhook"""
    brand = get_brand_by_api_key(api_key, db)

    webhook = db.query(Webhook).filter(
        Webhook.id == webhook_id,
        Webhook.brand_id == brand.id,
    ).first()

    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Webhook not found",
        )

    # Validate events if provided
    if webhook_data.events:
        for event in webhook_data.events:
            if event not in WEBHOOK_EVENTS:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid event type: {event}. Valid events are: {WEBHOOK_EVENTS}",
                )
        webhook.events = webhook_data.events

    if webhook_data.url is not None:
        webhook.url = str(webhook_data.url)
    if webhook_data.secret is not None:
        webhook.secret = webhook_data.secret
    if webhook_data.description is not None:
        webhook.description = webhook_data.description
    if webhook_data.is_active is not None:
        webhook.is_active = webhook_data.is_active

    db.commit()
    db.refresh(webhook)

    return WebhookResponse(
        id=str(webhook.id),
        url=str(webhook.url),
        events=webhook.events or [],
        is_active=webhook.is_active,
        description=webhook.description,
        total_deliveries=webhook.total_deliveries,
        successful_deliveries=webhook.successful_deliveries,
        failed_deliveries=webhook.failed_deliveries,
        last_delivery_at=webhook.last_delivery_at,
        last_delivery_status=webhook.last_delivery_status,
        created_at=webhook.created_at,
        updated_at=webhook.updated_at,
    )


@router.delete("/{webhook_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_webhook(
    webhook_id: str,
    api_key: str = Query(..., description="API key"),
    db: Session = Depends(get_db),
):
    """Delete a webhook"""
    brand = get_brand_by_api_key(api_key, db)

    webhook = db.query(Webhook).filter(
        Webhook.id == webhook_id,
        Webhook.brand_id == brand.id,
    ).first()

    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Webhook not found",
        )

    db.delete(webhook)
    db.commit()


@router.post("/{webhook_id}/test", response_model=WebhookTestResponse)
async def test_webhook_endpoint(
    webhook_id: str,
    api_key: str = Query(..., description="API key"),
    db: Session = Depends(get_db),
):
    """Send a test notification to a webhook"""
    brand = get_brand_by_api_key(api_key, db)

    webhook = db.query(Webhook).filter(
        Webhook.id == webhook_id,
        Webhook.brand_id == brand.id,
    ).first()

    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Webhook not found",
        )

    result = await test_webhook(webhook)

    return WebhookTestResponse(**result)


@router.get("/{webhook_id}/deliveries", response_model=List[WebhookDeliveryResponse])
async def list_webhook_deliveries(
    webhook_id: str,
    api_key: str = Query(..., description="API key"),
    limit: int = Query(20, ge=1, le=100, description="Number of deliveries to return"),
    db: Session = Depends(get_db),
):
    """List recent delivery attempts for a webhook"""
    brand = get_brand_by_api_key(api_key, db)

    # Verify webhook belongs to brand
    webhook = db.query(Webhook).filter(
        Webhook.id == webhook_id,
        Webhook.brand_id == brand.id,
    ).first()

    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Webhook not found",
        )

    deliveries = get_webhook_deliveries(db, webhook_id, limit)

    return [
        WebhookDeliveryResponse(
            id=str(d.id),
            webhook_id=str(d.webhook_id),
            event_type=d.event_type,
            status=d.status,
            response_status=d.response_status,
            error_message=d.error_message,
            created_at=d.created_at,
            delivered_at=d.delivered_at,
            duration_ms=d.duration_ms,
            attempt_number=d.attempt_number,
        )
        for d in deliveries
    ]
