import asyncio
import hashlib
import hmac
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import httpx

from sqlalchemy.orm import Session

from app.models import Webhook, WebhookDelivery

logger = logging.getLogger(__name__)

# Thread pool for async webhook delivery
_webhook_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="webhook_worker")


def generate_signature(payload: str, secret: str) -> str:
    """Generate HMAC-SHA256 signature for webhook payload"""
    return hmac.new(
        secret.encode('utf-8'),
        payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()


async def send_webhook(
    webhook: Webhook,
    event_type: str,
    payload: Dict[str, Any],
    db: Session,
) -> bool:
    """Send a webhook notification"""
    start_time = datetime.utcnow()
    payload_json = json.dumps(payload, default=str)

    # Create delivery record
    delivery = WebhookDelivery(
        webhook_id=webhook.id,
        event_type=event_type,
        payload=payload,
        status="pending",
    )
    db.add(delivery)
    db.commit()

    try:
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "FitWhisperer-Webhook/1.0",
            "X-Webhook-Event": event_type,
            "X-Webhook-Delivery": str(delivery.id),
            "X-Webhook-Timestamp": start_time.isoformat(),
        }

        # Add signature if secret is set
        if webhook.secret:
            signature = generate_signature(payload_json, webhook.secret)
            headers["X-Webhook-Signature"] = f"sha256={signature}"

        # Send the webhook
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                str(webhook.url),
                content=payload_json,
                headers=headers,
            )

        end_time = datetime.utcnow()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        # Update delivery record
        delivery.status = "success" if response.is_success else "failed"
        delivery.response_status = response.status_code
        delivery.response_body = response.text[:1000] if response.text else None
        delivery.delivered_at = end_time
        delivery.duration_ms = duration_ms

        if not response.is_success:
            delivery.error_message = f"HTTP {response.status_code}"

        # Update webhook statistics
        webhook.total_deliveries += 1
        webhook.last_delivery_at = end_time
        if response.is_success:
            webhook.successful_deliveries += 1
            webhook.last_delivery_status = "success"
            webhook.last_error = None
        else:
            webhook.failed_deliveries += 1
            webhook.last_delivery_status = "failed"
            webhook.last_error = f"HTTP {response.status_code}"

        db.commit()

        logger.info(f"Webhook delivered to {webhook.url}: status={response.status_code}")
        return response.is_success

    except Exception as e:
        end_time = datetime.utcnow()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        # Update delivery record with error
        delivery.status = "failed"
        delivery.error_message = str(e)[:500]
        delivery.delivered_at = end_time
        delivery.duration_ms = duration_ms

        # Update webhook statistics
        webhook.total_deliveries += 1
        webhook.failed_deliveries += 1
        webhook.last_delivery_at = end_time
        webhook.last_delivery_status = "failed"
        webhook.last_error = str(e)[:200]

        db.commit()

        logger.error(f"Webhook delivery failed to {webhook.url}: {e}")
        return False


async def trigger_webhooks(
    db: Session,
    brand_id: str,
    event_type: str,
    payload: Dict[str, Any],
) -> None:
    """Trigger all active webhooks for a brand for a specific event type"""
    # Find all active webhooks for this brand that subscribe to this event
    webhooks = db.query(Webhook).filter(
        Webhook.brand_id == brand_id,
        Webhook.is_active == True,
    ).all()

    # Filter webhooks that are subscribed to this event type
    relevant_webhooks = [
        w for w in webhooks
        if event_type in (w.events or [])
    ]

    if not relevant_webhooks:
        logger.debug(f"No webhooks found for brand {brand_id} event {event_type}")
        return

    # Send webhooks concurrently
    tasks = [
        send_webhook(webhook, event_type, payload, db)
        for webhook in relevant_webhooks
    ]

    await asyncio.gather(*tasks, return_exceptions=True)


async def test_webhook(webhook: Webhook) -> Dict[str, Any]:
    """Test a webhook by sending a test payload"""
    start_time = datetime.utcnow()

    test_payload = {
        "event": "webhook.test",
        "timestamp": start_time.isoformat(),
        "message": "This is a test webhook delivery",
    }
    payload_json = json.dumps(test_payload)

    try:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "FitWhisperer-Webhook/1.0",
            "X-Webhook-Event": "webhook.test",
            "X-Webhook-Timestamp": start_time.isoformat(),
        }

        if webhook.secret:
            signature = generate_signature(payload_json, webhook.secret)
            headers["X-Webhook-Signature"] = f"sha256={signature}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                str(webhook.url),
                content=payload_json,
                headers=headers,
            )

        end_time = datetime.utcnow()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        return {
            "success": response.is_success,
            "status_code": response.status_code,
            "response_time_ms": duration_ms,
            "error": None if response.is_success else f"HTTP {response.status_code}",
        }

    except Exception as e:
        end_time = datetime.utcnow()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        return {
            "success": False,
            "status_code": None,
            "response_time_ms": duration_ms,
            "error": str(e),
        }


def get_webhook_deliveries(
    db: Session,
    webhook_id: str,
    limit: int = 20,
) -> List[WebhookDelivery]:
    """Get recent delivery attempts for a webhook"""
    return db.query(WebhookDelivery).filter(
        WebhookDelivery.webhook_id == webhook_id
    ).order_by(WebhookDelivery.created_at.desc()).limit(limit).all()
