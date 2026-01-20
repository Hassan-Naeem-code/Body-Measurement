from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, BackgroundTasks, status
from sqlalchemy.orm import Session
from typing import List, Dict, Optional
from datetime import datetime
from urllib.parse import urlparse
import ipaddress
import socket
import uuid
import asyncio
import logging
import json
import redis
from concurrent.futures import ThreadPoolExecutor

from app.core.database import get_db
from app.core.auth import get_brand_by_api_key
from app.core.config import settings
from app.models import Brand
from app.schemas.batch import (
    BatchStatus,
    BatchImageResult,
    BatchJobResponse,
    BatchJobSummary,
)

logger = logging.getLogger(__name__)

# Redis client for batch job storage
try:
    redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
    redis_client.ping()  # Test connection
    REDIS_AVAILABLE = True
    logger.info("Redis connection established for batch storage")
except Exception as e:
    redis_client = None
    REDIS_AVAILABLE = False
    logger.warning(f"Redis not available, falling back to in-memory storage: {e}")

# Batch job TTL in Redis (24 hours)
BATCH_JOB_TTL = 86400


class BatchStorage:
    """Storage abstraction for batch jobs - uses Redis if available, falls back to in-memory"""

    def __init__(self):
        self._memory_storage: Dict[str, dict] = {}

    def _serialize_job(self, job: dict) -> str:
        """Serialize job for Redis storage"""
        serializable = job.copy()
        # Convert non-serializable types
        if 'status' in serializable and isinstance(serializable['status'], BatchStatus):
            serializable['status'] = serializable['status'].value
        if 'started_at' in serializable and serializable['started_at']:
            serializable['started_at'] = serializable['started_at'].isoformat()
        if 'completed_at' in serializable and serializable['completed_at']:
            serializable['completed_at'] = serializable['completed_at'].isoformat()
        if 'created_at' in serializable and serializable['created_at']:
            serializable['created_at'] = serializable['created_at'].isoformat()
        # Convert BatchImageResult objects to dicts
        if 'results' in serializable:
            serializable['results'] = [
                r.model_dump() if hasattr(r, 'model_dump') else r
                for r in serializable['results']
            ]
        return json.dumps(serializable)

    def _deserialize_job(self, data: str) -> dict:
        """Deserialize job from Redis storage"""
        job = json.loads(data)
        # Convert status back to enum
        if 'status' in job:
            job['status'] = BatchStatus(job['status'])
        # Convert datetime strings back
        if 'started_at' in job and job['started_at']:
            job['started_at'] = datetime.fromisoformat(job['started_at'])
        if 'completed_at' in job and job['completed_at']:
            job['completed_at'] = datetime.fromisoformat(job['completed_at'])
        if 'created_at' in job and job['created_at']:
            job['created_at'] = datetime.fromisoformat(job['created_at'])
        return job

    def get(self, batch_id: str) -> Optional[dict]:
        """Get a batch job"""
        if REDIS_AVAILABLE and redis_client:
            try:
                data = redis_client.get(f"batch:{batch_id}")
                if data:
                    return self._deserialize_job(data)
                return None
            except Exception as e:
                logger.error(f"Redis get error: {e}")
                # Fall back to memory
                return self._memory_storage.get(batch_id)
        return self._memory_storage.get(batch_id)

    def set(self, batch_id: str, job: dict) -> None:
        """Store a batch job"""
        if REDIS_AVAILABLE and redis_client:
            try:
                redis_client.setex(
                    f"batch:{batch_id}",
                    BATCH_JOB_TTL,
                    self._serialize_job(job)
                )
                return
            except Exception as e:
                logger.error(f"Redis set error: {e}")
                # Fall back to memory
        self._memory_storage[batch_id] = job

    def update(self, batch_id: str, updates: dict) -> None:
        """Update a batch job"""
        job = self.get(batch_id)
        if job:
            job.update(updates)
            self.set(batch_id, job)

    def delete(self, batch_id: str) -> None:
        """Delete a batch job"""
        if REDIS_AVAILABLE and redis_client:
            try:
                redis_client.delete(f"batch:{batch_id}")
            except Exception as e:
                logger.error(f"Redis delete error: {e}")
        if batch_id in self._memory_storage:
            del self._memory_storage[batch_id]


# Create batch storage instance
batch_storage = BatchStorage()


def is_safe_url(url: str) -> bool:
    """
    Validate URL to prevent SSRF attacks.
    Blocks internal IPs, localhost, and private networks.
    """
    try:
        parsed = urlparse(url)

        # Only allow http and https
        if parsed.scheme not in ('http', 'https'):
            return False

        # Get hostname
        hostname = parsed.hostname
        if not hostname:
            return False

        # Block localhost variations
        if hostname in ('localhost', '127.0.0.1', '::1', '0.0.0.0'):
            return False

        # Resolve hostname to IP and check if it's private
        try:
            ip = ipaddress.ip_address(socket.gethostbyname(hostname))
            # Block private, loopback, link-local, and reserved IPs
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
        except (socket.gaierror, ValueError):
            # If we can't resolve, allow it (could be valid external domain)
            pass

        return True
    except Exception:
        return False

router = APIRouter()

# Thread pool for batch processing
_batch_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="batch_worker")


async def process_batch_images(
    batch_id: str,
    images: List[tuple],  # List of (filename, content)
    api_key: str,
    webhook_url: Optional[str],
):
    """Background task to process batch images"""
    from app.routes.measurements import process_image_sync

    job = batch_storage.get(batch_id)
    if not job:
        return

    job["status"] = BatchStatus.PROCESSING
    job["started_at"] = datetime.utcnow()
    batch_storage.set(batch_id, job)

    results = []
    total_time = 0

    for idx, (filename, content) in enumerate(images):
        try:
            logger.info(f"Batch {batch_id}: Processing image {idx + 1}/{len(images)}: {filename}")

            # Process image
            start_time = datetime.utcnow()
            result = await asyncio.get_event_loop().run_in_executor(
                _batch_executor,
                process_image_sync,
                content,
                api_key,
                True,  # use_ml_ratios
            )
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds() * 1000

            total_time += processing_time

            results.append(BatchImageResult(
                image_index=idx,
                filename=filename,
                status="success",
                total_people_detected=result.get("total_people_detected", 0),
                valid_people_count=result.get("valid_people_count", 0),
                measurements=result.get("measurements", []),
                processing_time_ms=processing_time,
            ))

            job["successful_images"] += 1

        except Exception as e:
            logger.error(f"Batch {batch_id}: Error processing image {filename}: {e}")
            results.append(BatchImageResult(
                image_index=idx,
                filename=filename,
                status="failed",
                error=str(e),
            ))
            job["failed_images"] += 1

        job["processed_images"] = idx + 1
        job["results"] = results
        # Update storage after each image
        batch_storage.set(batch_id, job)

    # Determine final status
    if job["successful_images"] == job["total_images"]:
        job["status"] = BatchStatus.COMPLETED
    elif job["successful_images"] == 0:
        job["status"] = BatchStatus.FAILED
    else:
        job["status"] = BatchStatus.PARTIAL

    job["completed_at"] = datetime.utcnow()
    job["total_processing_time_ms"] = total_time
    batch_storage.set(batch_id, job)

    # Send webhook notification if configured and URL is safe
    if webhook_url and is_safe_url(webhook_url):
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    webhook_url,
                    json={
                        "event": "batch.completed",
                        "batch_id": batch_id,
                        "status": job["status"].value,
                        "total_images": job["total_images"],
                        "successful_images": job["successful_images"],
                        "failed_images": job["failed_images"],
                        "total_processing_time_ms": total_time,
                    }
                )
        except Exception as e:
            logger.warning(f"Failed to send batch webhook: {e}")
    elif webhook_url:
        logger.warning(f"Batch {batch_id}: Webhook URL rejected (security validation failed)")

    logger.info(f"Batch {batch_id} completed: {job['successful_images']}/{job['total_images']} successful")


@router.post("", response_model=BatchJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_batch_job(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Multiple image files to process"),
    api_key: str = Query(..., description="API key"),
    webhook_url: Optional[str] = Query(None, description="Webhook URL for completion notification"),
    db: Session = Depends(get_db),
):
    """
    Create a batch processing job for multiple images.

    - Maximum 10 images per batch
    - Each image max 10MB
    - Processing happens asynchronously
    - Use the returned batch_id to check status
    """
    brand = get_brand_by_api_key(api_key, db)

    # Validate webhook URL (SSRF protection)
    if webhook_url and not is_safe_url(webhook_url):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid webhook URL. Must be a valid external HTTPS URL.",
        )

    # Validate batch size
    if len(files) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 10 images per batch",
        )

    if len(files) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one image is required",
        )

    # Read all files and validate
    images = []
    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File {file.filename} is not an image",
            )

        content = await file.read()

        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File {file.filename} exceeds 10MB limit",
            )

        images.append((file.filename or f"image_{len(images)}", content))

    # Create batch job
    batch_id = str(uuid.uuid4())
    job = {
        "batch_id": batch_id,
        "brand_id": str(brand.id),
        "status": BatchStatus.PENDING,
        "total_images": len(images),
        "processed_images": 0,
        "successful_images": 0,
        "failed_images": 0,
        "created_at": datetime.utcnow(),
        "started_at": None,
        "completed_at": None,
        "total_processing_time_ms": None,
        "results": [],
    }
    batch_storage.set(batch_id, job)

    # Start background processing
    background_tasks.add_task(
        process_batch_images,
        batch_id,
        images,
        api_key,
        webhook_url,
    )

    return BatchJobResponse(**job)


@router.get("/{batch_id}", response_model=BatchJobResponse)
async def get_batch_job(
    batch_id: str,
    api_key: str = Query(..., description="API key"),
    db: Session = Depends(get_db),
):
    """Get the status and results of a batch job"""
    brand = get_brand_by_api_key(api_key, db)

    job = batch_storage.get(batch_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Batch job not found",
        )

    # Verify ownership
    if job["brand_id"] != str(brand.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this batch job",
        )

    return BatchJobResponse(**job)


@router.get("", response_model=List[BatchJobSummary])
async def list_batch_jobs(
    api_key: str = Query(..., description="API key"),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """List recent batch jobs for the brand"""
    brand = get_brand_by_api_key(api_key, db)
    brand_id = str(brand.id)

    # Filter jobs for this brand
    # Note: With Redis, we scan all batch keys; with in-memory, use the storage dict
    all_jobs = []
    if REDIS_AVAILABLE and redis_client:
        try:
            # Scan for all batch keys
            cursor = 0
            while True:
                cursor, keys = redis_client.scan(cursor, match="batch:*", count=100)
                for key in keys:
                    data = redis_client.get(key)
                    if data:
                        job = batch_storage._deserialize_job(data)
                        if job.get("brand_id") == brand_id:
                            all_jobs.append(job)
                if cursor == 0:
                    break
        except Exception as e:
            logger.error(f"Redis scan error: {e}")
            all_jobs = [j for j in batch_storage._memory_storage.values() if j.get("brand_id") == brand_id]
    else:
        all_jobs = [j for j in batch_storage._memory_storage.values() if j.get("brand_id") == brand_id]

    brand_jobs = all_jobs

    # Sort by created_at descending
    brand_jobs.sort(key=lambda x: x["created_at"], reverse=True)

    # Limit results
    brand_jobs = brand_jobs[:limit]

    return [
        BatchJobSummary(
            batch_id=job["batch_id"],
            status=job["status"],
            total_images=job["total_images"],
            processed_images=job["processed_images"],
            successful_images=job["successful_images"],
            failed_images=job["failed_images"],
            created_at=job["created_at"],
            completed_at=job["completed_at"],
        )
        for job in brand_jobs
    ]


@router.delete("/{batch_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_batch_job(
    batch_id: str,
    api_key: str = Query(..., description="API key"),
    db: Session = Depends(get_db),
):
    """Delete a completed batch job"""
    brand = get_brand_by_api_key(api_key, db)

    job = batch_storage.get(batch_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Batch job not found",
        )

    # Verify ownership
    if job["brand_id"] != str(brand.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this batch job",
        )

    # Only allow deletion of completed/failed jobs
    if job["status"] in [BatchStatus.PENDING, BatchStatus.PROCESSING]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete a job that is still processing",
        )

    batch_storage.delete(batch_id)
