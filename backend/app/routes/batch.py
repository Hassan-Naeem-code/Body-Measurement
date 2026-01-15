from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, BackgroundTasks, status
from sqlalchemy.orm import Session
from typing import List, Dict, Optional
from datetime import datetime
import uuid
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from app.core.database import get_db
from app.models import Brand
from app.schemas.batch import (
    BatchStatus,
    BatchImageResult,
    BatchJobResponse,
    BatchJobSummary,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory storage for batch jobs (in production, use Redis/database)
_batch_jobs: Dict[str, dict] = {}

# Thread pool for batch processing
_batch_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="batch_worker")


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


async def process_batch_images(
    batch_id: str,
    images: List[tuple],  # List of (filename, content)
    api_key: str,
    webhook_url: Optional[str],
):
    """Background task to process batch images"""
    from app.routes.measurements import process_image_sync

    job = _batch_jobs.get(batch_id)
    if not job:
        return

    job["status"] = BatchStatus.PROCESSING
    job["started_at"] = datetime.utcnow()

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

    # Determine final status
    if job["successful_images"] == job["total_images"]:
        job["status"] = BatchStatus.COMPLETED
    elif job["successful_images"] == 0:
        job["status"] = BatchStatus.FAILED
    else:
        job["status"] = BatchStatus.PARTIAL

    job["completed_at"] = datetime.utcnow()
    job["total_processing_time_ms"] = total_time

    # Send webhook notification if configured
    if webhook_url:
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
            logger.error(f"Failed to send batch webhook: {e}")

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
    _batch_jobs[batch_id] = job

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

    job = _batch_jobs.get(batch_id)
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
    brand_jobs = [
        job for job in _batch_jobs.values()
        if job["brand_id"] == brand_id
    ]

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

    job = _batch_jobs.get(batch_id)
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

    del _batch_jobs[batch_id]
