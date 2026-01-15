from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class BatchStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # Some images succeeded, some failed


class BatchImageResult(BaseModel):
    """Result for a single image in a batch"""
    image_index: int
    filename: str
    status: str  # "success", "failed"
    error: Optional[str] = None
    total_people_detected: Optional[int] = None
    valid_people_count: Optional[int] = None
    measurements: Optional[List[Dict[str, Any]]] = None
    processing_time_ms: Optional[float] = None


class BatchJobCreate(BaseModel):
    """Request to create a batch job"""
    webhook_url: Optional[str] = Field(
        None,
        description="URL to receive webhook notification when batch completes"
    )


class BatchJobResponse(BaseModel):
    """Response for batch job status"""
    batch_id: str
    status: BatchStatus
    total_images: int
    processed_images: int
    successful_images: int
    failed_images: int
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_processing_time_ms: Optional[float] = None
    results: Optional[List[BatchImageResult]] = None

    class Config:
        from_attributes = True


class BatchJobSummary(BaseModel):
    """Summary of a batch job (without full results)"""
    batch_id: str
    status: BatchStatus
    total_images: int
    processed_images: int
    successful_images: int
    failed_images: int
    created_at: datetime
    completed_at: Optional[datetime] = None
