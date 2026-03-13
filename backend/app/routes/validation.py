"""
Ground Truth Validation API - CRUD for collecting real tape measurements.

These endpoints allow submitting, listing, and deleting ground truth data
used to train and validate the ML pipeline accuracy.
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, status
from sqlalchemy.orm import Session
from typing import Optional, List
import uuid
import os
import shutil
import logging

from app.core.database import get_db
from app.core.auth import get_current_brand_by_api_key
from app.models import Brand, GroundTruth
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)

# Directory for ground truth images
GROUND_TRUTH_IMAGES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "validation", "images"
)
os.makedirs(GROUND_TRUTH_IMAGES_DIR, exist_ok=True)


class GroundTruthCreate(BaseModel):
    actual_chest_circumference: Optional[float] = None
    actual_waist_circumference: Optional[float] = None
    actual_hip_circumference: Optional[float] = None
    actual_shoulder_width: Optional[float] = None
    actual_inseam: Optional[float] = None
    actual_arm_length: Optional[float] = None
    actual_height: Optional[float] = None
    gender: Optional[str] = None
    age_group: Optional[str] = None
    body_type: Optional[str] = None
    pose_type: Optional[str] = None
    image_quality: Optional[str] = None
    measured_by: Optional[str] = None
    measurement_date: Optional[str] = None
    notes: Optional[str] = None


class GroundTruthResponse(BaseModel):
    id: str
    image_path: str
    actual_chest_circumference: Optional[float] = None
    actual_waist_circumference: Optional[float] = None
    actual_hip_circumference: Optional[float] = None
    actual_shoulder_width: Optional[float] = None
    actual_inseam: Optional[float] = None
    actual_arm_length: Optional[float] = None
    actual_height: Optional[float] = None
    gender: Optional[str] = None
    age_group: Optional[str] = None
    body_type: Optional[str] = None
    pose_type: Optional[str] = None
    measured_by: Optional[str] = None
    notes: Optional[str] = None


@router.post("/ground-truth", response_model=GroundTruthResponse, status_code=status.HTTP_201_CREATED)
async def create_ground_truth(
    file: UploadFile = File(...),
    actual_chest_circumference: Optional[float] = Query(None),
    actual_waist_circumference: Optional[float] = Query(None),
    actual_hip_circumference: Optional[float] = Query(None),
    actual_shoulder_width: Optional[float] = Query(None),
    actual_inseam: Optional[float] = Query(None),
    actual_arm_length: Optional[float] = Query(None),
    actual_height: Optional[float] = Query(None),
    gender: Optional[str] = Query(None),
    age_group: Optional[str] = Query(None),
    body_type: Optional[str] = Query(None),
    pose_type: Optional[str] = Query(None, description="front, side, or angled"),
    image_quality: Optional[str] = Query(None, description="good, average, or poor"),
    measured_by: Optional[str] = Query(None),
    measurement_date: Optional[str] = Query(None),
    notes: Optional[str] = Query(None),
    brand: Brand = Depends(get_current_brand_by_api_key),
    db: Session = Depends(get_db),
):
    """
    Submit a ground truth sample with an image and tape measurements.

    At least one measurement must be provided. Upload the image as multipart/form-data
    and pass measurements as query parameters.
    """
    # Validate at least one measurement provided
    measurements = {
        "chest": actual_chest_circumference,
        "waist": actual_waist_circumference,
        "hip": actual_hip_circumference,
        "shoulder": actual_shoulder_width,
        "inseam": actual_inseam,
        "arm": actual_arm_length,
        "height": actual_height,
    }
    if not any(v is not None for v in measurements.values()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one actual measurement must be provided.",
        )

    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Please upload JPG, PNG, or WEBP.",
        )

    # Save image
    file_ext = file.filename.rsplit(".", 1)[-1] if file.filename else "jpg"
    image_filename = f"gt_{uuid.uuid4().hex[:12]}.{file_ext}"
    image_path = os.path.join(GROUND_TRUTH_IMAGES_DIR, image_filename)

    with open(image_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Create DB record
    gt = GroundTruth(
        image_path=image_path,
        actual_chest_circumference=actual_chest_circumference,
        actual_waist_circumference=actual_waist_circumference,
        actual_hip_circumference=actual_hip_circumference,
        actual_shoulder_width=actual_shoulder_width,
        actual_inseam=actual_inseam,
        actual_arm_length=actual_arm_length,
        actual_height=actual_height,
        gender=gender,
        age_group=age_group,
        body_type=body_type,
        pose_type=pose_type,
        image_quality=image_quality,
        measured_by=measured_by,
        measurement_date=measurement_date,
        notes=notes,
    )
    db.add(gt)
    db.commit()
    db.refresh(gt)

    logger.info("Ground truth sample created: %s", gt.id)

    return GroundTruthResponse(
        id=str(gt.id),
        image_path=gt.image_path,
        actual_chest_circumference=gt.actual_chest_circumference,
        actual_waist_circumference=gt.actual_waist_circumference,
        actual_hip_circumference=gt.actual_hip_circumference,
        actual_shoulder_width=gt.actual_shoulder_width,
        actual_inseam=gt.actual_inseam,
        actual_arm_length=gt.actual_arm_length,
        actual_height=gt.actual_height,
        gender=gt.gender,
        age_group=gt.age_group,
        body_type=gt.body_type,
        pose_type=gt.pose_type,
        measured_by=gt.measured_by,
        notes=gt.notes,
    )


@router.get("/ground-truth", response_model=List[GroundTruthResponse])
async def list_ground_truth(
    gender: Optional[str] = Query(None),
    body_type: Optional[str] = Query(None),
    limit: int = Query(100, le=500),
    offset: int = Query(0),
    brand: Brand = Depends(get_current_brand_by_api_key),
    db: Session = Depends(get_db),
):
    """List all ground truth samples, optionally filtered by gender or body type."""
    query = db.query(GroundTruth)
    if gender:
        query = query.filter(GroundTruth.gender == gender)
    if body_type:
        query = query.filter(GroundTruth.body_type == body_type)

    samples = query.order_by(GroundTruth.created_at.desc()).offset(offset).limit(limit).all()

    return [
        GroundTruthResponse(
            id=str(s.id),
            image_path=s.image_path,
            actual_chest_circumference=s.actual_chest_circumference,
            actual_waist_circumference=s.actual_waist_circumference,
            actual_hip_circumference=s.actual_hip_circumference,
            actual_shoulder_width=s.actual_shoulder_width,
            actual_inseam=s.actual_inseam,
            actual_arm_length=s.actual_arm_length,
            actual_height=s.actual_height,
            gender=s.gender,
            age_group=s.age_group,
            body_type=s.body_type,
            pose_type=s.pose_type,
            measured_by=s.measured_by,
            notes=s.notes,
        )
        for s in samples
    ]


@router.delete("/ground-truth/{gt_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_ground_truth(
    gt_id: str,
    brand: Brand = Depends(get_current_brand_by_api_key),
    db: Session = Depends(get_db),
):
    """Delete a ground truth sample and its image."""
    gt = db.query(GroundTruth).filter(GroundTruth.id == gt_id).first()
    if not gt:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ground truth sample not found.")

    # Remove image file
    if gt.image_path and os.path.exists(gt.image_path):
        os.remove(gt.image_path)

    db.delete(gt)
    db.commit()
    logger.info("Ground truth sample deleted: %s", gt_id)
