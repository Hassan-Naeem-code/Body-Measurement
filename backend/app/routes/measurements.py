from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, status
from sqlalchemy.orm import Session
import cv2
import numpy as np
import time
import tempfile
import os

from app.core.database import get_db
from app.core.config import settings
from app.models import Brand, Measurement
from app.schemas import MeasurementResponse
from app.ml import PoseDetector, MeasurementExtractor, SizeRecommender

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


@router.post("/process", response_model=MeasurementResponse)
async def process_measurement(
    file: UploadFile = File(...),
    api_key: str = Query(..., description="API key for authentication"),
    db: Session = Depends(get_db),
):
    """
    Process a body image and extract measurements

    - **file**: Full-body image (JPG, PNG, WEBP)
    - **api_key**: Your API key for authentication

    Returns body measurements and size recommendation
    """
    start_time = time.time()

    # Validate brand
    brand = get_brand_by_api_key(api_key, db)

    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Please upload JPG, PNG, or WEBP",
        )

    # Read image file
    contents = await file.read()

    # Check file size
    file_size_mb = len(contents) / (1024 * 1024)
    if file_size_mb > settings.MAX_IMAGE_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {settings.MAX_IMAGE_SIZE_MB}MB",
        )

    # Convert to numpy array
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not read image file",
        )

    try:
        # Step 1: Detect pose landmarks
        detector = PoseDetector()
        pose_landmarks = detector.detect_from_array(image)

        if pose_landmarks is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not detect body pose in image. Please ensure the image shows a full body in clear view.",
            )

        # Step 2: Extract measurements
        extractor = MeasurementExtractor(reference_height_cm=settings.DEFAULT_HEIGHT_CM)
        body_measurements = extractor.extract_measurements(pose_landmarks)

        # Step 3: Recommend size
        recommender = SizeRecommender()
        size_recommendation = recommender.recommend_size(body_measurements)

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Save measurement to database
        measurement_record = Measurement(
            brand_id=brand.id,
            shoulder_width=body_measurements.shoulder_width,
            chest_width=body_measurements.chest_width,
            waist_width=body_measurements.waist_width,
            hip_width=body_measurements.hip_width,
            inseam=body_measurements.inseam,
            arm_length=body_measurements.arm_length,
            confidence_scores=body_measurements.confidence_scores,
            recommended_size=size_recommendation.recommended_size,
            size_probabilities=size_recommendation.size_probabilities,
            processing_time_ms=processing_time_ms,
        )

        db.add(measurement_record)
        db.commit()
        db.refresh(measurement_record)

        # Return response
        return MeasurementResponse(
            shoulder_width=body_measurements.shoulder_width,
            chest_width=body_measurements.chest_width,
            waist_width=body_measurements.waist_width,
            hip_width=body_measurements.hip_width,
            inseam=body_measurements.inseam,
            arm_length=body_measurements.arm_length,
            confidence_scores=body_measurements.confidence_scores,
            recommended_size=size_recommendation.recommended_size,
            size_probabilities=size_recommendation.size_probabilities,
            processing_time_ms=processing_time_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image: {str(e)}",
        )
