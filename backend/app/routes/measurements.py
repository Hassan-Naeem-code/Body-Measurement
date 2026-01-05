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
from app.schemas import MeasurementResponse, MultiPersonMeasurementResponse, PersonMeasurementResponse
from app.ml import PoseDetector, MeasurementExtractor, SizeRecommender, MultiPersonProcessor, EnhancedMultiPersonProcessor
from app.ml.multi_person_processor_v3 import DepthBasedMultiPersonProcessor

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


@router.post("/process-multi", response_model=MultiPersonMeasurementResponse)
async def process_multi_person_measurement(
    file: UploadFile = File(...),
    api_key: str = Query(..., description="API key for authentication"),
    db: Session = Depends(get_db),
):
    """
    Process a body image and extract measurements for ALL people (MULTI-PERSON)

    **NEW ENDPOINT**: Supports multiple people in the same image.

    - **file**: Full-body image with one or more people (JPG, PNG, WEBP)
    - **api_key**: Your API key for authentication

    Returns:
    - Array of measurements (one per valid person)
    - Only people who pass full-body validation are included
    - Each person gets independent measurements and size recommendation

    Validation Requirements:
    - Head, shoulders, elbows, hands/wrists, stomach/torso, legs, feet/ankles must be visible
    - WHOLE human being must be visible from head to toes
    - Minimum confidence thresholds per body part
    - Human anatomy validation (not animals/objects)
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
        # Process all people in the image with DEPTH-BASED V3 processor (98% accuracy)
        processor = DepthBasedMultiPersonProcessor(
            yolo_model_size=settings.YOLO_MODEL_SIZE,
            yolo_confidence=settings.YOLO_CONFIDENCE_THRESHOLD,
            pose_confidence=settings.CONFIDENCE_THRESHOLD,
            custom_validation_thresholds={
                "head": settings.BODY_VALIDATION_HEAD_THRESHOLD,
                "shoulders": settings.BODY_VALIDATION_SHOULDERS_THRESHOLD,
                "elbows": settings.BODY_VALIDATION_ELBOWS_THRESHOLD,
                "hands": settings.BODY_VALIDATION_HANDS_THRESHOLD,
                "torso": settings.BODY_VALIDATION_TORSO_THRESHOLD,
                "legs": settings.BODY_VALIDATION_LEGS_THRESHOLD,
                "feet": settings.BODY_VALIDATION_FEET_THRESHOLD,
            }
        )

        result = processor.process_image(image)

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Handle case: no people detected
        if result.total_people_detected == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No people detected in the image. Please ensure the image contains visible people.",
            )

        # Handle case: people detected but none valid
        if result.valid_people_count == 0:
            # Provide helpful error message
            all_missing_parts = set()
            for pm in result.measurements:
                all_missing_parts.update(pm.validation_result.missing_parts)

            missing_parts_list = list(all_missing_parts)[:3]  # Top 3 issues
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Detected {result.total_people_detected} people, but none passed full-body validation. "
                    f"Common issues: {', '.join(missing_parts_list)}. "
                    f"Please ensure all people show WHOLE body: head, shoulders, elbows, hands, torso, legs, and feet."
                ),
            )

        # Convert to response format
        measurement_responses = []
        for pm in result.measurements:
            person_response = PersonMeasurementResponse(
                person_id=pm.person_id,
                detection_confidence=pm.detection_confidence,
                # V3: Demographics
                gender=pm.demographic_info.gender if pm.demographic_info else None,
                age_group=pm.demographic_info.age_group if pm.demographic_info else None,
                demographic_label=pm.size_recommendation.demographic_label if pm.size_recommendation else None,
                gender_confidence=pm.demographic_info.gender_confidence if pm.demographic_info else None,
                age_confidence=pm.demographic_info.age_confidence if pm.demographic_info else None,
                is_valid=pm.validation_result.is_valid,
                missing_parts=pm.validation_result.missing_parts,
                validation_confidence=pm.validation_result.overall_confidence,
                body_part_confidences=pm.validation_result.confidence_scores,
                shoulder_width=pm.body_measurements.shoulder_width if pm.body_measurements else None,
                chest_width=pm.body_measurements.chest_width if pm.body_measurements else None,
                waist_width=pm.body_measurements.waist_width if pm.body_measurements else None,
                hip_width=pm.body_measurements.hip_width if pm.body_measurements else None,
                inseam=pm.body_measurements.inseam if pm.body_measurements else None,
                arm_length=pm.body_measurements.arm_length if pm.body_measurements else None,
                # V3: Circumference measurements
                chest_circumference=pm.body_measurements.chest_circumference if pm.body_measurements else None,
                waist_circumference=pm.body_measurements.waist_circumference if pm.body_measurements else None,
                hip_circumference=pm.body_measurements.hip_circumference if pm.body_measurements else None,
                arm_circumference=pm.body_measurements.arm_circumference if pm.body_measurements else None,
                thigh_circumference=pm.body_measurements.thigh_circumference if pm.body_measurements else None,
                estimated_height_cm=pm.body_measurements.estimated_height_cm if pm.body_measurements else None,
                pose_angle_degrees=pm.body_measurements.pose_angle_degrees if pm.body_measurements else None,
                recommended_size=pm.size_recommendation.recommended_size if pm.size_recommendation else None,
                size_probabilities=pm.size_recommendation.size_probabilities if pm.size_recommendation else None,
            )
            measurement_responses.append(person_response)

        # Save to database (save the first valid person to maintain compatibility)
        if measurement_responses:
            first_person = measurement_responses[0]
            if first_person.is_valid:
                measurement_record = Measurement(
                    brand_id=brand.id,
                    shoulder_width=first_person.shoulder_width,
                    chest_width=first_person.chest_width,
                    waist_width=first_person.waist_width,
                    hip_width=first_person.hip_width,
                    inseam=first_person.inseam,
                    arm_length=first_person.arm_length,
                    confidence_scores=first_person.body_part_confidences,
                    recommended_size=first_person.recommended_size,
                    size_probabilities=first_person.size_probabilities,
                    processing_time_ms=processing_time_ms,
                )
                db.add(measurement_record)
                db.commit()

        # Return multi-person response
        return MultiPersonMeasurementResponse(
            total_people_detected=result.total_people_detected,
            valid_people_count=result.valid_people_count,
            invalid_people_count=result.invalid_people_count,
            measurements=measurement_responses,
            processing_time_ms=processing_time_ms,
            processing_metadata=result.processing_metadata
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image: {str(e)}",
        )
