from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, status, Request
from sqlalchemy.orm import Session
import cv2
import numpy as np
import time
import tempfile
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

from app.core.database import get_db
from app.core.config import settings
from app.models import Brand, Measurement
from app.schemas import MeasurementResponse, MultiPersonMeasurementResponse, PersonMeasurementResponse
from app.ml import PoseDetector, MeasurementExtractor, SizeRecommender, MultiPersonProcessor, EnhancedMultiPersonProcessor
from app.ml.multi_person_processor_v3 import DepthBasedMultiPersonProcessor
from app.ml.measurement_extractor_v2 import EnhancedMeasurementExtractor
from app.ml.size_recommender_v2 import EnhancedSizeRecommender
from app.ml.size_recommender_v3 import ProductAwareSizeRecommender
from app.models.product import Product

router = APIRouter()
logger = logging.getLogger(__name__)

# Thread pool for CPU-bound ML operations
# This prevents blocking the async event loop
_ml_thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ml_worker")

# Request timeout in seconds
REQUEST_TIMEOUT_SECONDS = 120


async def check_client_disconnected(request: Request) -> bool:
    """Check if the client has disconnected"""
    return await request.is_disconnected()


async def run_ml_task_with_timeout(func, *args, timeout=REQUEST_TIMEOUT_SECONDS, request: Request = None):
    """
    Run CPU-bound ML task in thread pool with timeout and disconnection detection.

    This allows:
    1. Non-blocking execution (other requests can proceed)
    2. Timeout handling (prevents hanging forever)
    3. Client disconnection detection (cancels if client leaves)
    """
    loop = asyncio.get_event_loop()

    # Create a task for the ML processing
    ml_task = loop.run_in_executor(_ml_thread_pool, func, *args)

    try:
        # Wait for completion with timeout
        result = await asyncio.wait_for(ml_task, timeout=timeout)
        return result
    except asyncio.TimeoutError:
        logger.warning("ML processing timed out after %d seconds", timeout)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Processing timed out after {timeout} seconds. Please try with a smaller image."
        )
    except asyncio.CancelledError:
        logger.info("ML task was cancelled (client disconnected)")
        raise


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
    product_id: str = Query(None, description="Optional product ID for product-specific sizing"),
    fit_preference: str = Query("regular", description="Fit preference: tight, regular, or loose"),
    db: Session = Depends(get_db),
):
    """
    Process a body image and extract measurements

    - **file**: Full-body image (JPG, PNG, WEBP)
    - **api_key**: Your API key for authentication
    - **product_id**: (Optional) Product UUID for product-specific size recommendation
    - **fit_preference**: (Optional) Fit preference - tight, regular, or loose (default: regular)

    Returns body measurements and size recommendation

    **NEW**: Now supports product-specific sizing! Pass a product_id to get size recommendations
    based on that product's specific size chart instead of generic demographic charts.
    """
    start_time = time.time()

    # Validate brand
    brand = get_brand_by_api_key(api_key, db)

    # Validate fit preference
    if fit_preference not in ["tight", "regular", "loose"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid fit_preference '{fit_preference}'. Must be 'tight', 'regular', or 'loose'",
        )

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

        # Step 2: Extract measurements (with circumferences)
        extractor = EnhancedMeasurementExtractor(reference_height_cm=settings.DEFAULT_HEIGHT_CM)
        body_measurements = extractor.extract_measurements(pose_landmarks)

        # Step 3: Recommend size (product-aware with v3)
        recommender = ProductAwareSizeRecommender(db_session=db)
        size_recommendation = recommender.recommend_size(
            measurements=body_measurements,
            gender=body_measurements.gender,
            age_group=body_measurements.age_group,
            demographic_label=body_measurements.demographic_label,
            product_id=product_id,
            fit_preference=fit_preference,
        )

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Save measurement to database
        measurement_record = Measurement(
            brand_id=brand.id,
            product_id=product_id if product_id else None,  # NEW: Save product association
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
    request: Request,
    file: UploadFile = File(...),
    api_key: str = Query(..., description="API key for authentication"),
    use_ml_ratios: bool = Query(True, description="Use ML-based depth ratio prediction (recommended for better accuracy)"),
    db: Session = Depends(get_db),
):
    """
    Process a body image and extract measurements for ALL people (MULTI-PERSON)

    **NEW ENDPOINT**: Supports multiple people in the same image.

    - **file**: Full-body image with one or more people (JPG, PNG, WEBP)
    - **api_key**: Your API key for authentication
    - **use_ml_ratios**: (Optional) Use ML depth ratio prediction for better accuracy (default: True)

    **ML Enhancement (NEW):**
    - Set use_ml_ratios=true for personalized depth ratios based on body type (recommended)
    - Set use_ml_ratios=false for fixed anthropometric ratios (legacy mode)
    - ML mode provides 10-15% accuracy improvement for non-average body types

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
        # Check if client disconnected before heavy processing
        if await request.is_disconnected():
            logger.info("Client disconnected before ML processing started")
            raise HTTPException(
                status_code=status.HTTP_499_CLIENT_CLOSED_REQUEST,
                detail="Client disconnected"
            )

        # Process all people in the image with DEPTH-BASED V3 processor
        # (ML-enhanced for better accuracy)
        def run_ml_processing():
            """CPU-bound ML processing function to run in thread pool"""
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
                    "overall_min": settings.BODY_VALIDATION_OVERALL_MIN,
                },
                use_ml_ratios=use_ml_ratios,
            )
            return processor.process_image(image)

        # Run ML processing in thread pool with timeout
        result = await run_ml_task_with_timeout(
            run_ml_processing,
            timeout=REQUEST_TIMEOUT_SECONDS,
            request=request
        )

        # DEBUG: Log detection results
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"🔍 DETECTION RESULTS:")
        logger.error(f"  Total people detected: {result.total_people_detected}")
        logger.error(f"  Valid people count: {result.valid_people_count}")
        logger.error(f"  Number of measurements: {len(result.measurements)}")

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
            # Check if the issue is "not a human" vs "missing body parts"
            all_missing_parts = set()
            not_human_count = 0

            for pm in result.measurements:
                missing_parts = pm.validation_result.missing_parts
                all_missing_parts.update(missing_parts)

                # DEBUG: Log validation details
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"🔍 DEBUG Validation:")
                logger.error(f"  Missing parts: {missing_parts}")
                logger.error(f"  Overall confidence: {pm.validation_result.overall_confidence:.2%}")
                logger.error(f"  Confidence scores: {pm.validation_result.confidence_scores}")
                logger.error(f"  Threshold: {settings.BODY_VALIDATION_OVERALL_MIN:.2%}")

                # Check if marked as "not human"
                if "not_human_detected" in missing_parts:
                    not_human_count += 1

            # Case 1: Not a real human (mask, drawing, animal, mannequin)
            if not_human_count > 0:
                if not_human_count == result.total_people_detected:
                    # All detected objects are not humans
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=(
                            "❌ NOT A REAL HUMAN DETECTED\n\n"
                            "The image does not contain a real human being. "
                            "Please upload a photo of a REAL PERSON (not a mask, mannequin, drawing, cartoon, or animal).\n\n"
                            "Requirements:\n"
                            "✓ Real human being (not an object or costume)\n"
                            "✓ Full body visible (head to feet)\n"
                            "✓ Clear photo (not blurry or obscured)"
                        ),
                    )
                else:
                    # Some are not human, some have other issues
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=(
                            f"Detected {result.total_people_detected} people, but validation failed:\n"
                            f"• {not_human_count} NOT REAL HUMANS (masks, drawings, objects)\n"
                            f"• {result.total_people_detected - not_human_count} missing body parts\n\n"
                            "Please upload a photo with REAL PEOPLE showing FULL BODY (head to feet)."
                        ),
                    )

            # Case 2: Real humans but missing body parts (headshots, cropped images)
            missing_parts_list = [p for p in all_missing_parts if p != "not_human_detected"][:5]
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"❌ FULL BODY NOT VISIBLE\n\n"
                    f"Detected {result.total_people_detected} people, but they are not showing their WHOLE BODY.\n\n"
                    f"Missing parts: {', '.join(missing_parts_list)}\n\n"
                    "Requirements:\n"
                    "✓ Full body visible from HEAD to FEET\n"
                    "✓ All parts visible: head, shoulders, elbows, hands, torso, legs, feet\n"
                    "✓ Not a headshot or cropped photo\n"
                    "✓ Person standing upright (not sitting or lying down)"
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
                # Prepare ML metadata
                first_measurement = result.measurements[0]
                ml_metadata = None
                if use_ml_ratios and hasattr(first_measurement.body_measurements, 'predicted_ratios'):
                    ml_metadata = {
                        "used": True,
                        "method": first_measurement.body_measurements.predicted_ratios.method,
                        "confidence": first_measurement.body_measurements.predicted_ratios.confidence,
                        "body_shape": first_measurement.body_measurements.body_shape_category,
                        "bmi_estimate": first_measurement.body_measurements.bmi_estimate,
                        "ratios": {
                            "chest": first_measurement.body_measurements.predicted_ratios.chest_ratio,
                            "waist": first_measurement.body_measurements.predicted_ratios.waist_ratio,
                            "hip": first_measurement.body_measurements.predicted_ratios.hip_ratio,
                        }
                    }
                elif not use_ml_ratios:
                    ml_metadata = {
                        "used": False,
                        "method": "fixed_anthropometric",
                        "confidence": 0.75,
                    }

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
                    used_ml_ratios=ml_metadata,  # NEW: Track ML usage
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
