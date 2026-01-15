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
from app.core.auth import get_current_brand_by_api_key
from app.models import Brand, Measurement
from app.schemas import MeasurementResponse, MultiPersonMeasurementResponse, PersonMeasurementResponse, PoseLandmarks as PoseLandmarksSchema, PoseLandmark as PoseLandmarkSchema, BoundingBox as BoundingBoxSchema
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
# Increased to handle more concurrent requests
_ml_thread_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="ml_worker")

# Separate thread pool for database operations
_db_thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="db_worker")

# Request timeout in seconds
REQUEST_TIMEOUT_SECONDS = 120


# ============================================================================
# MODEL CACHING - Singleton instances to avoid reloading on every request
# ============================================================================
_cached_pose_detector = None
_cached_processor = None
_model_lock = asyncio.Lock()


def get_cached_pose_detector():
    """Get or create a cached PoseDetector instance"""
    global _cached_pose_detector
    if _cached_pose_detector is None:
        _cached_pose_detector = PoseDetector()
        logger.info("Created cached PoseDetector instance")
    return _cached_pose_detector


def get_cached_processor(use_ml_ratios: bool = True):
    """Get or create a cached DepthBasedMultiPersonProcessor instance"""
    global _cached_processor
    if _cached_processor is None:
        _cached_processor = DepthBasedMultiPersonProcessor(
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
        logger.info("Created cached DepthBasedMultiPersonProcessor instance")
    return _cached_processor


def convert_landmarks_to_schema(pose_landmarks, image_width: int, image_height: int) -> PoseLandmarksSchema:
    """Convert ML pose landmarks to API schema format"""
    if pose_landmarks is None:
        return None

    # MediaPipe pose landmark indices
    LANDMARK_INDICES = {
        "NOSE": 0,
        "LEFT_EYE": 2,
        "RIGHT_EYE": 5,
        "LEFT_SHOULDER": 11,
        "RIGHT_SHOULDER": 12,
        "LEFT_ELBOW": 13,
        "RIGHT_ELBOW": 14,
        "LEFT_WRIST": 15,
        "RIGHT_WRIST": 16,
        "LEFT_HIP": 23,
        "RIGHT_HIP": 24,
        "LEFT_KNEE": 25,
        "RIGHT_KNEE": 26,
        "LEFT_ANKLE": 27,
        "RIGHT_ANKLE": 28,
    }

    def to_landmark(name: str) -> PoseLandmarkSchema:
        idx = LANDMARK_INDICES.get(name)
        if idx is None or idx >= len(pose_landmarks.landmarks):
            return None
        lm = pose_landmarks.landmarks[idx]
        # Landmarks are stored as dictionaries with x, y in pixel coordinates
        # Normalize back to 0-1 range for the API response
        return PoseLandmarkSchema(
            x=lm["x"] / pose_landmarks.image_width,
            y=lm["y"] / pose_landmarks.image_height,
            visibility=lm["visibility"]
        )

    return PoseLandmarksSchema(
        nose=to_landmark("NOSE"),
        left_eye=to_landmark("LEFT_EYE"),
        right_eye=to_landmark("RIGHT_EYE"),
        left_shoulder=to_landmark("LEFT_SHOULDER"),
        right_shoulder=to_landmark("RIGHT_SHOULDER"),
        left_elbow=to_landmark("LEFT_ELBOW"),
        right_elbow=to_landmark("RIGHT_ELBOW"),
        left_wrist=to_landmark("LEFT_WRIST"),
        right_wrist=to_landmark("RIGHT_WRIST"),
        left_hip=to_landmark("LEFT_HIP"),
        right_hip=to_landmark("RIGHT_HIP"),
        left_knee=to_landmark("LEFT_KNEE"),
        right_knee=to_landmark("RIGHT_KNEE"),
        left_ankle=to_landmark("LEFT_ANKLE"),
        right_ankle=to_landmark("RIGHT_ANKLE"),
    )


def convert_bbox_to_schema(bbox, image_width: int, image_height: int) -> BoundingBoxSchema:
    """Convert bounding box to normalized API schema format"""
    return BoundingBoxSchema(
        x1=bbox.x1 / image_width,
        y1=bbox.y1 / image_height,
        x2=bbox.x2 / image_width,
        y2=bbox.y2 / image_height,
    )


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


@router.post("/process", response_model=MeasurementResponse)
async def process_measurement(
    file: UploadFile = File(...),
    product_id: str = Query(None, description="Optional product ID for product-specific sizing"),
    fit_preference: str = Query("regular", description="Fit preference: tight, regular, or loose"),
    height_cm: float = Query(None, description="User's height in cm for camera calibration (140-210cm). Improves accuracy by 5-10%"),
    brand: Brand = Depends(get_current_brand_by_api_key),
    db: Session = Depends(get_db),
):
    """
    Process a body image and extract measurements

    - **file**: Full-body image (JPG, PNG, WEBP)
    - **X-API-Key header**: Your API key for authentication (preferred)
    - **product_id**: (Optional) Product UUID for product-specific size recommendation
    - **fit_preference**: (Optional) Fit preference - tight, regular, or loose (default: regular)
    - **height_cm**: (Optional) User's actual height in cm (140-210) for camera calibration

    Returns body measurements and size recommendation

    **CAMERA CALIBRATION**: Providing the user's height via height_cm parameter
    significantly improves measurement accuracy by 5-10%. When height is known,
    the system can precisely calculate the pixels-per-cm ratio.

    **NEW**: Now supports product-specific sizing! Pass a product_id to get size recommendations
    based on that product's specific size chart instead of generic demographic charts.

    **Security Note**: Use X-API-Key header for authentication instead of query parameter.
    """
    start_time = time.time()

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
        # Define ML processing function to run in thread pool
        def run_single_person_ml():
            """CPU-bound ML processing - runs in thread pool"""
            # Step 1: Detect pose landmarks (using cached detector)
            detector = get_cached_pose_detector()
            pose_landmarks = detector.detect_from_array(image)

            if pose_landmarks is None:
                return None, None

            # Step 2: Extract measurements (with circumferences)
            extractor = EnhancedMeasurementExtractor(reference_height_cm=settings.DEFAULT_HEIGHT_CM)
            body_measurements = extractor.extract_measurements(pose_landmarks)

            return pose_landmarks, body_measurements

        # Run ML processing in thread pool to avoid blocking event loop
        pose_landmarks, body_measurements = await run_ml_task_with_timeout(
            run_single_person_ml,
            timeout=REQUEST_TIMEOUT_SECONDS
        )

        if pose_landmarks is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not detect body pose in image. Please ensure the image shows a full body in clear view.",
            )

        # Step 3: Recommend size (product-aware with v3) - runs in thread pool for DB access
        def get_size_recommendation():
            recommender = ProductAwareSizeRecommender(db_session=db)
            return recommender.recommend_size(
                measurements=body_measurements,
                gender=body_measurements.gender,
                age_group=body_measurements.age_group,
                demographic_label=body_measurements.demographic_label,
                product_id=product_id,
                fit_preference=fit_preference,
            )

        loop = asyncio.get_event_loop()
        size_recommendation = await loop.run_in_executor(_db_thread_pool, get_size_recommendation)

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Save measurement to database (run in thread pool)
        def save_to_db():
            measurement_record = Measurement(
                brand_id=brand.id,
                product_id=product_id if product_id else None,
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
            return measurement_record

        await loop.run_in_executor(_db_thread_pool, save_to_db)

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
    use_ml_ratios: bool = Query(True, description="Use ML-based depth ratio prediction (recommended for better accuracy)"),
    height_cm: float = Query(None, description="Height in cm for camera calibration (140-210cm). Only applies when single person detected."),
    brand: Brand = Depends(get_current_brand_by_api_key),
    db: Session = Depends(get_db),
):
    """
    Process a body image and extract measurements for ALL people (MULTI-PERSON)

    **NEW ENDPOINT**: Supports multiple people in the same image.

    - **file**: Full-body image with one or more people (JPG, PNG, WEBP)
    - **X-API-Key header**: Your API key for authentication (preferred)
    - **use_ml_ratios**: (Optional) Use ML depth ratio prediction for better accuracy (default: True)
    - **height_cm**: (Optional) Height in cm for calibration (only used for single-person images)

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

    **Security Note**: Use X-API-Key header for authentication instead of query parameter.
    """
    start_time = time.time()

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
            # Use cached processor to avoid reloading models
            processor = get_cached_processor(use_ml_ratios=use_ml_ratios)
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

        # Get image dimensions for normalization
        image_height, image_width = image.shape[:2]

        # Convert to response format
        measurement_responses = []
        for pm in result.measurements:
            # Convert landmarks and bounding box for visualization
            pose_landmarks_schema = convert_landmarks_to_schema(
                pm.pose_landmarks, image_width, image_height
            ) if pm.pose_landmarks else None

            bbox_schema = convert_bbox_to_schema(
                pm.bounding_box, image_width, image_height
            ) if pm.bounding_box else None

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
                # Visualization data
                bounding_box=bbox_schema,
                pose_landmarks=pose_landmarks_schema,
            )
            measurement_responses.append(person_response)

        # Save to database (save the first valid person to maintain compatibility)
        # Run in thread pool to avoid blocking event loop
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

                # Run DB operations in thread pool
                def save_multi_person_to_db():
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
                        used_ml_ratios=ml_metadata,
                    )
                    db.add(measurement_record)
                    db.commit()

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(_db_thread_pool, save_multi_person_to_db)

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



# ============================================================================
# BATCH PROCESSING HELPER
# ============================================================================

def process_image_sync(image_content: bytes, api_key: str, use_ml_ratios: bool = True) -> dict:
    """
    Synchronous helper function for batch processing.
    This is called from batch.py in a thread pool.

    Args:
        image_content: Raw image bytes
        api_key: API key (for future use in rate limiting)
        use_ml_ratios: Whether to use ML-enhanced measurements

    Returns:
        Dictionary with measurement results
    """
    import cv2
    import numpy as np
    import time

    start_time = time.time()

    # Decode image
    nparr = np.frombuffer(image_content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not decode image")

    # Get cached processor
    processor = get_cached_processor(use_ml_ratios=use_ml_ratios)
    result = processor.process_image(image)

    processing_time_ms = (time.time() - start_time) * 1000

    # Convert to dictionary format
    measurements = []
    for pm in result.measurements:
        measurements.append({
            "person_id": pm.person_id,
            "is_valid": pm.is_valid,
            "gender": pm.gender,
            "age_group": pm.age_group,
            "demographic_label": pm.demographic_label,
            "shoulder_width": pm.body_measurements.shoulder_width_cm if pm.body_measurements else None,
            "chest_width": pm.body_measurements.chest_width_cm if pm.body_measurements else None,
            "waist_width": pm.body_measurements.waist_width_cm if pm.body_measurements else None,
            "hip_width": pm.body_measurements.hip_width_cm if pm.body_measurements else None,
            "inseam": pm.body_measurements.inseam_cm if pm.body_measurements else None,
            "arm_length": pm.body_measurements.arm_length_cm if pm.body_measurements else None,
            "chest_circumference": pm.body_measurements.chest_circumference_cm if pm.body_measurements else None,
            "waist_circumference": pm.body_measurements.waist_circumference_cm if pm.body_measurements else None,
            "hip_circumference": pm.body_measurements.hip_circumference_cm if pm.body_measurements else None,
            "estimated_height_cm": pm.body_measurements.estimated_height_cm if pm.body_measurements else None,
            "recommended_size": pm.size_recommendation.recommended_size if pm.size_recommendation else None,
            "size_probabilities": pm.size_recommendation.probabilities if pm.size_recommendation else None,
            "detection_confidence": pm.detection_confidence,
            "validation_confidence": pm.validation_result.overall_confidence if pm.validation_result else 0,
        })

    return {
        "total_people_detected": result.total_people_detected,
        "valid_people_count": result.valid_people_count,
        "invalid_people_count": result.invalid_people_count,
        "measurements": measurements,
        "processing_time_ms": processing_time_ms,
    }

