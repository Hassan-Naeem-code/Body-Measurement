"""
Multi-Person Body Measurement Processor
Orchestrates YOLOv8 detection, MediaPipe pose estimation, validation, and measurement extraction
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass

from app.ml.person_detector import PersonDetector, PersonBoundingBox
from app.ml.pose_detector import PoseDetector, PoseLandmarks
from app.ml.body_validator import FullBodyValidator, ValidationResult
from app.ml.measurement_extractor import MeasurementExtractor, BodyMeasurements
from app.ml.size_recommender import SizeRecommender, SizeRecommendation


@dataclass
class PersonMeasurement:
    """Complete measurement result for one person"""
    person_id: int
    detection_confidence: float  # From YOLO
    validation_result: ValidationResult
    body_measurements: Optional[BodyMeasurements]
    size_recommendation: Optional[SizeRecommendation]
    bounding_box: PersonBoundingBox


@dataclass
class MultiPersonResult:
    """Result of multi-person processing"""
    total_people_detected: int
    valid_people_count: int
    invalid_people_count: int
    measurements: List[PersonMeasurement]  # Only valid people
    processing_metadata: dict


class MultiPersonProcessor:
    """
    Complete multi-person body measurement pipeline

    Pipeline:
    1. Detect all people with YOLOv8
    2. For each person:
       a. Crop region
       b. Detect pose with MediaPipe
       c. Validate full-body visibility
       d. Extract measurements (if valid)
       e. Recommend size (if valid)
    3. Filter out invalid people
    4. Return measurements for valid people only
    """

    def __init__(
        self,
        yolo_model_size: str = "yolov8m.pt",
        yolo_confidence: float = 0.5,
        reference_height_cm: float = 170.0,
        pose_confidence: float = 0.5,
        custom_validation_thresholds: dict = None
    ):
        """
        Args:
            yolo_model_size: YOLOv8 model size
            yolo_confidence: Minimum YOLO detection confidence
            reference_height_cm: Default height for measurement calibration
            pose_confidence: Minimum MediaPipe pose detection confidence
            custom_validation_thresholds: Custom body part visibility thresholds
        """
        self.person_detector = PersonDetector(yolo_model_size, yolo_confidence)
        self.pose_detector = PoseDetector(min_detection_confidence=pose_confidence)
        self.body_validator = FullBodyValidator(custom_validation_thresholds)
        self.measurement_extractor = MeasurementExtractor(reference_height_cm)
        self.size_recommender = SizeRecommender()

    def process_image(self, image: np.ndarray) -> MultiPersonResult:
        """
        Process image and extract measurements for all valid people

        Args:
            image: BGR image as numpy array

        Returns:
            MultiPersonResult with measurements for valid people
        """
        # Step 1: Detect all people
        people_bboxes = self.person_detector.detect_people(image)

        if not people_bboxes:
            return MultiPersonResult(
                total_people_detected=0,
                valid_people_count=0,
                invalid_people_count=0,
                measurements=[],
                processing_metadata={
                    "stage_failed": "person_detection",
                    "reason": "No people detected in image"
                }
            )

        # Step 2: Process each detected person
        all_person_measurements = []

        for bbox in people_bboxes:
            person_result = self._process_single_person(image, bbox)
            all_person_measurements.append(person_result)

        # Step 3: Filter for valid people only
        valid_measurements = [
            pm for pm in all_person_measurements
            if pm.validation_result.is_valid
        ]

        invalid_count = len(all_person_measurements) - len(valid_measurements)

        return MultiPersonResult(
            total_people_detected=len(people_bboxes),
            valid_people_count=len(valid_measurements),
            invalid_people_count=invalid_count,
            measurements=valid_measurements,
            processing_metadata={
                "detection_model": "yolov8m",
                "pose_model": "mediapipe_pose_v2",
                "validation_version": "1.0"
            }
        )

    def _process_single_person(
        self,
        image: np.ndarray,
        bbox: PersonBoundingBox
    ) -> PersonMeasurement:
        """
        Process a single detected person through the full pipeline

        Args:
            image: Original image
            bbox: Person bounding box from YOLO

        Returns:
            PersonMeasurement (may be invalid)
        """
        # Crop person region
        cropped_image, crop_metadata = self.person_detector.crop_person(image, bbox)

        # Detect pose landmarks
        pose_landmarks = self.pose_detector.detect_from_array(cropped_image)

        # If pose detection failed
        if pose_landmarks is None:
            return PersonMeasurement(
                person_id=bbox.person_id,
                detection_confidence=bbox.confidence,
                validation_result=ValidationResult(
                    is_valid=False,
                    missing_parts=["pose_detection_failed"],
                    confidence_scores={},
                    overall_confidence=0.0,
                    validation_details={}
                ),
                body_measurements=None,
                size_recommendation=None,
                bounding_box=bbox
            )

        # Validate full body
        validation_result = self.body_validator.validate_full_body(pose_landmarks)

        # Check if human (not animal/object)
        if not self.body_validator.is_human(pose_landmarks):
            validation_result.is_valid = False
            validation_result.missing_parts.append("not_human_detected")

        # If validation failed
        if not validation_result.is_valid:
            return PersonMeasurement(
                person_id=bbox.person_id,
                detection_confidence=bbox.confidence,
                validation_result=validation_result,
                body_measurements=None,
                size_recommendation=None,
                bounding_box=bbox
            )

        # Extract measurements
        body_measurements = self.measurement_extractor.extract_measurements(pose_landmarks)

        # Recommend size
        size_recommendation = self.size_recommender.recommend_size(body_measurements)

        return PersonMeasurement(
            person_id=bbox.person_id,
            detection_confidence=bbox.confidence,
            validation_result=validation_result,
            body_measurements=body_measurements,
            size_recommendation=size_recommendation,
            bounding_box=bbox
        )
