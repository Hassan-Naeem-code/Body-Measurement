"""
Full-Body Validation System
Validates that all required body parts are visible with sufficient confidence
Ensures WHOLE human being is visible from head to toes
"""

from typing import Dict, List
from dataclasses import dataclass
from app.ml.pose_detector import PoseLandmarks


@dataclass
class ValidationResult:
    """Result of full-body validation"""
    is_valid: bool
    missing_parts: List[str]
    confidence_scores: Dict[str, float]
    overall_confidence: float
    validation_details: Dict[str, Dict]  # Detailed per-body-part results


class FullBodyValidator:
    """
    Validates that a person has ALL required body parts visible

    Requirements for WHOLE human being:
    - Head (nose visible)
    - Shoulders (both left and right)
    - Elbows (both visible)
    - Hands/Wrists (both visible)
    - Stomach/Torso (hips visible)
    - Legs (both knees visible)
    - Feet/Ankles (both ankles visible)
    """

    # Visibility thresholds (MediaPipe visibility score 0-1)
    VISIBILITY_THRESHOLDS = {
        "head": 0.6,          # Nose must be clearly visible
        "shoulders": 0.7,      # Both shoulders critical for measurements
        "elbows": 0.5,         # Both elbows for arm measurements
        "hands": 0.5,          # Both wrists needed
        "torso": 0.6,          # Hips for waist/torso
        "legs": 0.6,           # Both knees needed
        "feet": 0.6            # Both ankles needed
    }

    # Minimum overall confidence to pass validation
    MIN_OVERALL_CONFIDENCE = 0.65

    def __init__(self, custom_thresholds: Dict[str, float] = None):
        """
        Args:
            custom_thresholds: Override default visibility thresholds
        """
        self.thresholds = {**self.VISIBILITY_THRESHOLDS}
        if custom_thresholds:
            self.thresholds.update(custom_thresholds)

    def validate_full_body(self, pose_landmarks: PoseLandmarks) -> ValidationResult:
        """
        Validate that ALL required body parts are visible (whole human)

        Args:
            pose_landmarks: Detected pose landmarks from MediaPipe

        Returns:
            ValidationResult with validation status and details
        """
        validation_details = {}
        missing_parts = []

        # 1. Validate Head
        head_result = self._validate_head(pose_landmarks)
        validation_details["head"] = head_result
        if not head_result["is_valid"]:
            missing_parts.append("head")

        # 2. Validate Shoulders
        shoulder_result = self._validate_shoulders(pose_landmarks)
        validation_details["shoulders"] = shoulder_result
        if not shoulder_result["is_valid"]:
            missing_parts.append("shoulders")

        # 3. Validate Elbows
        elbows_result = self._validate_elbows(pose_landmarks)
        validation_details["elbows"] = elbows_result
        if not elbows_result["is_valid"]:
            missing_parts.append("elbows")

        # 4. Validate Hands/Wrists
        hands_result = self._validate_hands(pose_landmarks)
        validation_details["hands"] = hands_result
        if not hands_result["is_valid"]:
            missing_parts.append("hands/wrists")

        # 5. Validate Torso
        torso_result = self._validate_torso(pose_landmarks)
        validation_details["torso"] = torso_result
        if not torso_result["is_valid"]:
            missing_parts.append("stomach/torso")

        # 6. Validate Legs
        legs_result = self._validate_legs(pose_landmarks)
        validation_details["legs"] = legs_result
        if not legs_result["is_valid"]:
            missing_parts.append("legs")

        # 7. Validate Feet/Ankles
        feet_result = self._validate_feet(pose_landmarks)
        validation_details["feet"] = feet_result
        if not feet_result["is_valid"]:
            missing_parts.append("feet/ankles")

        # Calculate overall confidence
        all_confidences = []
        for part_result in validation_details.values():
            all_confidences.extend(part_result["landmark_scores"].values())

        overall_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

        # Determine if valid - ALL parts must be present
        is_valid = (
            len(missing_parts) == 0 and
            overall_confidence >= self.MIN_OVERALL_CONFIDENCE
        )

        # Collect confidence scores
        confidence_scores = {
            part: result["average_confidence"]
            for part, result in validation_details.items()
        }

        return ValidationResult(
            is_valid=is_valid,
            missing_parts=missing_parts,
            confidence_scores=confidence_scores,
            overall_confidence=overall_confidence,
            validation_details=validation_details
        )

    def _validate_head(self, pose_landmarks: PoseLandmarks) -> Dict:
        """Validate head visibility (nose)"""
        nose_visibility = pose_landmarks.visibility_scores.get("NOSE", 0.0)

        is_valid = nose_visibility >= self.thresholds["head"]

        return {
            "is_valid": is_valid,
            "average_confidence": nose_visibility,
            "landmark_scores": {"nose": nose_visibility},
            "threshold": self.thresholds["head"]
        }

    def _validate_shoulders(self, pose_landmarks: PoseLandmarks) -> Dict:
        """Validate both shoulders are visible"""
        left_shoulder = pose_landmarks.visibility_scores.get("LEFT_SHOULDER", 0.0)
        right_shoulder = pose_landmarks.visibility_scores.get("RIGHT_SHOULDER", 0.0)

        avg_confidence = (left_shoulder + right_shoulder) / 2
        is_valid = (
            left_shoulder >= self.thresholds["shoulders"] and
            right_shoulder >= self.thresholds["shoulders"]
        )

        return {
            "is_valid": is_valid,
            "average_confidence": avg_confidence,
            "landmark_scores": {
                "left_shoulder": left_shoulder,
                "right_shoulder": right_shoulder
            },
            "threshold": self.thresholds["shoulders"]
        }

    def _validate_elbows(self, pose_landmarks: PoseLandmarks) -> Dict:
        """Validate both elbows are visible"""
        left_elbow = pose_landmarks.visibility_scores.get("LEFT_ELBOW", 0.0)
        right_elbow = pose_landmarks.visibility_scores.get("RIGHT_ELBOW", 0.0)

        avg_confidence = (left_elbow + right_elbow) / 2
        is_valid = (
            left_elbow >= self.thresholds["elbows"] and
            right_elbow >= self.thresholds["elbows"]
        )

        return {
            "is_valid": is_valid,
            "average_confidence": avg_confidence,
            "landmark_scores": {
                "left_elbow": left_elbow,
                "right_elbow": right_elbow
            },
            "threshold": self.thresholds["elbows"]
        }

    def _validate_hands(self, pose_landmarks: PoseLandmarks) -> Dict:
        """Validate both wrists are visible"""
        left_wrist = pose_landmarks.visibility_scores.get("LEFT_WRIST", 0.0)
        right_wrist = pose_landmarks.visibility_scores.get("RIGHT_WRIST", 0.0)

        avg_confidence = (left_wrist + right_wrist) / 2
        is_valid = (
            left_wrist >= self.thresholds["hands"] and
            right_wrist >= self.thresholds["hands"]
        )

        return {
            "is_valid": is_valid,
            "average_confidence": avg_confidence,
            "landmark_scores": {
                "left_wrist": left_wrist,
                "right_wrist": right_wrist
            },
            "threshold": self.thresholds["hands"]
        }

    def _validate_torso(self, pose_landmarks: PoseLandmarks) -> Dict:
        """Validate torso visibility (using shoulders and hips)"""
        left_shoulder = pose_landmarks.visibility_scores.get("LEFT_SHOULDER", 0.0)
        right_shoulder = pose_landmarks.visibility_scores.get("RIGHT_SHOULDER", 0.0)
        left_hip = pose_landmarks.visibility_scores.get("LEFT_HIP", 0.0)
        right_hip = pose_landmarks.visibility_scores.get("RIGHT_HIP", 0.0)

        avg_confidence = (left_shoulder + right_shoulder + left_hip + right_hip) / 4

        # All four points should be reasonably visible
        is_valid = avg_confidence >= self.thresholds["torso"]

        return {
            "is_valid": is_valid,
            "average_confidence": avg_confidence,
            "landmark_scores": {
                "left_shoulder": left_shoulder,
                "right_shoulder": right_shoulder,
                "left_hip": left_hip,
                "right_hip": right_hip
            },
            "threshold": self.thresholds["torso"]
        }

    def _validate_legs(self, pose_landmarks: PoseLandmarks) -> Dict:
        """Validate both legs are visible (knees)"""
        left_knee = pose_landmarks.visibility_scores.get("LEFT_KNEE", 0.0)
        right_knee = pose_landmarks.visibility_scores.get("RIGHT_KNEE", 0.0)

        avg_confidence = (left_knee + right_knee) / 2
        is_valid = (
            left_knee >= self.thresholds["legs"] and
            right_knee >= self.thresholds["legs"]
        )

        return {
            "is_valid": is_valid,
            "average_confidence": avg_confidence,
            "landmark_scores": {
                "left_knee": left_knee,
                "right_knee": right_knee
            },
            "threshold": self.thresholds["legs"]
        }

    def _validate_feet(self, pose_landmarks: PoseLandmarks) -> Dict:
        """Validate both feet are visible (ankles)"""
        left_ankle = pose_landmarks.visibility_scores.get("LEFT_ANKLE", 0.0)
        right_ankle = pose_landmarks.visibility_scores.get("RIGHT_ANKLE", 0.0)

        avg_confidence = (left_ankle + right_ankle) / 2
        is_valid = (
            left_ankle >= self.thresholds["feet"] and
            right_ankle >= self.thresholds["feet"]
        )

        return {
            "is_valid": is_valid,
            "average_confidence": avg_confidence,
            "landmark_scores": {
                "left_ankle": left_ankle,
                "right_ankle": right_ankle
            },
            "threshold": self.thresholds["feet"]
        }

    def is_human(self, pose_landmarks: PoseLandmarks) -> bool:
        """
        Quick check to verify this is likely a human (not animal/object)

        Humans have:
        - Nose above shoulders
        - Shoulders above hips
        - Hips above knees
        - Knees above ankles
        - Bilateral symmetry
        """
        try:
            # Get landmark coordinates
            nose = pose_landmarks.landmarks[0]
            left_shoulder = pose_landmarks.landmarks[11]
            right_shoulder = pose_landmarks.landmarks[12]
            left_hip = pose_landmarks.landmarks[23]
            right_hip = pose_landmarks.landmarks[24]
            left_knee = pose_landmarks.landmarks[25]
            right_knee = pose_landmarks.landmarks[26]
            left_ankle = pose_landmarks.landmarks[27]
            right_ankle = pose_landmarks.landmarks[28]

            # Check vertical ordering (Y increases downward in images)
            vertical_order_correct = (
                nose["y"] < left_shoulder["y"] < left_hip["y"] < left_knee["y"] < left_ankle["y"] and
                nose["y"] < right_shoulder["y"] < right_hip["y"] < right_knee["y"] < right_ankle["y"]
            )

            # Check reasonable bilateral symmetry (shoulders and hips roughly same Y level)
            shoulder_symmetry = abs(left_shoulder["y"] - right_shoulder["y"]) < 50  # pixels
            hip_symmetry = abs(left_hip["y"] - right_hip["y"]) < 50

            # Check aspect ratio (humans are taller than wide)
            body_height = max(left_ankle["y"], right_ankle["y"]) - nose["y"]
            body_width = abs(left_shoulder["x"] - right_shoulder["x"])
            aspect_ratio_valid = body_height > body_width  # Height > width

            return vertical_order_correct and shoulder_symmetry and hip_symmetry and aspect_ratio_valid

        except (KeyError, IndexError):
            return False
