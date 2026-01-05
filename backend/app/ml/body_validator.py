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
        Strict check to verify this is a REAL HUMAN (not mask/mannequin/drawing/animal)

        Real humans have:
        1. Proper vertical body structure (nose > shoulders > hips > knees > ankles)
        2. High visibility scores for key landmarks (real humans have clear body parts)
        3. Bilateral symmetry (left/right sides at similar heights)
        4. Realistic proportions (height > width, proper segment ratios)
        5. Natural pose (not statue-like, not cartoon-like)
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

            # === CHECK 1: Vertical ordering (basic anatomy) ===
            vertical_order_correct = (
                nose["y"] < left_shoulder["y"] < left_hip["y"] < left_knee["y"] < left_ankle["y"] and
                nose["y"] < right_shoulder["y"] < right_hip["y"] < right_knee["y"] < right_ankle["y"]
            )
            if not vertical_order_correct:
                return False

            # === CHECK 2: High visibility scores (real humans have clear landmarks) ===
            # Masks/drawings have face but NO BODY - this is the key difference!

            # Check BODY landmarks separately (not face)
            body_landmarks = [
                "LEFT_SHOULDER", "RIGHT_SHOULDER",
                "LEFT_HIP", "RIGHT_HIP",
                "LEFT_KNEE", "RIGHT_KNEE",
                "LEFT_ANKLE", "RIGHT_ANKLE"
            ]

            body_visibility_scores = [
                pose_landmarks.visibility_scores.get(landmark, 0.0)
                for landmark in body_landmarks
            ]

            # CRITICAL: Real humans must have visible BODY (not just face)
            # Masks/drawings have face but no clear body landmarks
            high_body_visibility_count = sum(1 for score in body_visibility_scores if score > 0.5)
            if high_body_visibility_count < 6:  # At least 6/8 BODY landmarks must be visible
                return False

            # Average BODY visibility must be high
            avg_body_visibility = sum(body_visibility_scores) / len(body_visibility_scores)
            if avg_body_visibility < 0.5:  # Real humans have clearly visible bodies
                return False

            # Additionally check all critical landmarks (including face)
            all_critical_landmarks = [
                "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER",
                "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE"
            ]
            all_visibility_scores = [
                pose_landmarks.visibility_scores.get(landmark, 0.0)
                for landmark in all_critical_landmarks
            ]

            avg_overall_visibility = sum(all_visibility_scores) / len(all_visibility_scores)
            if avg_overall_visibility < 0.45:
                return False

            # === CHECK 3: Bilateral symmetry ===
            shoulder_symmetry = abs(left_shoulder["y"] - right_shoulder["y"]) < 50
            hip_symmetry = abs(left_hip["y"] - right_hip["y"]) < 50
            if not (shoulder_symmetry and hip_symmetry):
                return False

            # === CHECK 4: Realistic aspect ratio ===
            body_height = max(left_ankle["y"], right_ankle["y"]) - nose["y"]
            body_width = abs(left_shoulder["x"] - right_shoulder["x"])
            if body_height <= body_width:  # Height must be > width
                return False

            # Aspect ratio should be reasonable (humans are 1.5-3x taller than wide)
            aspect_ratio = body_height / max(body_width, 1)
            if aspect_ratio < 1.3 or aspect_ratio > 4.0:  # Too extreme = not human
                return False

            # === CHECK 5: Realistic body segment proportions ===
            # Check torso length vs leg length (should be somewhat balanced)
            torso_length = abs(left_hip["y"] - left_shoulder["y"])
            leg_length = abs(left_ankle["y"] - left_hip["y"])

            if torso_length < 10 or leg_length < 10:  # Too small = suspicious
                return False

            # Torso/leg ratio should be reasonable (0.8-1.3 for humans)
            torso_leg_ratio = torso_length / max(leg_length, 1)
            if torso_leg_ratio < 0.6 or torso_leg_ratio > 1.5:  # Too extreme = not human
                return False

            # === CHECK 6: Arms should exist and be reasonable ===
            left_wrist = pose_landmarks.landmarks[15]
            right_wrist = pose_landmarks.landmarks[16]

            # Arms should be detected with some visibility
            arm_visibility = (
                pose_landmarks.visibility_scores.get("LEFT_WRIST", 0.0) +
                pose_landmarks.visibility_scores.get("RIGHT_WRIST", 0.0)
            ) / 2

            # At least one arm should be somewhat visible (humans have arms!)
            if arm_visibility < 0.2:  # Very low = might not be human
                return False

            # All checks passed - likely a real human
            return True

        except (KeyError, IndexError, ZeroDivisionError):
            return False
