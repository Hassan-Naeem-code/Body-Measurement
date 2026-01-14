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
    DEFAULT_OVERALL_MIN = 0.65

    def __init__(self, custom_thresholds: Dict[str, float] = None):
        """
        Args:
            custom_thresholds: Override default visibility thresholds
                             Can include 'overall_min' to override MIN_OVERALL_CONFIDENCE
        """
        self.thresholds = {**self.VISIBILITY_THRESHOLDS}

        # Set overall minimum - use custom if provided, otherwise use default
        self.MIN_OVERALL_CONFIDENCE = self.DEFAULT_OVERALL_MIN

        if custom_thresholds:
            # Extract overall_min if provided
            if 'overall_min' in custom_thresholds:
                self.MIN_OVERALL_CONFIDENCE = custom_thresholds['overall_min']
                # Don't add it to thresholds dict (it's not a body part threshold)
                custom_thresholds = {k: v for k, v in custom_thresholds.items() if k != 'overall_min'}

            # Update body part thresholds
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
        Check to verify this is a REAL HUMAN (not mask/mannequin/drawing/animal)

        This check is VERY LENIENT - we want to accept real humans even if some parts
        are occluded (e.g., backpack covering arms, hands behind back, bag in front).

        Only reject if CLEARLY NOT human (completely wrong anatomy).

        Requirements (intentionally minimal):
        1. Basic vertical structure (head somewhere above hips)
        2. At least some core body landmarks detected
        3. Not impossibly proportioned
        """
        try:
            # Get landmark coordinates
            nose = pose_landmarks.landmarks[0]
            left_shoulder = pose_landmarks.landmarks[11]
            right_shoulder = pose_landmarks.landmarks[12]
            left_hip = pose_landmarks.landmarks[23]
            right_hip = pose_landmarks.landmarks[24]
            left_ankle = pose_landmarks.landmarks[27]
            right_ankle = pose_landmarks.landmarks[28]

            # === CHECK 1: Basic vertical ordering (very lenient) ===
            # Only check that head is above hips (allows for many poses)
            avg_shoulder_y = (left_shoulder["y"] + right_shoulder["y"]) / 2
            avg_hip_y = (left_hip["y"] + right_hip["y"]) / 2
            avg_ankle_y = (left_ankle["y"] + right_ankle["y"]) / 2

            # Head should be above hips (very basic check)
            if nose["y"] > avg_hip_y:
                return False

            # Hips should be above ankles
            if avg_hip_y > avg_ankle_y + 50:  # Allow some tolerance for poses
                return False

            # === CHECK 2: Core body landmarks must have SOME detection ===
            # Very lenient - just need evidence that a body was detected
            core_landmarks = [
                "LEFT_SHOULDER", "RIGHT_SHOULDER",
                "LEFT_HIP", "RIGHT_HIP"
            ]

            core_visibility_scores = [
                pose_landmarks.visibility_scores.get(landmark, 0.0)
                for landmark in core_landmarks
            ]

            # At least 2 of 4 core landmarks must be detected (>0.15)
            # This is very lenient to handle occlusion
            visible_core_count = sum(1 for score in core_visibility_scores if score > 0.15)
            if visible_core_count < 2:
                return False

            # === CHECK 3: Not impossibly proportioned ===
            body_height = avg_ankle_y - nose["y"]
            body_width = abs(left_shoulder["x"] - right_shoulder["x"])

            # Only reject if clearly impossible
            if body_height <= 0:
                return False

            if body_width > 0:
                # Aspect ratio check - very lenient range
                aspect_ratio = body_height / max(body_width, 0.01)
                if aspect_ratio < 0.5 or aspect_ratio > 10.0:
                    return False

            # === CHECK 4: Basic body structure exists ===
            torso_length = abs(avg_hip_y - avg_shoulder_y)

            # Torso must exist (even if small due to camera angle)
            if torso_length < 2:
                return False

            # === SKIP strict symmetry check ===
            # People with bags, backpacks, or standing at angles will have asymmetry
            # Only check for extreme asymmetry that would indicate non-human
            shoulder_diff = abs(left_shoulder["y"] - right_shoulder["y"])
            hip_diff = abs(left_hip["y"] - right_hip["y"])

            # Very lenient - allow significant asymmetry (e.g., bag on one shoulder)
            # Image coordinates typically 0-1 normalized, so 0.3 is very tolerant
            max_asymmetry = max(body_height * 0.4, 150)  # 40% of body height or 150 pixels
            if shoulder_diff > max_asymmetry or hip_diff > max_asymmetry:
                return False

            # All basic checks passed - likely a real human
            return True

        except (KeyError, IndexError, ZeroDivisionError):
            # If we can't check properly, give benefit of doubt
            return True
