"""
Measurement Extraction Engine
Converts pixel distances to real-world measurements in centimeters
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

from app.ml.pose_detector import PoseDetector, PoseLandmarks


@dataclass
class BodyMeasurements:
    """Stores extracted body measurements"""
    shoulder_width: float
    chest_width: float
    waist_width: float
    hip_width: float
    inseam: float
    arm_length: float
    confidence_scores: Dict[str, float]


class MeasurementExtractor:
    """
    Extracts body measurements from pose landmarks
    Converts pixel distances to centimeters using calibration
    """

    def __init__(self, reference_height_cm: float = 170.0):
        """
        Args:
            reference_height_cm: Assumed height for calibration (default: 170cm average)
        """
        self.reference_height_cm = reference_height_cm
        self.detector = PoseDetector()

    def extract_measurements(self, pose_landmarks: PoseLandmarks) -> BodyMeasurements:
        """
        Extract all body measurements from pose landmarks

        Args:
            pose_landmarks: Detected pose landmarks

        Returns:
            BodyMeasurements object with all measurements in cm
        """
        # Calculate pixel-to-cm ratio using body height
        pixels_per_cm = self._calculate_calibration(pose_landmarks)

        # Extract individual measurements
        shoulder_width = self._measure_shoulder_width(pose_landmarks, pixels_per_cm)
        chest_width = self._measure_chest_width(pose_landmarks, pixels_per_cm)
        waist_width = self._measure_waist_width(pose_landmarks, pixels_per_cm)
        hip_width = self._measure_hip_width(pose_landmarks, pixels_per_cm)
        inseam = self._measure_inseam(pose_landmarks, pixels_per_cm)
        arm_length = self._measure_arm_length(pose_landmarks, pixels_per_cm)

        # Calculate confidence scores
        confidence_scores = self._calculate_confidence_scores(pose_landmarks)

        return BodyMeasurements(
            shoulder_width=shoulder_width,
            chest_width=chest_width,
            waist_width=waist_width,
            hip_width=hip_width,
            inseam=inseam,
            arm_length=arm_length,
            confidence_scores=confidence_scores,
        )

    def _calculate_calibration(self, pose_landmarks: PoseLandmarks) -> float:
        """
        Calculate pixels per centimeter using body height
        Estimates body height from nose to ankle distance
        """
        # Get nose landmark
        nose = self.detector.get_landmark(pose_landmarks, "NOSE")

        # Get average ankle position
        left_ankle = self.detector.get_landmark(pose_landmarks, "LEFT_ANKLE")
        right_ankle = self.detector.get_landmark(pose_landmarks, "RIGHT_ANKLE")

        avg_ankle_y = (left_ankle["y"] + right_ankle["y"]) / 2

        # Calculate body height in pixels
        body_height_pixels = abs(avg_ankle_y - nose["y"])

        # Calculate pixels per cm
        pixels_per_cm = body_height_pixels / self.reference_height_cm

        return pixels_per_cm

    def _measure_shoulder_width(self, pose_landmarks: PoseLandmarks, pixels_per_cm: float) -> float:
        """Measure shoulder width in cm"""
        distance_pixels = self.detector.calculate_distance(
            pose_landmarks, "LEFT_SHOULDER", "RIGHT_SHOULDER"
        )
        return distance_pixels / pixels_per_cm

    def _measure_chest_width(self, pose_landmarks: PoseLandmarks, pixels_per_cm: float) -> float:
        """
        Measure chest width in cm
        Approximated from shoulder-to-shoulder distance with adjustment
        """
        # Use shoulders as reference, apply 0.85 multiplier for chest
        shoulder_distance = self.detector.calculate_distance(
            pose_landmarks, "LEFT_SHOULDER", "RIGHT_SHOULDER"
        )
        chest_distance = shoulder_distance * 0.85
        return chest_distance / pixels_per_cm

    def _measure_waist_width(self, pose_landmarks: PoseLandmarks, pixels_per_cm: float) -> float:
        """
        Measure waist width in cm
        Estimated from hip position with adjustment
        """
        # Calculate midpoint between shoulders and hips for waist approximation
        left_shoulder = self.detector.get_landmark(pose_landmarks, "LEFT_SHOULDER")
        right_shoulder = self.detector.get_landmark(pose_landmarks, "RIGHT_SHOULDER")
        left_hip = self.detector.get_landmark(pose_landmarks, "LEFT_HIP")
        right_hip = self.detector.get_landmark(pose_landmarks, "RIGHT_HIP")

        # Waist is approximately 70% of hip width at waist level
        hip_distance = np.sqrt(
            (left_hip["x"] - right_hip["x"])**2 + (left_hip["y"] - right_hip["y"])**2
        )
        waist_distance = hip_distance * 0.75

        return waist_distance / pixels_per_cm

    def _measure_hip_width(self, pose_landmarks: PoseLandmarks, pixels_per_cm: float) -> float:
        """Measure hip width in cm"""
        distance_pixels = self.detector.calculate_distance(
            pose_landmarks, "LEFT_HIP", "RIGHT_HIP"
        )
        return distance_pixels / pixels_per_cm

    def _measure_inseam(self, pose_landmarks: PoseLandmarks, pixels_per_cm: float) -> float:
        """
        Measure inseam (hip to ankle) in cm
        Uses average of left and right leg measurements
        """
        # Left leg
        left_hip = self.detector.get_landmark(pose_landmarks, "LEFT_HIP")
        left_ankle = self.detector.get_landmark(pose_landmarks, "LEFT_ANKLE")
        left_inseam = np.sqrt(
            (left_hip["x"] - left_ankle["x"])**2 + (left_hip["y"] - left_ankle["y"])**2
        )

        # Right leg
        right_hip = self.detector.get_landmark(pose_landmarks, "RIGHT_HIP")
        right_ankle = self.detector.get_landmark(pose_landmarks, "RIGHT_ANKLE")
        right_inseam = np.sqrt(
            (right_hip["x"] - right_ankle["x"])**2 + (right_hip["y"] - right_ankle["y"])**2
        )

        # Average of both legs
        avg_inseam = (left_inseam + right_inseam) / 2

        return avg_inseam / pixels_per_cm

    def _measure_arm_length(self, pose_landmarks: PoseLandmarks, pixels_per_cm: float) -> float:
        """
        Measure arm length (shoulder to wrist) in cm
        Uses average of left and right arm measurements
        """
        # Left arm
        left_shoulder = self.detector.get_landmark(pose_landmarks, "LEFT_SHOULDER")
        left_elbow = self.detector.get_landmark(pose_landmarks, "LEFT_ELBOW")
        left_wrist = self.detector.get_landmark(pose_landmarks, "LEFT_WRIST")

        left_upper_arm = np.sqrt(
            (left_shoulder["x"] - left_elbow["x"])**2 +
            (left_shoulder["y"] - left_elbow["y"])**2
        )
        left_forearm = np.sqrt(
            (left_elbow["x"] - left_wrist["x"])**2 +
            (left_elbow["y"] - left_wrist["y"])**2
        )
        left_arm = left_upper_arm + left_forearm

        # Right arm
        right_shoulder = self.detector.get_landmark(pose_landmarks, "RIGHT_SHOULDER")
        right_elbow = self.detector.get_landmark(pose_landmarks, "RIGHT_ELBOW")
        right_wrist = self.detector.get_landmark(pose_landmarks, "RIGHT_WRIST")

        right_upper_arm = np.sqrt(
            (right_shoulder["x"] - right_elbow["x"])**2 +
            (right_shoulder["y"] - right_elbow["y"])**2
        )
        right_forearm = np.sqrt(
            (right_elbow["x"] - right_wrist["x"])**2 +
            (right_elbow["y"] - right_wrist["y"])**2
        )
        right_arm = right_upper_arm + right_forearm

        # Average of both arms
        avg_arm_length = (left_arm + right_arm) / 2

        return avg_arm_length / pixels_per_cm

    def _calculate_confidence_scores(self, pose_landmarks: PoseLandmarks) -> Dict[str, float]:
        """
        Calculate confidence scores for each measurement
        Based on landmark visibility scores
        """
        visibility = pose_landmarks.visibility_scores

        return {
            "shoulder_width": (visibility.get("LEFT_SHOULDER", 0) + visibility.get("RIGHT_SHOULDER", 0)) / 2,
            "chest_width": (visibility.get("LEFT_SHOULDER", 0) + visibility.get("RIGHT_SHOULDER", 0)) / 2,
            "waist_width": (visibility.get("LEFT_HIP", 0) + visibility.get("RIGHT_HIP", 0)) / 2,
            "hip_width": (visibility.get("LEFT_HIP", 0) + visibility.get("RIGHT_HIP", 0)) / 2,
            "inseam": (
                visibility.get("LEFT_HIP", 0) +
                visibility.get("RIGHT_HIP", 0) +
                visibility.get("LEFT_ANKLE", 0) +
                visibility.get("RIGHT_ANKLE", 0)
            ) / 4,
            "arm_length": (
                visibility.get("LEFT_SHOULDER", 0) +
                visibility.get("RIGHT_SHOULDER", 0) +
                visibility.get("LEFT_WRIST", 0) +
                visibility.get("RIGHT_WRIST", 0)
            ) / 4,
        }
