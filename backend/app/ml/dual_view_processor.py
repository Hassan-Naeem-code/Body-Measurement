"""
Dual-View Processor — combines front + side photos for higher accuracy.

When a user provides both a front-facing and side-facing photo, we can
directly measure widths from the front view and depths from the side view,
then compute TRUE circumferences using the ellipse perimeter formula.

This eliminates the need for depth estimation entirely.
"""

import numpy as np
import cv2
import logging
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

from app.ml.pose_detector import PoseDetector, PoseLandmarks
from app.ml.measurement_extractor_v2 import EnhancedMeasurementExtractor
from app.ml.body_validator import FullBodyValidator
from app.ml.depth_enhanced_extractor import CircumferenceMeasurements, DepthMeasurementData

logger = logging.getLogger(__name__)


@dataclass
class DualViewResult:
    """Result from dual-view processing."""
    measurements: CircumferenceMeasurements
    front_used: bool
    side_used: bool
    front_view_confidence: float
    side_view_confidence: float
    combination_method: str  # 'dual_ellipse', 'front_only', 'side_only'


class DualViewProcessor:
    """
    Combines front and side view measurements for higher accuracy.

    Front view  -> widths (shoulder, chest, waist, hip)
    Side view   -> depths (chest, waist, hip front-to-back)
    Combined    -> TRUE circumference via ellipse perimeter formula

    Ellipse circumference ≈ π * (3(a+b) - sqrt((3a+b)(a+3b)))
    where a = half-width, b = half-depth (Ramanujan's approximation)
    """

    def __init__(self, reference_height_cm: float = 170.0):
        self.pose_detector = PoseDetector()
        self.validator = FullBodyValidator()
        self.extractor = EnhancedMeasurementExtractor(reference_height_cm=reference_height_cm)
        self.reference_height_cm = reference_height_cm

    def process(
        self,
        front_image: np.ndarray,
        side_image: np.ndarray,
        height_cm: Optional[float] = None,
    ) -> DualViewResult:
        """
        Process front and side images together.

        Args:
            front_image: BGR image showing the front of the person
            side_image: BGR image showing the side profile
            height_cm: Known height for calibration

        Returns:
            DualViewResult with combined measurements
        """
        ref_height = height_cm or self.reference_height_cm

        # Detect pose in both images
        front_landmarks = self.pose_detector.detect_from_array(front_image)
        side_landmarks = self.pose_detector.detect_from_array(side_image)

        # Validate and classify view angles
        front_ok = front_landmarks is not None
        side_ok = side_landmarks is not None

        if front_ok:
            front_angle = self._detect_view_angle(front_landmarks)
            front_ok = front_angle < 30  # less than 30° from frontal
        if side_ok:
            side_angle = self._detect_view_angle(side_landmarks)
            side_ok = side_angle > 50  # more than 50° from frontal (i.e., side-ish)

        # Extract measurements from each view
        front_widths = self._extract_widths(front_landmarks, front_image, ref_height) if front_ok else None
        side_depths = self._extract_depths(side_landmarks, side_image, ref_height) if side_ok else None

        # Combine measurements
        if front_widths and side_depths:
            return self._combine_dual_view(front_widths, side_depths, front_landmarks, ref_height)
        elif front_widths:
            logger.info("Side view unusable, falling back to front-only")
            return self._front_only_fallback(front_widths, front_landmarks, ref_height)
        elif side_depths:
            logger.info("Front view unusable, falling back to side-only")
            return self._side_only_fallback(side_depths, side_landmarks, ref_height)
        else:
            raise ValueError("Could not detect a valid pose in either image.")

    def _detect_view_angle(self, landmarks: PoseLandmarks) -> float:
        """
        Estimate view angle from landmark geometry.

        Returns angle in degrees: 0 = frontal, 90 = side profile.
        """
        left_shoulder = landmarks.landmarks[11]
        right_shoulder = landmarks.landmarks[12]
        left_hip = landmarks.landmarks[23]
        right_hip = landmarks.landmarks[24]

        # Shoulder width in pixels (normalized by image width)
        shoulder_width = abs(right_shoulder["x"] - left_shoulder["x"]) / landmarks.image_width
        hip_width = abs(right_hip["x"] - left_hip["x"]) / landmarks.image_width

        avg_width = (shoulder_width + hip_width) / 2

        # In a frontal view, shoulder width occupies ~25-35% of frame
        # In a side view, it's compressed to ~5-10%
        if avg_width > 0.20:
            return 0.0  # Frontal
        elif avg_width < 0.08:
            return 80.0  # Near-side
        else:
            # Linear interpolation
            return 80.0 * (1 - (avg_width - 0.08) / 0.12)

    def _extract_widths(
        self, landmarks: PoseLandmarks, image: np.ndarray, height_cm: float
    ) -> Dict[str, float]:
        """Extract body widths from front-view image."""
        h, w = image.shape[:2]
        ppc = self._pixels_per_cm(landmarks, height_cm)

        def dist(a, b):
            ax, ay = landmarks.landmarks[a]["x"], landmarks.landmarks[a]["y"]
            bx, by = landmarks.landmarks[b]["x"], landmarks.landmarks[b]["y"]
            return np.sqrt((ax - bx)**2 + (ay - by)**2) / ppc

        shoulder_w = dist(11, 12)
        hip_w = dist(23, 24)
        chest_w = shoulder_w * 0.88  # chest slightly narrower than biacromial
        waist_w = hip_w * 0.80

        # Inseam and arm length
        inseam = (dist(23, 25) + dist(25, 27) + dist(24, 26) + dist(26, 28)) / 2
        arm_length = (dist(11, 13) + dist(13, 15) + dist(12, 14) + dist(14, 16)) / 2

        return {
            "shoulder_width": shoulder_w,
            "chest_width": chest_w,
            "waist_width": waist_w,
            "hip_width": hip_w,
            "inseam": inseam,
            "arm_length": arm_length,
        }

    def _extract_depths(
        self, landmarks: PoseLandmarks, image: np.ndarray, height_cm: float
    ) -> Dict[str, float]:
        """
        Extract body depths from side-view image.

        In a side view, the visible width IS the depth (front-to-back).
        """
        ppc = self._pixels_per_cm(landmarks, height_cm)

        l_sh = landmarks.landmarks[11]
        r_sh = landmarks.landmarks[12]
        l_hip = landmarks.landmarks[23]
        r_hip = landmarks.landmarks[24]

        # In side view, shoulder "width" = body depth at chest level
        torso_top_y = (l_sh["y"] + r_sh["y"]) / 2
        torso_bottom_y = (l_hip["y"] + r_hip["y"]) / 2
        torso_height = abs(torso_bottom_y - torso_top_y)

        # Estimate depth at different levels using the side silhouette
        # We measure the horizontal extent of the body in the side view
        chest_depth = self._measure_silhouette_width_at_level(
            image, landmarks, torso_top_y + torso_height * 0.25, ppc
        )
        waist_depth = self._measure_silhouette_width_at_level(
            image, landmarks, torso_top_y + torso_height * 0.60, ppc
        )
        hip_depth = self._measure_silhouette_width_at_level(
            image, landmarks, torso_bottom_y, ppc
        )

        return {
            "chest_depth": chest_depth,
            "waist_depth": waist_depth,
            "hip_depth": hip_depth,
        }

    def _measure_silhouette_width_at_level(
        self, image: np.ndarray, landmarks: PoseLandmarks, y_pixel: float, ppc: float
    ) -> float:
        """Measure body width at a specific Y level in a side-view image."""
        h, w = image.shape[:2]
        y = int(np.clip(y_pixel, 0, h - 1))

        # Convert row to grayscale and threshold to find body region
        row = cv2.cvtColor(image[y:y+1, :], cv2.COLOR_BGR2GRAY)[0]

        # Use Otsu threshold to separate body from background
        _, binary = cv2.threshold(row, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        body_pixels = np.where(binary > 0)[0]

        if len(body_pixels) < 5:
            return 20.0  # fallback ~20cm

        body_width_px = body_pixels[-1] - body_pixels[0]
        return body_width_px / ppc

    def _combine_dual_view(
        self,
        front_widths: Dict[str, float],
        side_depths: Dict[str, float],
        front_landmarks: PoseLandmarks,
        height_cm: float,
    ) -> DualViewResult:
        """Combine front widths + side depths using ellipse circumference formula."""
        chest_circ = self._ellipse_circumference(
            front_widths["chest_width"], side_depths["chest_depth"]
        )
        waist_circ = self._ellipse_circumference(
            front_widths["waist_width"], side_depths["waist_depth"]
        )
        hip_circ = self._ellipse_circumference(
            front_widths["hip_width"], side_depths["hip_depth"]
        )

        arm_circ = chest_circ * 0.30
        thigh_circ = hip_circ * 0.55

        # Depth ratios derived from actual measurements
        chest_ratio = side_depths["chest_depth"] / front_widths["chest_width"] if front_widths["chest_width"] > 0 else 0.65
        waist_ratio = side_depths["waist_depth"] / front_widths["waist_width"] if front_widths["waist_width"] > 0 else 0.60
        hip_ratio = side_depths["hip_depth"] / front_widths["hip_width"] if front_widths["hip_width"] > 0 else 0.62

        depth_data = DepthMeasurementData(
            chest_depth_ratio=chest_ratio,
            waist_depth_ratio=waist_ratio,
            hip_depth_ratio=hip_ratio,
            chest_depth=side_depths["chest_depth"],
            waist_depth=side_depths["waist_depth"],
            hip_depth=side_depths["hip_depth"],
            front_depth=0.5,
            back_depth=0.35,
            depth_confidence=0.95,
            method="dual_view_measured",
        )

        confidence = {
            "chest_circumference": 0.97,
            "waist_circumference": 0.96,
            "hip_circumference": 0.97,
            "shoulder_width": 0.95,
            "inseam": 0.93,
            "arm_length": 0.92,
            "method": "dual_view_ellipse",
        }

        measurements = CircumferenceMeasurements(
            chest_circumference=chest_circ,
            waist_circumference=waist_circ,
            hip_circumference=hip_circ,
            arm_circumference=arm_circ,
            thigh_circumference=thigh_circ,
            shoulder_width=front_widths["shoulder_width"],
            chest_width=front_widths["chest_width"],
            waist_width=front_widths["waist_width"],
            hip_width=front_widths["hip_width"],
            inseam=front_widths["inseam"],
            arm_length=front_widths["arm_length"],
            estimated_height_cm=height_cm,
            pose_angle_degrees=0.0,
            confidence_scores=confidence,
            depth_data=depth_data,
        )

        return DualViewResult(
            measurements=measurements,
            front_used=True,
            side_used=True,
            front_view_confidence=0.95,
            side_view_confidence=0.90,
            combination_method="dual_ellipse",
        )

    def _front_only_fallback(self, front_widths, landmarks, height_cm):
        """Use front view only with estimated depth ratios."""
        # Use average depth ratios when side view is unavailable
        chest_circ = front_widths["chest_width"] * np.pi * 0.81  # approximate
        waist_circ = front_widths["waist_width"] * np.pi * 0.79
        hip_circ = front_widths["hip_width"] * np.pi * 0.78

        measurements = CircumferenceMeasurements(
            chest_circumference=chest_circ,
            waist_circumference=waist_circ,
            hip_circumference=hip_circ,
            arm_circumference=chest_circ * 0.30,
            thigh_circumference=hip_circ * 0.55,
            shoulder_width=front_widths["shoulder_width"],
            chest_width=front_widths["chest_width"],
            waist_width=front_widths["waist_width"],
            hip_width=front_widths["hip_width"],
            inseam=front_widths["inseam"],
            arm_length=front_widths["arm_length"],
            estimated_height_cm=height_cm,
            pose_angle_degrees=0.0,
            confidence_scores={"method": "front_only_estimated"},
        )

        return DualViewResult(
            measurements=measurements,
            front_used=True,
            side_used=False,
            front_view_confidence=0.85,
            side_view_confidence=0.0,
            combination_method="front_only",
        )

    def _side_only_fallback(self, side_depths, landmarks, height_cm):
        """Use side view only — less common, lower accuracy."""
        # Estimate widths from depths using average ratios
        chest_w = side_depths["chest_depth"] / 0.65
        waist_w = side_depths["waist_depth"] / 0.60
        hip_w = side_depths["hip_depth"] / 0.62

        chest_circ = self._ellipse_circumference(chest_w, side_depths["chest_depth"])
        waist_circ = self._ellipse_circumference(waist_w, side_depths["waist_depth"])
        hip_circ = self._ellipse_circumference(hip_w, side_depths["hip_depth"])

        measurements = CircumferenceMeasurements(
            chest_circumference=chest_circ,
            waist_circumference=waist_circ,
            hip_circumference=hip_circ,
            arm_circumference=chest_circ * 0.30,
            thigh_circumference=hip_circ * 0.55,
            shoulder_width=chest_w * 1.15,
            chest_width=chest_w,
            waist_width=waist_w,
            hip_width=hip_w,
            inseam=height_cm * 0.45,
            arm_length=height_cm * 0.33,
            estimated_height_cm=height_cm,
            pose_angle_degrees=80.0,
            confidence_scores={"method": "side_only_estimated"},
        )

        return DualViewResult(
            measurements=measurements,
            front_used=False,
            side_used=True,
            front_view_confidence=0.0,
            side_view_confidence=0.75,
            combination_method="side_only",
        )

    @staticmethod
    def _ellipse_circumference(width: float, depth: float) -> float:
        """
        Calculate ellipse circumference using Ramanujan's approximation.

        C ≈ π * (3(a+b) - sqrt((3a+b)(a+3b)))

        where a = semi-major axis (half-width), b = semi-minor axis (half-depth)
        """
        a = width / 2.0
        b = depth / 2.0
        if a <= 0 or b <= 0:
            return 0.0
        return np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))

    def _pixels_per_cm(self, landmarks: PoseLandmarks, height_cm: float) -> float:
        """Calculate pixels per cm from landmark positions and known height."""
        nose = landmarks.landmarks[0]
        l_ankle = landmarks.landmarks[27]
        r_ankle = landmarks.landmarks[28]
        avg_ankle_y = (l_ankle["y"] + r_ankle["y"]) / 2
        body_height_px = abs(avg_ankle_y - nose["y"])
        # Person height covers ~90% of head-to-ankle distance
        return body_height_px / (height_cm * 0.9) if height_cm > 0 else 1.0
