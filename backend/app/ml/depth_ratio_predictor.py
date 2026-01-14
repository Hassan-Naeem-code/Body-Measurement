"""
ML-Based Depth Ratio Predictor
Predicts personalized depth/width ratios based on body characteristics
Replaces fixed ratios with adaptive predictions for improved accuracy
"""

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass
import logging

from app.ml.pose_detector import PoseLandmarks

logger = logging.getLogger(__name__)


@dataclass
class BodyFeatures:
    """Extracted features from pose landmarks for ratio prediction"""
    shoulder_to_hip_ratio: float       # Width ratio (indicates body shape)
    torso_length_ratio: float          # Torso/height ratio
    bmi_estimate: float                # Estimated BMI from body shape
    shoulder_width_normalized: float   # Shoulder width relative to height
    hip_width_normalized: float        # Hip width relative to height
    waist_width_normalized: float      # Waist width relative to height
    chest_to_waist_ratio: float        # Taper from chest to waist
    upper_body_mass_indicator: float   # Indicates chest/shoulder development
    lower_body_mass_indicator: float   # Indicates hip/thigh development
    body_shape_category: str           # Rectangle, Triangle, Inverted Triangle, Hourglass
    estimated_gender: str              # Male, Female (helps with typical ratios)

    # Pose quality indicators
    pose_angle: float                  # Body rotation angle
    landmark_confidence: float         # Average confidence of key landmarks


@dataclass
class DepthRatios:
    """Predicted depth/width ratios for different body parts"""
    chest_ratio: float         # Chest depth / chest width
    waist_ratio: float         # Waist depth / waist width
    hip_ratio: float           # Hip depth / hip width
    shoulder_ratio: float      # Shoulder depth / shoulder width (for future use)
    confidence: float          # Prediction confidence (0-1)
    method: str                # Prediction method used


class MLDepthRatioPredictor:
    """
    Predicts personalized depth/width ratios using body characteristics

    Strategy:
    1. Extract observable features from pose landmarks
    2. Use anthropometric knowledge + ML to predict depth ratios
    3. Start with rule-based system, upgrade to trained model later
    4. Provide confidence scores for predictions
    """

    def __init__(self):
        """Initialize the predictor"""
        # Default ratios (from CAESAR anthropometric data averages)
        self.DEFAULT_CHEST_RATIO = 0.62
        self.DEFAULT_WAIST_RATIO = 0.58
        self.DEFAULT_HIP_RATIO = 0.55

        # Known anthropometric relationships
        # Male typically: chest deeper (0.65-0.70), waist less deep (0.52-0.58)
        # Female typically: chest less deep (0.58-0.62), waist deeper (0.60-0.65)
        # Athletic: deeper chest (0.68-0.75), narrower waist (0.50-0.55)
        # Overweight: all ratios higher (0.70-0.85)

        self.MALE_RATIOS = {
            'chest': 0.67,
            'waist': 0.55,
            'hip': 0.53
        }

        self.FEMALE_RATIOS = {
            'chest': 0.60,
            'waist': 0.62,
            'hip': 0.58
        }

        self.ATHLETIC_ADJUSTMENT = {
            'chest': +0.08,   # More developed chest
            'waist': -0.05,   # Tighter waist
            'hip': -0.03      # Tighter hips
        }

        self.OVERWEIGHT_ADJUSTMENT = {
            'chest': +0.12,
            'waist': +0.15,
            'hip': +0.12
        }

        logger.info("ML Depth Ratio Predictor initialized")

    def extract_features(self, pose_landmarks: PoseLandmarks) -> BodyFeatures:
        """
        Extract body features from pose landmarks

        Args:
            pose_landmarks: MediaPipe pose landmarks

        Returns:
            BodyFeatures with all extracted characteristics
        """
        landmarks = pose_landmarks.landmarks
        img_width = pose_landmarks.image_width
        img_height = pose_landmarks.image_height

        # Helper to get landmark
        def get_lm(idx):
            return landmarks[idx]

        # Key landmarks
        left_shoulder = get_lm(11)
        right_shoulder = get_lm(12)
        left_hip = get_lm(23)
        right_hip = get_lm(24)
        nose = get_lm(0)
        left_ankle = get_lm(27)
        right_ankle = get_lm(28)

        # 1. Shoulder width (normalized to image)
        shoulder_width_px = abs(left_shoulder['x'] - right_shoulder['x'])
        shoulder_width_norm = shoulder_width_px / img_width

        # 2. Hip width (normalized)
        hip_width_px = abs(left_hip['x'] - right_hip['x'])
        hip_width_norm = hip_width_px / img_width

        # 3. Waist width (estimate between shoulders and hips)
        waist_width_norm = (shoulder_width_norm + hip_width_norm) / 2 * 0.85

        # 4. Shoulder-to-hip ratio (indicates body shape)
        shoulder_to_hip_ratio = shoulder_width_norm / max(hip_width_norm, 0.01)

        # 5. Torso length
        avg_shoulder_y = (left_shoulder['y'] + right_shoulder['y']) / 2
        avg_hip_y = (left_hip['y'] + right_hip['y']) / 2
        torso_length_px = abs(avg_hip_y - avg_shoulder_y)

        # 6. Total body height
        avg_ankle_y = (left_ankle['y'] + right_ankle['y']) / 2
        body_height_px = abs(avg_ankle_y - nose['y'])
        torso_length_ratio = torso_length_px / max(body_height_px, 1.0)

        # 7. BMI estimate (from body proportions)
        # Use waist-to-height ratio as BMI proxy (more accurate than shoulder width)
        # WHtR (Waist-to-Height Ratio): <0.4 = underweight, 0.4-0.5 = healthy, 0.5-0.6 = overweight, >0.6 = obese
        # We estimate waist width relative to body height
        if body_height_px > 0:
            # Waist-to-height ratio converted to BMI-like scale (18.5-30)
            waist_to_height_ratio = waist_width_norm / (body_height_px / img_height)
            # Also consider shoulder-to-hip ratio for body mass distribution
            body_mass_indicator = (shoulder_width_norm + hip_width_norm) / 2

            # Map ratios to BMI-like estimate
            # Thin: waist_to_height < 0.25 → BMI ~18
            # Normal: waist_to_height 0.25-0.35 → BMI ~22
            # Overweight: waist_to_height 0.35-0.45 → BMI ~27
            # Obese: waist_to_height > 0.45 → BMI ~32+
            if waist_to_height_ratio < 0.20:
                bmi_estimate = 17.5
            elif waist_to_height_ratio < 0.28:
                bmi_estimate = 18.5 + (waist_to_height_ratio - 0.20) * 50  # 18.5-22.5
            elif waist_to_height_ratio < 0.38:
                bmi_estimate = 22.5 + (waist_to_height_ratio - 0.28) * 50  # 22.5-27.5
            else:
                bmi_estimate = 27.5 + (waist_to_height_ratio - 0.38) * 40  # 27.5+

            # Adjust based on overall body mass indicator (broader = higher BMI)
            if body_mass_indicator > 0.30:
                bmi_estimate += 2.0
            elif body_mass_indicator < 0.18:
                bmi_estimate -= 2.0
        else:
            bmi_estimate = 22.0  # Default healthy BMI

        bmi_estimate = max(16.0, min(38.0, bmi_estimate))  # Clamp to reasonable range

        # 8. Chest-to-waist ratio (taper indicator)
        chest_width_norm = shoulder_width_norm * 0.88
        chest_to_waist_ratio = chest_width_norm / max(waist_width_norm, 0.01)

        # 9. Upper body mass indicator (broad shoulders = developed upper body)
        upper_body_mass = shoulder_width_norm * chest_to_waist_ratio

        # 10. Lower body mass indicator
        lower_body_mass = hip_width_norm / max(shoulder_width_norm, 0.01)

        # 11. Body shape category
        body_shape = self._classify_body_shape(
            shoulder_to_hip_ratio,
            chest_to_waist_ratio
        )

        # 12. Estimated gender (from shoulder-hip ratio)
        if shoulder_to_hip_ratio > 1.15:
            estimated_gender = 'male'  # Broader shoulders
        elif shoulder_to_hip_ratio < 0.95:
            estimated_gender = 'female'  # Broader hips
        else:
            estimated_gender = 'neutral'

        # 13. Pose angle (body rotation)
        pose_angle = self._estimate_pose_angle(
            shoulder_width_norm,
            hip_width_norm
        )

        # 14. Landmark confidence
        key_landmarks = [left_shoulder, right_shoulder, left_hip, right_hip]
        confidences = [lm.get('visibility', 0.5) for lm in key_landmarks]
        avg_confidence = sum(confidences) / len(confidences)

        return BodyFeatures(
            shoulder_to_hip_ratio=shoulder_to_hip_ratio,
            torso_length_ratio=torso_length_ratio,
            bmi_estimate=bmi_estimate,
            shoulder_width_normalized=shoulder_width_norm,
            hip_width_normalized=hip_width_norm,
            waist_width_normalized=waist_width_norm,
            chest_to_waist_ratio=chest_to_waist_ratio,
            upper_body_mass_indicator=upper_body_mass,
            lower_body_mass_indicator=lower_body_mass,
            body_shape_category=body_shape,
            estimated_gender=estimated_gender,
            pose_angle=pose_angle,
            landmark_confidence=avg_confidence
        )

    def predict_ratios(self, features: BodyFeatures) -> DepthRatios:
        """
        Predict depth/width ratios based on body features

        Args:
            features: Extracted body features

        Returns:
            DepthRatios with personalized predictions
        """
        # Start with gender-based baseline
        if features.estimated_gender == 'male':
            chest_ratio = self.MALE_RATIOS['chest']
            waist_ratio = self.MALE_RATIOS['waist']
            hip_ratio = self.MALE_RATIOS['hip']
        elif features.estimated_gender == 'female':
            chest_ratio = self.FEMALE_RATIOS['chest']
            waist_ratio = self.FEMALE_RATIOS['waist']
            hip_ratio = self.FEMALE_RATIOS['hip']
        else:
            # Neutral/unknown - use defaults
            chest_ratio = self.DEFAULT_CHEST_RATIO
            waist_ratio = self.DEFAULT_WAIST_RATIO
            hip_ratio = self.DEFAULT_HIP_RATIO

        # Adjust based on BMI estimate
        if features.bmi_estimate < 20:
            # Thin/athletic build - less depth
            chest_ratio += self.ATHLETIC_ADJUSTMENT['chest'] * 0.5
            waist_ratio += self.ATHLETIC_ADJUSTMENT['waist']
            hip_ratio += self.ATHLETIC_ADJUSTMENT['hip']
        elif features.bmi_estimate > 27:
            # Overweight - more depth
            bmi_factor = min((features.bmi_estimate - 27) / 8, 1.0)  # Cap at BMI 35
            chest_ratio += self.OVERWEIGHT_ADJUSTMENT['chest'] * bmi_factor
            waist_ratio += self.OVERWEIGHT_ADJUSTMENT['waist'] * bmi_factor
            hip_ratio += self.OVERWEIGHT_ADJUSTMENT['hip'] * bmi_factor

        # Adjust based on body shape
        if features.body_shape_category == 'inverted_triangle':
            # Broad shoulders, narrow hips - athletic upper body
            chest_ratio += 0.05
            waist_ratio -= 0.03
        elif features.body_shape_category == 'triangle':
            # Narrow shoulders, wide hips - more lower body mass
            chest_ratio -= 0.03
            hip_ratio += 0.05
        elif features.body_shape_category == 'hourglass':
            # Balanced but with curves
            waist_ratio += 0.03  # Deeper waist relative to width

        # Adjust based on chest-to-waist ratio (taper)
        if features.chest_to_waist_ratio > 1.3:
            # Strong taper = athletic/muscular
            chest_ratio += 0.04
            waist_ratio -= 0.04
        elif features.chest_to_waist_ratio < 1.1:
            # Little taper = straighter body
            chest_ratio -= 0.02
            waist_ratio += 0.03

        # Adjust based on shoulder-to-hip ratio
        if features.shoulder_to_hip_ratio > 1.2:
            # Very broad shoulders
            chest_ratio += 0.03
        elif features.shoulder_to_hip_ratio < 0.9:
            # Very wide hips
            hip_ratio += 0.04

        # Clamp ratios to reasonable ranges
        chest_ratio = max(0.50, min(0.85, chest_ratio))
        waist_ratio = max(0.45, min(0.85, waist_ratio))
        hip_ratio = max(0.45, min(0.80, hip_ratio))

        # Calculate shoulder ratio (for future use)
        shoulder_ratio = chest_ratio * 0.95  # Shoulders slightly less deep than chest

        # Confidence based on landmark quality and feature consistency
        confidence = self._calculate_confidence(features)

        logger.debug(
            f"Predicted ratios - Gender: {features.estimated_gender}, "
            f"BMI: {features.bmi_estimate:.1f}, Shape: {features.body_shape_category}, "
            f"Ratios: chest={chest_ratio:.3f}, waist={waist_ratio:.3f}, hip={hip_ratio:.3f}"
        )

        return DepthRatios(
            chest_ratio=chest_ratio,
            waist_ratio=waist_ratio,
            hip_ratio=hip_ratio,
            shoulder_ratio=shoulder_ratio,
            confidence=confidence,
            method='ml_anthropometric'
        )

    def predict_from_pose(self, pose_landmarks: PoseLandmarks) -> DepthRatios:
        """
        One-step prediction from pose landmarks

        Args:
            pose_landmarks: MediaPipe pose landmarks

        Returns:
            DepthRatios with personalized predictions
        """
        features = self.extract_features(pose_landmarks)
        return self.predict_ratios(features)

    def _classify_body_shape(
        self,
        shoulder_hip_ratio: float,
        chest_waist_ratio: float
    ) -> str:
        """Classify body shape based on ratios"""
        if shoulder_hip_ratio > 1.15:
            if chest_waist_ratio > 1.25:
                return 'inverted_triangle'  # V-shape, athletic
            else:
                return 'rectangle'  # Straight build
        elif shoulder_hip_ratio < 0.95:
            return 'triangle'  # Pear shape
        else:
            if chest_waist_ratio > 1.20:
                return 'hourglass'  # Curved
            else:
                return 'rectangle'  # Straight

    def _estimate_pose_angle(
        self,
        shoulder_width_norm: float,
        hip_width_norm: float
    ) -> float:
        """Estimate body rotation angle from visible widths"""
        avg_width = (shoulder_width_norm + hip_width_norm) / 2

        if avg_width > 0.25:
            return 0.0  # Facing camera
        elif avg_width < 0.10:
            return 75.0  # Side view
        else:
            # Linear interpolation
            return 60.0 * (1 - (avg_width - 0.10) / 0.15)

    def _calculate_confidence(self, features: BodyFeatures) -> float:
        """Calculate prediction confidence based on feature quality"""
        # Start with landmark confidence
        confidence = features.landmark_confidence

        # Reduce confidence if pose is too angled
        if features.pose_angle > 30:
            angle_penalty = (features.pose_angle - 30) / 60  # 0-1 for angles 30-90
            confidence *= (1 - angle_penalty * 0.3)

        # Reduce confidence if BMI estimate is extreme
        if features.bmi_estimate < 17 or features.bmi_estimate > 35:
            confidence *= 0.85

        # Increase confidence if body shape is clear
        if features.body_shape_category in ['inverted_triangle', 'hourglass', 'triangle']:
            confidence *= 1.05

        return min(1.0, max(0.5, confidence))

    def get_comparison_stats(
        self,
        pose_landmarks: PoseLandmarks
    ) -> Dict[str, any]:
        """
        Compare ML predictions vs fixed ratios
        Useful for analysis and validation
        """
        features = self.extract_features(pose_landmarks)
        ml_ratios = self.predict_ratios(features)

        fixed_ratios = DepthRatios(
            chest_ratio=self.DEFAULT_CHEST_RATIO,
            waist_ratio=self.DEFAULT_WAIST_RATIO,
            hip_ratio=self.DEFAULT_HIP_RATIO,
            shoulder_ratio=0.60,
            confidence=1.0,
            method='fixed'
        )

        return {
            'features': features,
            'ml_ratios': ml_ratios,
            'fixed_ratios': fixed_ratios,
            'differences': {
                'chest': ml_ratios.chest_ratio - fixed_ratios.chest_ratio,
                'waist': ml_ratios.waist_ratio - fixed_ratios.waist_ratio,
                'hip': ml_ratios.hip_ratio - fixed_ratios.hip_ratio,
            },
            'confidence': ml_ratios.confidence
        }
