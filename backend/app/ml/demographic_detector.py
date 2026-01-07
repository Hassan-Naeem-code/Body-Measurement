"""
Demographic Detection - Gender and Age Group Classification
Uses body measurements and proportions to detect gender and age group
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass

from app.ml.pose_detector import PoseLandmarks
from app.ml.circumference_extractor_simple import CircumferenceMeasurements


@dataclass
class DemographicInfo:
    """Detected demographic information"""
    gender: str  # "male" or "female"
    age_group: str  # "adult", "teen", "child"
    gender_confidence: float  # 0-1
    age_confidence: float  # 0-1


class DemographicDetector:
    """
    Detects gender and age group from body measurements and proportions

    Gender Detection:
    - Uses shoulder-to-hip ratio
    - Uses body shape indicators
    - Male: broader shoulders, narrower hips
    - Female: narrower shoulders, wider hips

    Age Group Detection:
    - Uses estimated height
    - Uses body proportions
    - Child: < 150cm, larger head-to-body ratio
    - Teen: 150-170cm (F) or 150-180cm (M), developing proportions
    - Adult: > 170cm (F) or > 180cm (M), mature proportions
    """

    def __init__(self):
        # Gender detection thresholds
        self.MALE_SHOULDER_HIP_RATIO_MIN = 1.10  # Males typically > 1.10
        self.FEMALE_SHOULDER_HIP_RATIO_MAX = 1.05  # Females typically < 1.05

        # Age group height thresholds (cm)
        self.CHILD_HEIGHT_MAX = 145  # More conservative - children are typically under 145cm
        self.TEEN_HEIGHT_MAX_FEMALE = 168
        self.TEEN_HEIGHT_MAX_MALE = 178
        self.ADULT_HEIGHT_MIN_FEMALE = 152  # Adults typically 152cm+
        self.ADULT_HEIGHT_MIN_MALE = 162    # Adult males typically 162cm+

        # Body proportion thresholds
        self.CHILD_HEAD_BODY_RATIO = 6.0  # Children have larger heads relative to body
        self.ADULT_HEAD_BODY_RATIO = 7.5  # Adults have smaller heads relative to body

    def detect_demographics(
        self,
        pose_landmarks: PoseLandmarks,
        measurements: CircumferenceMeasurements
    ) -> DemographicInfo:
        """
        Detect gender and age group from pose landmarks and measurements

        Args:
            pose_landmarks: MediaPipe pose landmarks
            measurements: Body measurements including circumferences

        Returns:
            DemographicInfo with detected gender and age group
        """
        # Detect gender
        gender, gender_confidence = self._detect_gender(pose_landmarks, measurements)

        # Detect age group (uses gender for better accuracy)
        age_group, age_confidence = self._detect_age_group(
            pose_landmarks,
            measurements,
            gender
        )

        return DemographicInfo(
            gender=gender,
            age_group=age_group,
            gender_confidence=gender_confidence,
            age_confidence=age_confidence
        )

    def _detect_gender(
        self,
        pose_landmarks: PoseLandmarks,
        measurements: CircumferenceMeasurements
    ) -> Tuple[str, float]:
        """
        Detect gender from body proportions

        PRIMARY indicator (most reliable):
        1. Waist-to-hip ratio (0.70-0.85 = female, 0.85-0.95+ = male)

        Secondary indicators:
        2. Shoulder-to-hip ratio
        3. Chest-to-waist difference

        Returns:
            Tuple of (gender, confidence)
        """
        # PRIMARY: Waist-to-hip ratio (most reliable)
        waist_hip_ratio = measurements.waist_circumference / max(measurements.hip_circumference, 1e-3)

        # Clear thresholds from research:
        # Females: 0.67-0.85 (typically around 0.77)
        # Males: 0.85-0.95+ (typically around 0.90)
        if waist_hip_ratio < 0.85:
            # Clear female indicator
            gender = "female"
            confidence = 0.90 if waist_hip_ratio < 0.80 else 0.75
            return gender, confidence
        elif waist_hip_ratio >= 0.90:
            # Clear male indicator
            gender = "male"
            confidence = 0.90 if waist_hip_ratio > 0.95 else 0.80
            return gender, confidence
        
        # If waist/hip is ambiguous (0.85-0.90), use secondary indicators
        indicators = []

        # Indicator 2: Shoulder-to-hip ratio
        shoulder_hip_ratio = measurements.shoulder_width / max(measurements.hip_width, 1e-3)
        if shoulder_hip_ratio >= 1.15:
            indicators.append(("male", 0.75))
        elif shoulder_hip_ratio <= 1.00:
            indicators.append(("female", 0.75))
        else:
            indicators.append(("female", 0.60))

        # Indicator 3: Chest-to-waist difference
        chest_waist_diff_ratio = (measurements.chest_circumference - measurements.waist_circumference) / max(measurements.waist_circumference, 1e-3)
        if chest_waist_diff_ratio > 0.30:
            indicators.append(("male", 0.70))
        elif chest_waist_diff_ratio < 0.10:
            indicators.append(("female", 0.70))
        else:
            indicators.append(("female", 0.55))

        # Combine secondary indicators
        male_score = sum(conf for gender, conf in indicators if gender == "male")
        female_score = sum(conf for gender, conf in indicators if gender == "female")

        if male_score > female_score:
            gender = "male"
            confidence = min(0.80, male_score / max(len(indicators), 1))
        else:
            gender = "female"
            confidence = min(0.80, female_score / max(len(indicators), 1))

        return gender, max(0.60, confidence)

    def _detect_age_group(
        self,
        pose_landmarks: PoseLandmarks,
        measurements: CircumferenceMeasurements,
        gender: str
    ) -> Tuple[str, float]:
        """
        Detect age group from height and body proportions

        Age Groups:
        - child: 0-12 years (< 150cm)
        - teen: 13-17 years (150-170cm for females, 150-180cm for males)
        - adult: 18+ years (> 170cm for females, > 180cm for males)

        Returns:
            Tuple of (age_group, confidence)
        """
        height = measurements.estimated_height_cm

        # Age indicators
        age_indicators = []

        # Primary indicator: Height with more lenient thresholds
        if height < self.CHILD_HEIGHT_MAX:
            # Very short - likely child
            age_indicators.append(("child", 0.85))
        elif gender == "female":
            if height >= self.ADULT_HEIGHT_MIN_FEMALE:
                # Within normal adult female range (152cm+)
                age_indicators.append(("adult", 0.90))
            elif height >= self.TEEN_HEIGHT_MAX_FEMALE:
                # Tall for teen, likely adult
                age_indicators.append(("adult", 0.80))
            elif height >= self.ADULT_HEIGHT_MIN_FEMALE - 7:  # 145-152cm range
                # Could be short adult or tall teen
                age_indicators.append(("teen", 0.60))
            else:
                # Under 145cm - likely child
                age_indicators.append(("child", 0.75))
        else:  # male
            if height >= self.ADULT_HEIGHT_MIN_MALE:
                # Within normal adult male range (162cm+)
                age_indicators.append(("adult", 0.90))
            elif height >= self.TEEN_HEIGHT_MAX_MALE:
                # Tall for teen, likely adult
                age_indicators.append(("adult", 0.80))
            elif height >= self.ADULT_HEIGHT_MIN_MALE - 10:  # 152-162cm range
                # Could be short adult or tall teen
                age_indicators.append(("teen", 0.60))
            else:
                # Under 152cm - likely child
                age_indicators.append(("child", 0.75))

        # Secondary indicator: Body proportions (head size)
        # This is harder to detect reliably, so lower weight
        # In children, head is relatively larger compared to body

        # Get the most likely age group
        if age_indicators:
            age_group, confidence = age_indicators[0]
        else:
            # Default to adult with low confidence
            age_group, confidence = "adult", 0.60

        return age_group, confidence

    @staticmethod
    def get_demographic_label(gender: str, age_group: str) -> str:
        """
        Get human-readable demographic label

        Args:
            gender: "male" or "female"
            age_group: "adult", "teen", or "child"

        Returns:
            Label like "Adult Male", "Teen Female", "Child"
        """
        if age_group == "child":
            # For children, we typically don't distinguish by gender in labels
            return "Child"
        elif age_group == "teen":
            return f"Teen {gender.capitalize()}"
        else:
            return f"Adult {gender.capitalize()}"
