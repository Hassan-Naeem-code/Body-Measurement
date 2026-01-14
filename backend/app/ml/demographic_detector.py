"""
Demographic Detection - Gender and Age Group Classification
Uses body measurements and proportions to detect gender and age group

Now supports trained ML model for gender detection when available.
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass
import logging

from app.ml.pose_detector import PoseLandmarks
from app.ml.circumference_extractor_simple import CircumferenceMeasurements

logger = logging.getLogger(__name__)


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

    def __init__(self, use_ml_model: bool = True):
        """
        Initialize the demographic detector

        Args:
            use_ml_model: If True, try to use trained ML model for gender detection
        """
        # Gender detection thresholds (used as fallback)
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

        # Try to load ML gender detector
        self.ml_gender_detector = None
        if use_ml_model:
            try:
                from app.ml.trained_gender_detector import get_gender_detector
                self.ml_gender_detector = get_gender_detector()
                if self.ml_gender_detector.is_model_loaded:
                    logger.info("Using trained ML model for gender detection")
                else:
                    logger.info("ML model not loaded, using rule-based gender detection")
                    self.ml_gender_detector = None
            except Exception as e:
                logger.warning(f"Could not load ML gender detector: {e}")

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

        Uses trained ML model if available, otherwise falls back to rule-based.

        PRIMARY indicator (most reliable for all body types including athletic):
        1. Shoulder-to-hip WIDTH ratio - Males have broader shoulders relative to hips

        Secondary indicators:
        2. Shoulder width absolute (males typically wider)
        3. Chest-to-waist taper (athletic males have high taper)

        Returns:
            Tuple of (gender, confidence)
        """
        # Use ML model if available
        if self.ml_gender_detector is not None:
            try:
                gender, confidence = self.ml_gender_detector.predict(
                    measurements=measurements
                )
                logger.debug(f"ML gender prediction: {gender} ({confidence:.2%})")
                return gender, confidence
            except Exception as e:
                logger.warning(f"ML gender prediction failed: {e}, using rule-based")

        # Fall back to rule-based detection
        # Collect gender scores from multiple indicators
        male_score = 0.0
        female_score = 0.0

        # PRIMARY INDICATOR: Shoulder-to-hip WIDTH ratio
        # Males: typically 1.15-1.40 (broad shoulders, narrower hips)
        # Females: typically 0.90-1.10 (shoulders similar or narrower than hips)
        shoulder_hip_ratio = measurements.shoulder_width / max(measurements.hip_width, 1e-3)

        if shoulder_hip_ratio >= 1.25:
            # Very broad shoulders - strong male indicator
            male_score += 3.0
        elif shoulder_hip_ratio >= 1.15:
            # Broad shoulders - male indicator
            male_score += 2.0
        elif shoulder_hip_ratio >= 1.05:
            # Slightly broader shoulders - slight male indicator
            male_score += 0.5
        elif shoulder_hip_ratio <= 0.95:
            # Hips wider than shoulders - strong female indicator
            female_score += 2.5
        elif shoulder_hip_ratio <= 1.00:
            # Hips similar or wider - female indicator
            female_score += 1.5
        else:
            # Ambiguous range (1.00-1.05)
            female_score += 0.5

        # SECONDARY: Absolute shoulder width
        # Males typically have wider shoulders (40-50cm)
        # Females typically have narrower shoulders (35-42cm)
        if measurements.shoulder_width >= 45:
            male_score += 1.5
        elif measurements.shoulder_width >= 42:
            male_score += 0.5
        elif measurements.shoulder_width <= 38:
            female_score += 1.0
        elif measurements.shoulder_width <= 40:
            female_score += 0.5

        # SECONDARY: Chest-to-waist taper ratio
        # Athletic males have pronounced taper (chest much larger than waist)
        # This helps distinguish athletic males from females
        chest_waist_ratio = measurements.chest_circumference / max(measurements.waist_circumference, 1e-3)

        if chest_waist_ratio >= 1.35:
            # Very high taper - athletic male
            male_score += 1.5
        elif chest_waist_ratio >= 1.20:
            # High taper - likely male
            male_score += 0.8
        elif chest_waist_ratio <= 1.05:
            # Low taper - rectangular build, could be either
            pass  # No strong indicator

        # SECONDARY: Hip circumference relative to chest
        # Females typically have larger hips relative to chest
        hip_chest_ratio = measurements.hip_circumference / max(measurements.chest_circumference, 1e-3)

        if hip_chest_ratio >= 1.10:
            # Hips larger than chest - female indicator
            female_score += 1.5
        elif hip_chest_ratio >= 1.02:
            # Slightly larger hips - slight female indicator
            female_score += 0.5
        elif hip_chest_ratio <= 0.90:
            # Chest much larger than hips - male indicator
            male_score += 1.0

        # Determine gender based on scores
        total_score = male_score + female_score
        if total_score < 1.0:
            # Very low confidence - default to female with low confidence
            return "female", 0.55

        if male_score > female_score:
            gender = "male"
            confidence = min(0.95, 0.55 + (male_score / total_score) * 0.40)
        else:
            gender = "female"
            confidence = min(0.95, 0.55 + (female_score / total_score) * 0.40)

        return gender, confidence

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
