"""
Enhanced Size Recommendation Engine V2
Supports different size charts for men, women, teens, and children
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

from app.ml.measurement_extractor_v2 import BodyMeasurements


@dataclass
class SizeRecommendation:
    """Stores size recommendation results"""
    recommended_size: str
    size_probabilities: Dict[str, float]
    confidence: float
    demographic_label: str  # "Adult Male", "Teen Female", etc.


class EnhancedSizeRecommender:
    """
    Recommends clothing sizes based on body measurements and demographics
    Uses appropriate size charts for men, women, teens, and children
    """

    # Adult Male Size Chart (cm) - Circumferences
    ADULT_MALE_CHART = {
        "XS": {"chest_circ": 86, "waist_circ": 76, "hip_circ": 91, "height_range": (160, 170)},
        "S": {"chest_circ": 91, "waist_circ": 81, "hip_circ": 96, "height_range": (165, 175)},
        "M": {"chest_circ": 97, "waist_circ": 86, "hip_circ": 101, "height_range": (170, 180)},
        "L": {"chest_circ": 102, "waist_circ": 91, "hip_circ": 106, "height_range": (175, 185)},
        "XL": {"chest_circ": 107, "waist_circ": 97, "hip_circ": 111, "height_range": (180, 190)},
        "XXL": {"chest_circ": 112, "waist_circ": 102, "hip_circ": 116, "height_range": (180, 195)},
    }

    # Adult Female Size Chart (cm) - Circumferences
    ADULT_FEMALE_CHART = {
        "XS": {"chest_circ": 81, "waist_circ": 63, "hip_circ": 89, "height_range": (155, 165)},
        "S": {"chest_circ": 86, "waist_circ": 68, "hip_circ": 94, "height_range": (160, 170)},
        "M": {"chest_circ": 91, "waist_circ": 73, "hip_circ": 99, "height_range": (165, 175)},
        "L": {"chest_circ": 97, "waist_circ": 79, "hip_circ": 104, "height_range": (165, 175)},
        "XL": {"chest_circ": 102, "waist_circ": 84, "hip_circ": 109, "height_range": (170, 180)},
        "XXL": {"chest_circ": 109, "waist_circ": 91, "hip_circ": 116, "height_range": (170, 180)},
    }

    # Teen Male Size Chart (cm) - Ages 13-17
    TEEN_MALE_CHART = {
        "XS": {"chest_circ": 81, "waist_circ": 71, "hip_circ": 86, "height_range": (150, 160)},
        "S": {"chest_circ": 86, "waist_circ": 76, "hip_circ": 91, "height_range": (155, 165)},
        "M": {"chest_circ": 91, "waist_circ": 81, "hip_circ": 96, "height_range": (160, 170)},
        "L": {"chest_circ": 97, "waist_circ": 86, "hip_circ": 101, "height_range": (165, 175)},
        "XL": {"chest_circ": 102, "waist_circ": 91, "hip_circ": 106, "height_range": (170, 180)},
    }

    # Teen Female Size Chart (cm) - Ages 13-17
    TEEN_FEMALE_CHART = {
        "XS": {"chest_circ": 76, "waist_circ": 61, "hip_circ": 84, "height_range": (150, 160)},
        "S": {"chest_circ": 81, "waist_circ": 66, "hip_circ": 89, "height_range": (155, 165)},
        "M": {"chest_circ": 86, "waist_circ": 71, "hip_circ": 94, "height_range": (160, 170)},
        "L": {"chest_circ": 91, "waist_circ": 76, "hip_circ": 99, "height_range": (160, 170)},
        "XL": {"chest_circ": 97, "waist_circ": 81, "hip_circ": 104, "height_range": (165, 175)},
    }

    # Child Size Chart (cm) - Ages 4-12 (unisex)
    CHILD_CHART = {
        "4Y": {"chest_circ": 58, "waist_circ": 53, "hip_circ": 61, "height_range": (100, 110)},
        "6Y": {"chest_circ": 63, "waist_circ": 56, "hip_circ": 66, "height_range": (110, 120)},
        "8Y": {"chest_circ": 68, "waist_circ": 58, "hip_circ": 71, "height_range": (120, 130)},
        "10Y": {"chest_circ": 73, "waist_circ": 61, "hip_circ": 76, "height_range": (130, 140)},
        "12Y": {"chest_circ": 78, "waist_circ": 64, "hip_circ": 81, "height_range": (140, 150)},
        "14Y": {"chest_circ": 83, "waist_circ": 68, "hip_circ": 86, "height_range": (150, 160)},
    }

    def __init__(self):
        """Initialize the enhanced size recommender"""
        pass

    def recommend_size(
        self,
        measurements: BodyMeasurements,
        gender: str,
        age_group: str,
        demographic_label: str
    ) -> SizeRecommendation:
        """
        Recommend size based on measurements and demographics

        Args:
            measurements: Body measurements (with circumferences)
            gender: "male" or "female"
            age_group: "adult", "teen", or "child"
            demographic_label: Human-readable label (e.g., "Adult Male")

        Returns:
            SizeRecommendation with best size and probabilities
        """
        # Select appropriate size chart
        size_chart = self._get_size_chart(gender, age_group)

        # Calculate fit scores for each size
        fit_scores = self._calculate_fit_scores(measurements, size_chart)

        # Convert scores to probabilities
        size_probabilities = self._scores_to_probabilities(fit_scores)

        # Get best size
        recommended_size = max(size_probabilities.items(), key=lambda x: x[1])[0]
        confidence = size_probabilities[recommended_size]

        return SizeRecommendation(
            recommended_size=recommended_size,
            size_probabilities=size_probabilities,
            confidence=confidence,
            demographic_label=demographic_label
        )

    def _get_size_chart(self, gender: str, age_group: str) -> Dict:
        """Get the appropriate size chart based on demographics"""
        if age_group == "child":
            return self.CHILD_CHART
        elif age_group == "teen":
            if gender == "male":
                return self.TEEN_MALE_CHART
            else:
                return self.TEEN_FEMALE_CHART
        else:  # adult
            if gender == "male":
                return self.ADULT_MALE_CHART
            else:
                return self.ADULT_FEMALE_CHART

    def _calculate_fit_scores(
        self,
        measurements: BodyMeasurements,
        size_chart: Dict
    ) -> Dict[str, float]:
        """
        Calculate how well each size fits the measurements
        Lower score = better fit
        """
        fit_scores = {}

        # Check if we have circumference measurements (v3) or just widths (v2)
        has_circumferences = (
            hasattr(measurements, 'chest_circumference') and
            measurements.chest_circumference is not None
        )

        for size_name, size_spec in size_chart.items():
            distances = []

            if has_circumferences:
                # Use circumference measurements (95%+ accuracy)
                # Chest circumference (weight: 2.5 - most important)
                if "chest_circ" in size_spec:
                    chest_diff = abs(measurements.chest_circumference - size_spec["chest_circ"])
                    distances.append(chest_diff * 2.5)

                # Waist circumference (weight: 2.0)
                if "waist_circ" in size_spec:
                    waist_diff = abs(measurements.waist_circumference - size_spec["waist_circ"])
                    distances.append(waist_diff * 2.0)

                # Hip circumference (weight: 2.0)
                if "hip_circ" in size_spec:
                    hip_diff = abs(measurements.hip_circumference - size_spec["hip_circ"])
                    distances.append(hip_diff * 2.0)
            else:
                # Fallback to width measurements (backward compatibility)
                # Convert widths to approximate circumferences
                if "chest_circ" in size_spec:
                    estimated_chest_circ = measurements.chest_width * 2.0  # Rough approximation
                    chest_diff = abs(estimated_chest_circ - size_spec["chest_circ"])
                    distances.append(chest_diff * 2.0)

                if "waist_circ" in size_spec:
                    estimated_waist_circ = measurements.waist_width * 2.0
                    waist_diff = abs(estimated_waist_circ - size_spec["waist_circ"])
                    distances.append(waist_diff * 1.5)

                if "hip_circ" in size_spec:
                    estimated_hip_circ = measurements.hip_width * 2.0
                    hip_diff = abs(estimated_hip_circ - size_spec["hip_circ"])
                    distances.append(hip_diff * 1.5)

            # Height consideration (weight: 0.5 - minor factor)
            if "height_range" in size_spec and hasattr(measurements, 'estimated_height_cm'):
                height_min, height_max = size_spec["height_range"]
                if measurements.estimated_height_cm < height_min:
                    height_diff = height_min - measurements.estimated_height_cm
                    distances.append(height_diff * 0.5)
                elif measurements.estimated_height_cm > height_max:
                    height_diff = measurements.estimated_height_cm - height_max
                    distances.append(height_diff * 0.5)
                # else: within range, no penalty

            # Calculate average weighted distance
            if distances:
                fit_scores[size_name] = np.mean(distances)
            else:
                fit_scores[size_name] = float('inf')

        return fit_scores

    def _scores_to_probabilities(self, fit_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Convert fit scores to probabilities using softmax
        Better fit = higher probability
        """
        # Invert scores (lower is better -> higher is better)
        max_score = max(fit_scores.values())
        inverted_scores = {
            size: max_score - score + 1
            for size, score in fit_scores.items()
        }

        # Apply softmax with temperature
        temperature = 5.0  # Lower = more confident, higher = more distributed
        exp_scores = {
            size: np.exp(score / temperature)
            for size, score in inverted_scores.items()
        }

        total = sum(exp_scores.values())

        probabilities = {
            size: score / total
            for size, score in exp_scores.items()
        }

        return probabilities
