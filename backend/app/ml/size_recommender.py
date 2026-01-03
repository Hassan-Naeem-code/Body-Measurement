"""
Size Recommendation Engine
Matches body measurements to clothing sizes using product size charts
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from app.ml.measurement_extractor import BodyMeasurements


@dataclass
class SizeRecommendation:
    """Stores size recommendation results"""
    recommended_size: str
    size_probabilities: Dict[str, float]
    confidence: float


class SizeRecommender:
    """
    Recommends clothing sizes based on body measurements
    Uses product size charts to find the best match
    """

    # Default size chart (cm) - used when no product-specific chart is available
    DEFAULT_SIZE_CHART = {
        "XS": {"chest": 85, "waist": 70, "hip": 88, "inseam": 76},
        "S": {"chest": 90, "waist": 75, "hip": 93, "inseam": 78},
        "M": {"chest": 95, "waist": 80, "hip": 98, "inseam": 80},
        "L": {"chest": 100, "waist": 85, "hip": 103, "inseam": 82},
        "XL": {"chest": 105, "waist": 90, "hip": 108, "inseam": 84},
        "XXL": {"chest": 110, "waist": 95, "hip": 113, "inseam": 86},
    }

    def __init__(self, size_chart: Optional[Dict] = None):
        """
        Args:
            size_chart: Product-specific size chart or None for default
        """
        self.size_chart = size_chart or self.DEFAULT_SIZE_CHART

    def recommend_size(self, measurements: BodyMeasurements) -> SizeRecommendation:
        """
        Recommend the best size based on body measurements

        Args:
            measurements: Body measurements in cm

        Returns:
            SizeRecommendation with best size and probabilities
        """
        # Calculate fit scores for each size
        fit_scores = self._calculate_fit_scores(measurements)

        # Convert scores to probabilities
        size_probabilities = self._scores_to_probabilities(fit_scores)

        # Get best size
        recommended_size = max(size_probabilities.items(), key=lambda x: x[1])[0]
        confidence = size_probabilities[recommended_size]

        return SizeRecommendation(
            recommended_size=recommended_size,
            size_probabilities=size_probabilities,
            confidence=confidence,
        )

    def _calculate_fit_scores(self, measurements: BodyMeasurements) -> Dict[str, float]:
        """
        Calculate how well each size fits the measurements
        Lower score = better fit
        """
        fit_scores = {}

        for size_name, size_measurements in self.size_chart.items():
            # Calculate weighted distance for each measurement
            distances = []

            # Chest (weight: 2.0 - most important for tops)
            if "chest" in size_measurements:
                chest_diff = abs(measurements.chest_width - size_measurements["chest"])
                distances.append(chest_diff * 2.0)

            # Waist (weight: 1.5)
            if "waist" in size_measurements:
                waist_diff = abs(measurements.waist_width - size_measurements["waist"])
                distances.append(waist_diff * 1.5)

            # Hip (weight: 1.5)
            if "hip" in size_measurements:
                hip_diff = abs(measurements.hip_width - size_measurements["hip"])
                distances.append(hip_diff * 1.5)

            # Inseam (weight: 1.0 - for pants)
            if "inseam" in size_measurements:
                inseam_diff = abs(measurements.inseam - size_measurements["inseam"])
                distances.append(inseam_diff * 1.0)

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

    def recommend_size_for_product(
        self,
        measurements: BodyMeasurements,
        product_size_chart: Dict,
    ) -> SizeRecommendation:
        """
        Recommend size for a specific product with its own size chart

        Args:
            measurements: Body measurements
            product_size_chart: Product-specific size chart

        Returns:
            SizeRecommendation
        """
        # Create temporary recommender with product chart
        temp_recommender = SizeRecommender(size_chart=product_size_chart)
        return temp_recommender.recommend_size(measurements)

    def get_size_range_recommendation(
        self, measurements: BodyMeasurements
    ) -> Tuple[str, str]:
        """
        Get recommended size range (min and max)

        Args:
            measurements: Body measurements

        Returns:
            Tuple of (min_size, max_size)
        """
        recommendation = self.recommend_size(measurements)
        probs = recommendation.size_probabilities

        # Get sizes with probability > 0.15
        likely_sizes = [
            size for size, prob in probs.items()
            if prob > 0.15
        ]

        if not likely_sizes:
            return (recommendation.recommended_size, recommendation.recommended_size)

        # Get size order
        size_order = list(self.size_chart.keys())
        likely_indices = [size_order.index(s) for s in likely_sizes if s in size_order]

        if not likely_indices:
            return (recommendation.recommended_size, recommendation.recommended_size)

        min_idx = min(likely_indices)
        max_idx = max(likely_indices)

        return (size_order[min_idx], size_order[max_idx])

    @staticmethod
    def validate_size_chart(size_chart: Dict) -> bool:
        """
        Validate that a size chart has the correct format

        Args:
            size_chart: Size chart to validate

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(size_chart, dict):
            return False

        required_fields = {"chest", "waist", "hip", "inseam"}

        for size_name, measurements in size_chart.items():
            if not isinstance(measurements, dict):
                return False

            # At least one measurement must be present
            if not any(field in measurements for field in required_fields):
                return False

            # All values must be numeric
            for value in measurements.values():
                if not isinstance(value, (int, float)):
                    return False

        return True
