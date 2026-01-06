"""
Product-Aware Size Recommendation Engine V3
Supports product-specific size charts with fallback to demographic charts
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from sqlalchemy.orm import Session

from app.ml.measurement_extractor_v2 import BodyMeasurements
from app.ml.size_recommender_v2 import EnhancedSizeRecommender
from app.models.product import Product, SizeChart


@dataclass
class ProductSizeRecommendation:
    """Stores product-specific size recommendation results"""
    recommended_size: str
    size_probabilities: Dict[str, float]
    confidence: float
    fit_quality: str  # "perfect", "good", "acceptable", "poor"
    size_scores: Dict[str, float]  # Raw fit scores (lower = better)
    alternative_sizes: Optional[List[str]] = None

    # Product info
    product_name: str = ""
    product_category: str = ""
    fit_type: str = "regular"


class ProductAwareSizeRecommender:
    """
    Recommends clothing sizes using product-specific size charts
    Falls back to demographic charts if no product specified
    """

    def __init__(self, db_session: Optional[Session] = None):
        """
        Initialize the product-aware size recommender

        Args:
            db_session: Database session for loading products (optional)
        """
        self.db_session = db_session
        self.demographic_recommender = EnhancedSizeRecommender()

    def recommend_size(
        self,
        measurements: BodyMeasurements,
        gender: str,
        age_group: str,
        demographic_label: str,
        product_id: Optional[str] = None,
        fit_preference: str = "regular",
    ) -> ProductSizeRecommendation:
        """
        Recommend size based on measurements, demographics, and optional product

        Args:
            measurements: Body measurements (with circumferences)
            gender: "male" or "female"
            age_group: "adult", "teen", or "child"
            demographic_label: Human-readable label (e.g., "Adult Male")
            product_id: Optional product ID for product-specific sizing
            fit_preference: "tight", "regular", or "loose" (for Feature #2)

        Returns:
            ProductSizeRecommendation with best size and probabilities
        """
        # Try product-specific recommendation first
        if product_id and self.db_session:
            try:
                product_rec = self._recommend_for_product(
                    measurements, product_id, fit_preference
                )
                if product_rec:
                    return product_rec
            except Exception as e:
                # Log error and fall back to demographic charts
                print(f"Product recommendation failed: {e}, falling back to demographic charts")

        # Fall back to demographic-based recommendation
        return self._recommend_demographic(
            measurements, gender, age_group, demographic_label, fit_preference
        )

    def _recommend_for_product(
        self,
        measurements: BodyMeasurements,
        product_id: str,
        fit_preference: str = "regular",
    ) -> Optional[ProductSizeRecommendation]:
        """
        Recommend size using product-specific size charts

        Args:
            measurements: Body measurements
            product_id: Product UUID
            fit_preference: Fit preference (tight/regular/loose)

        Returns:
            ProductSizeRecommendation or None if product not found
        """
        # Load product with size charts
        product = self.db_session.query(Product).filter(
            Product.id == product_id
        ).first()

        if not product or not product.size_charts:
            return None

        # Calculate fit scores for each size chart
        size_scores = {}
        size_charts_map = {}

        for size_chart in product.size_charts:
            # Use the SizeChart.matches_measurements() method
            score = size_chart.matches_measurements(
                chest=measurements.chest_circumference if hasattr(measurements, 'chest_circumference') else None,
                waist=measurements.waist_circumference if hasattr(measurements, 'waist_circumference') else None,
                hip=measurements.hip_circumference if hasattr(measurements, 'hip_circumference') else None,
                height=measurements.estimated_height_cm if hasattr(measurements, 'estimated_height_cm') else None,
            )
            size_scores[size_chart.size_name] = score
            size_charts_map[size_chart.size_name] = size_chart

        if not size_scores:
            return None

        # Get best fitting size
        best_size = min(size_scores.items(), key=lambda x: x[1])[0]
        best_score = size_scores[best_size]

        # Apply fit preference adjustment (Feature #2 ready!)
        adjusted_size = self._apply_fit_preference(
            best_size, fit_preference, list(size_scores.keys())
        )

        # Convert scores to probabilities
        size_probabilities = self._scores_to_probabilities(size_scores)

        # Determine fit quality
        fit_quality = self._determine_fit_quality(best_score)

        # Get alternative sizes
        alternative_sizes = self._get_alternative_sizes(
            adjusted_size, size_scores, size_probabilities
        )

        return ProductSizeRecommendation(
            recommended_size=adjusted_size,
            size_probabilities=size_probabilities,
            confidence=size_probabilities.get(adjusted_size, 0.0),
            fit_quality=fit_quality,
            size_scores=size_scores,
            alternative_sizes=alternative_sizes,
            product_name=product.name,
            product_category=product.category,
            fit_type=size_charts_map[adjusted_size].fit_type if adjusted_size in size_charts_map else "regular",
        )

    def _recommend_demographic(
        self,
        measurements: BodyMeasurements,
        gender: str,
        age_group: str,
        demographic_label: str,
        fit_preference: str = "regular",
    ) -> ProductSizeRecommendation:
        """
        Fall back to demographic-based recommendation

        Args:
            measurements: Body measurements
            gender: Gender
            age_group: Age group
            demographic_label: Demographic label
            fit_preference: Fit preference

        Returns:
            ProductSizeRecommendation using demographic charts
        """
        # Use the existing demographic recommender
        demo_rec = self.demographic_recommender.recommend_size(
            measurements, gender, age_group, demographic_label
        )

        # Apply fit preference adjustment
        size_chart = self.demographic_recommender._get_size_chart(gender, age_group)
        adjusted_size = self._apply_fit_preference(
            demo_rec.recommended_size, fit_preference, list(size_chart.keys())
        )

        # Convert to ProductSizeRecommendation format
        # Estimate size scores from probabilities (inverse relationship)
        max_prob = max(demo_rec.size_probabilities.values())
        size_scores = {
            size: (max_prob - prob) * 10  # Convert back to approximate scores
            for size, prob in demo_rec.size_probabilities.items()
        }

        fit_quality = "good" if demo_rec.confidence > 0.5 else "acceptable"

        alternative_sizes = self._get_alternative_sizes(
            adjusted_size, size_scores, demo_rec.size_probabilities
        )

        return ProductSizeRecommendation(
            recommended_size=adjusted_size,
            size_probabilities=demo_rec.size_probabilities,
            confidence=demo_rec.size_probabilities.get(adjusted_size, demo_rec.confidence),
            fit_quality=fit_quality,
            size_scores=size_scores,
            alternative_sizes=alternative_sizes,
            product_name="Generic",
            product_category="general",
            fit_type="regular",
        )

    def _apply_fit_preference(
        self,
        base_size: str,
        fit_preference: str,
        available_sizes: List[str]
    ) -> str:
        """
        Adjust size based on fit preference (Feature #2)

        Args:
            base_size: Original recommended size
            fit_preference: "tight", "regular", or "loose"
            available_sizes: List of available sizes (will be sorted automatically)

        Returns:
            Adjusted size based on preference
        """
        # Validate fit preference
        if fit_preference not in ["tight", "regular", "loose"]:
            fit_preference = "regular"

        if fit_preference == "regular" or not available_sizes:
            return base_size

        # Sort sizes in standard order
        sorted_sizes = self._sort_sizes(available_sizes)

        try:
            current_index = sorted_sizes.index(base_size)
        except ValueError:
            # Base size not found in available sizes
            return base_size

        if fit_preference == "tight":
            # Go down one size for tighter fit
            if current_index > 0:
                return sorted_sizes[current_index - 1]
        elif fit_preference == "loose":
            # Go up one size for looser fit
            if current_index < len(sorted_sizes) - 1:
                return sorted_sizes[current_index + 1]

        return base_size

    def _sort_sizes(self, sizes: List[str]) -> List[str]:
        """
        Sort sizes in standard order (XS -> XXL, or numeric)

        Args:
            sizes: List of size names

        Returns:
            Sorted list of sizes
        """
        # Standard size order mapping
        standard_order = {
            "XXXS": 0, "XXS": 1, "XS": 2, "S": 3, "M": 4, "L": 5, "XL": 6, "XXL": 7, "XXXL": 8,
            # Numeric child sizes
            "2Y": 10, "4Y": 11, "6Y": 12, "8Y": 13, "10Y": 14, "12Y": 15, "14Y": 16,
            # Numeric adult sizes
            "28": 20, "30": 21, "32": 22, "34": 23, "36": 24, "38": 25, "40": 26,
            "42": 27, "44": 28, "46": 29, "48": 30, "50": 31,
        }

        def get_size_order(size: str) -> int:
            """Get numeric order for a size"""
            size_upper = size.upper().strip()

            # Check standard sizes
            if size_upper in standard_order:
                return standard_order[size_upper]

            # Try to extract numeric value
            import re
            numeric_match = re.search(r'\d+', size)
            if numeric_match:
                return int(numeric_match.group()) + 100  # Offset to avoid conflicts

            # Unknown size - place at end, maintain original order
            return 1000

        # Sort using the size order function
        return sorted(sizes, key=get_size_order)

    def _determine_fit_quality(self, score: float) -> str:
        """
        Determine fit quality based on score

        Args:
            score: Fit score (lower = better)

        Returns:
            Fit quality: "perfect", "good", "acceptable", or "poor"
        """
        if score < 2.0:
            return "perfect"
        elif score < 5.0:
            return "good"
        elif score < 10.0:
            return "acceptable"
        else:
            return "poor"

    def _get_alternative_sizes(
        self,
        recommended_size: str,
        size_scores: Dict[str, float],
        size_probabilities: Dict[str, float]
    ) -> List[str]:
        """
        Get alternative sizes to consider

        Args:
            recommended_size: Main recommendation
            size_scores: Fit scores for all sizes
            size_probabilities: Probabilities for all sizes

        Returns:
            List of alternative size names (up to 2)
        """
        # Sort sizes by probability (descending)
        sorted_sizes = sorted(
            size_probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )

        alternatives = []
        for size, prob in sorted_sizes:
            if size != recommended_size and prob > 0.15:  # At least 15% probability
                alternatives.append(size)
                if len(alternatives) >= 2:
                    break

        return alternatives if alternatives else None

    def _scores_to_probabilities(self, fit_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Convert fit scores to probabilities using softmax
        Lower score = better fit = higher probability

        Args:
            fit_scores: Dictionary of size -> score (lower = better)

        Returns:
            Dictionary of size -> probability
        """
        if not fit_scores:
            return {}

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

        if total == 0:
            # Equal probabilities if all scores are the same
            prob = 1.0 / len(fit_scores)
            return {size: prob for size in fit_scores.keys()}

        probabilities = {
            size: score / total
            for size, score in exp_scores.items()
        }

        return probabilities
