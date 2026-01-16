"""
Accuracy Metrics Calculator for Body Measurement Validation

Provides comprehensive metrics to evaluate measurement accuracy:
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- RMSE (Root Mean Square Error)
- Correlation coefficient
- Per-measurement breakdown
- Error distribution analysis
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class AccuracyMetrics:
    """
    Comprehensive accuracy metrics for body measurements
    """
    # Overall metrics
    mae: float  # Mean Absolute Error (cm)
    mape: float  # Mean Absolute Percentage Error (%)
    rmse: float  # Root Mean Square Error (cm)
    correlation: float  # Pearson correlation coefficient
    sample_count: int  # Number of samples

    # Per-measurement metrics
    per_measurement: Dict[str, 'MeasurementAccuracy'] = field(default_factory=dict)

    # Error distribution
    error_percentiles: Dict[str, float] = field(default_factory=dict)  # 25th, 50th, 75th, 90th
    within_tolerance: Dict[str, float] = field(default_factory=dict)  # % within 1cm, 2cm, 3cm

    # Metadata
    extractor_type: str = "unknown"
    validation_date: str = ""


@dataclass
class MeasurementAccuracy:
    """Accuracy metrics for a single measurement type"""
    measurement_name: str
    mae: float  # Mean Absolute Error (cm)
    mape: float  # Mean Absolute Percentage Error (%)
    rmse: float  # Root Mean Square Error (cm)
    correlation: float
    sample_count: int
    mean_actual: float
    mean_predicted: float
    std_error: float  # Standard deviation of errors
    max_error: float
    min_error: float


class MetricsCalculator:
    """
    Calculator for validation metrics

    Compares predicted measurements against ground truth to compute accuracy.
    """

    # Standard measurement names for consistency
    MEASUREMENT_NAMES = [
        'chest_circumference',
        'waist_circumference',
        'hip_circumference',
        'shoulder_width',
        'inseam',
        'arm_length',
        'height',
    ]

    def __init__(self):
        self._results: List[Dict] = []

    def add_comparison(
        self,
        predicted: Dict[str, float],
        actual: Dict[str, float],
        metadata: Optional[Dict] = None
    ):
        """
        Add a single prediction vs actual comparison

        Args:
            predicted: Dict of predicted measurements (e.g., {'chest_circumference': 95.5})
            actual: Dict of actual measurements from ground truth
            metadata: Optional metadata (image_id, gender, etc.)
        """
        self._results.append({
            'predicted': predicted,
            'actual': actual,
            'metadata': metadata or {}
        })

    def calculate(self, extractor_type: str = "unknown") -> AccuracyMetrics:
        """
        Calculate comprehensive accuracy metrics

        Returns:
            AccuracyMetrics with all computed values
        """
        if not self._results:
            return self._empty_metrics(extractor_type)

        # Collect all errors by measurement type
        errors_by_type: Dict[str, List[Tuple[float, float]]] = {
            name: [] for name in self.MEASUREMENT_NAMES
        }

        for result in self._results:
            predicted = result['predicted']
            actual = result['actual']

            for name in self.MEASUREMENT_NAMES:
                if name in predicted and name in actual:
                    pred_val = predicted[name]
                    actual_val = actual[name]
                    if pred_val is not None and actual_val is not None:
                        errors_by_type[name].append((pred_val, actual_val))

        # Calculate per-measurement metrics
        per_measurement = {}
        all_errors = []
        all_actual = []
        all_predicted = []

        for name in self.MEASUREMENT_NAMES:
            pairs = errors_by_type[name]
            if len(pairs) >= 3:  # Need at least 3 samples for meaningful stats
                predicted_vals = np.array([p[0] for p in pairs])
                actual_vals = np.array([p[1] for p in pairs])
                errors = predicted_vals - actual_vals
                abs_errors = np.abs(errors)
                pct_errors = np.abs(errors / actual_vals) * 100

                # Calculate metrics
                mae = float(np.mean(abs_errors))
                mape = float(np.mean(pct_errors))
                rmse = float(np.sqrt(np.mean(errors ** 2)))

                # Correlation (handle edge cases)
                if np.std(predicted_vals) > 0 and np.std(actual_vals) > 0:
                    correlation = float(np.corrcoef(predicted_vals, actual_vals)[0, 1])
                else:
                    correlation = 0.0

                per_measurement[name] = MeasurementAccuracy(
                    measurement_name=name,
                    mae=mae,
                    mape=mape,
                    rmse=rmse,
                    correlation=correlation,
                    sample_count=len(pairs),
                    mean_actual=float(np.mean(actual_vals)),
                    mean_predicted=float(np.mean(predicted_vals)),
                    std_error=float(np.std(errors)),
                    max_error=float(np.max(abs_errors)),
                    min_error=float(np.min(abs_errors)),
                )

                # Accumulate for overall metrics
                all_errors.extend(abs_errors.tolist())
                all_actual.extend(actual_vals.tolist())
                all_predicted.extend(predicted_vals.tolist())

        # Calculate overall metrics
        if all_errors:
            all_errors = np.array(all_errors)
            all_actual = np.array(all_actual)
            all_predicted = np.array(all_predicted)

            overall_mae = float(np.mean(all_errors))
            overall_mape = float(np.mean(np.abs(all_predicted - all_actual) / all_actual) * 100)
            overall_rmse = float(np.sqrt(np.mean((all_predicted - all_actual) ** 2)))

            if np.std(all_predicted) > 0 and np.std(all_actual) > 0:
                overall_correlation = float(np.corrcoef(all_predicted, all_actual)[0, 1])
            else:
                overall_correlation = 0.0

            # Error percentiles
            error_percentiles = {
                'p25': float(np.percentile(all_errors, 25)),
                'p50': float(np.percentile(all_errors, 50)),  # Median
                'p75': float(np.percentile(all_errors, 75)),
                'p90': float(np.percentile(all_errors, 90)),
                'p95': float(np.percentile(all_errors, 95)),
            }

            # Percentage within tolerance
            within_tolerance = {
                'within_1cm': float(np.mean(all_errors <= 1.0) * 100),
                'within_2cm': float(np.mean(all_errors <= 2.0) * 100),
                'within_3cm': float(np.mean(all_errors <= 3.0) * 100),
                'within_5cm': float(np.mean(all_errors <= 5.0) * 100),
            }
        else:
            overall_mae = 0.0
            overall_mape = 0.0
            overall_rmse = 0.0
            overall_correlation = 0.0
            error_percentiles = {}
            within_tolerance = {}

        from datetime import datetime

        return AccuracyMetrics(
            mae=overall_mae,
            mape=overall_mape,
            rmse=overall_rmse,
            correlation=overall_correlation,
            sample_count=len(self._results),
            per_measurement=per_measurement,
            error_percentiles=error_percentiles,
            within_tolerance=within_tolerance,
            extractor_type=extractor_type,
            validation_date=datetime.utcnow().isoformat(),
        )

    def _empty_metrics(self, extractor_type: str) -> AccuracyMetrics:
        """Return empty metrics when no data"""
        from datetime import datetime

        return AccuracyMetrics(
            mae=0.0,
            mape=0.0,
            rmse=0.0,
            correlation=0.0,
            sample_count=0,
            per_measurement={},
            error_percentiles={},
            within_tolerance={},
            extractor_type=extractor_type,
            validation_date=datetime.utcnow().isoformat(),
        )

    def reset(self):
        """Reset collected results"""
        self._results = []

    def generate_report(self, metrics: AccuracyMetrics) -> str:
        """
        Generate a human-readable accuracy report

        Args:
            metrics: Calculated accuracy metrics

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            "BODY MEASUREMENT ACCURACY REPORT",
            "=" * 60,
            f"Extractor: {metrics.extractor_type}",
            f"Validation Date: {metrics.validation_date}",
            f"Total Samples: {metrics.sample_count}",
            "",
            "-" * 60,
            "OVERALL METRICS",
            "-" * 60,
            f"Mean Absolute Error (MAE): {metrics.mae:.2f} cm",
            f"Mean Absolute % Error (MAPE): {metrics.mape:.2f}%",
            f"Root Mean Square Error (RMSE): {metrics.rmse:.2f} cm",
            f"Correlation: {metrics.correlation:.3f}",
            "",
        ]

        # Error distribution
        if metrics.error_percentiles:
            lines.extend([
                "-" * 60,
                "ERROR DISTRIBUTION",
                "-" * 60,
                f"25th percentile: {metrics.error_percentiles.get('p25', 0):.2f} cm",
                f"Median (50th): {metrics.error_percentiles.get('p50', 0):.2f} cm",
                f"75th percentile: {metrics.error_percentiles.get('p75', 0):.2f} cm",
                f"90th percentile: {metrics.error_percentiles.get('p90', 0):.2f} cm",
                f"95th percentile: {metrics.error_percentiles.get('p95', 0):.2f} cm",
                "",
            ])

        # Tolerance
        if metrics.within_tolerance:
            lines.extend([
                "-" * 60,
                "ACCURACY BY TOLERANCE",
                "-" * 60,
                f"Within 1 cm: {metrics.within_tolerance.get('within_1cm', 0):.1f}%",
                f"Within 2 cm: {metrics.within_tolerance.get('within_2cm', 0):.1f}%",
                f"Within 3 cm: {metrics.within_tolerance.get('within_3cm', 0):.1f}%",
                f"Within 5 cm: {metrics.within_tolerance.get('within_5cm', 0):.1f}%",
                "",
            ])

        # Per-measurement breakdown
        if metrics.per_measurement:
            lines.extend([
                "-" * 60,
                "PER-MEASUREMENT BREAKDOWN",
                "-" * 60,
            ])

            for name, m in metrics.per_measurement.items():
                lines.extend([
                    f"\n{name.upper().replace('_', ' ')}:",
                    f"  Samples: {m.sample_count}",
                    f"  MAE: {m.mae:.2f} cm",
                    f"  MAPE: {m.mape:.2f}%",
                    f"  RMSE: {m.rmse:.2f} cm",
                    f"  Correlation: {m.correlation:.3f}",
                    f"  Mean Actual: {m.mean_actual:.1f} cm",
                    f"  Mean Predicted: {m.mean_predicted:.1f} cm",
                    f"  Max Error: {m.max_error:.1f} cm",
                ])

        lines.extend([
            "",
            "=" * 60,
            "END OF REPORT",
            "=" * 60,
        ])

        return "\n".join(lines)


def calculate_accuracy_grade(mape: float) -> str:
    """
    Convert MAPE to accuracy grade

    Args:
        mape: Mean Absolute Percentage Error

    Returns:
        Grade string (e.g., "A+", "B", "C")
    """
    if mape <= 2.0:
        return "A+ (Excellent: 98%+ accuracy)"
    elif mape <= 4.0:
        return "A (Very Good: 96-98% accuracy)"
    elif mape <= 6.0:
        return "B+ (Good: 94-96% accuracy)"
    elif mape <= 8.0:
        return "B (Above Average: 92-94% accuracy)"
    elif mape <= 10.0:
        return "C+ (Average: 90-92% accuracy)"
    elif mape <= 15.0:
        return "C (Below Average: 85-90% accuracy)"
    elif mape <= 20.0:
        return "D (Poor: 80-85% accuracy)"
    else:
        return "F (Failing: <80% accuracy)"
