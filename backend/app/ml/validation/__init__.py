"""
Validation Pipeline for Body Measurement Accuracy Testing

This module provides tools to:
1. Store ground truth measurements from real tape measurements
2. Compare predicted vs actual measurements
3. Calculate accuracy metrics (MAE, MAPE, RMSE, etc.)
4. Generate validation reports

Target: Know actual accuracy before claiming 95-98%
"""

from app.ml.validation.ground_truth import GroundTruthMeasurement, ValidationDataset
from app.ml.validation.accuracy_metrics import AccuracyMetrics, MetricsCalculator
from app.ml.validation.validation_runner import ValidationRunner, ValidationResult

__all__ = [
    "GroundTruthMeasurement",
    "ValidationDataset",
    "AccuracyMetrics",
    "MetricsCalculator",
    "ValidationRunner",
    "ValidationResult",
]
