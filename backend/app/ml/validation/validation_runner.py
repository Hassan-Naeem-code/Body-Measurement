"""
Validation Runner - Runs measurement pipeline against ground truth data

This is the main entry point for validation testing.
Loads images, runs predictions, compares to ground truth, and generates reports.
"""

import cv2
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

from app.ml.validation.ground_truth import GroundTruthMeasurement, ValidationDataset
from app.ml.validation.accuracy_metrics import AccuracyMetrics, MetricsCalculator, calculate_accuracy_grade

logger = logging.getLogger(__name__)


@dataclass
class SingleValidationResult:
    """Result of validating a single image"""
    id: str
    image_path: str
    success: bool
    error_message: Optional[str]

    # Predicted measurements
    predicted: Dict[str, float]

    # Ground truth
    actual: Dict[str, float]

    # Per-measurement errors
    errors: Dict[str, float]  # absolute errors
    percentage_errors: Dict[str, float]

    # Metadata
    extractor_type: str
    processing_time_ms: float


@dataclass
class ValidationResult:
    """Complete validation result"""
    # Summary
    total_samples: int
    successful_samples: int
    failed_samples: int

    # Metrics
    metrics: AccuracyMetrics
    accuracy_grade: str

    # Individual results
    results: List[SingleValidationResult]

    # Metadata
    validation_date: str
    extractor_type: str
    dataset_path: str


class ValidationRunner:
    """
    Runs validation against ground truth dataset

    Usage:
        runner = ValidationRunner()
        result = runner.run_validation()
        print(runner.metrics_calculator.generate_report(result.metrics))
    """

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        use_midas_depth: bool = True,
    ):
        """
        Initialize validation runner

        Args:
            dataset_path: Path to validation dataset directory
            use_midas_depth: Whether to use MiDaS depth estimation
        """
        self.dataset = ValidationDataset(dataset_path)
        self.metrics_calculator = MetricsCalculator()
        self.use_midas_depth = use_midas_depth
        self._processor = None

    def _get_processor(self):
        """Lazy-load the measurement processor"""
        if self._processor is None:
            from app.ml.multi_person_processor_v3 import DepthBasedMultiPersonProcessor

            self._processor = DepthBasedMultiPersonProcessor(
                use_midas_depth=self.use_midas_depth,
                use_ml_ratios=True,
            )

        return self._processor

    def run_validation(
        self,
        filter_gender: Optional[str] = None,
        filter_age_group: Optional[str] = None,
        filter_body_type: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> ValidationResult:
        """
        Run validation against ground truth dataset

        Args:
            filter_gender: Only validate samples with this gender
            filter_age_group: Only validate samples with this age group
            filter_body_type: Only validate samples with this body type
            max_samples: Maximum number of samples to validate

        Returns:
            ValidationResult with all metrics and individual results
        """
        # Get ground truth data
        ground_truth = self.dataset.filter_by(
            gender=filter_gender,
            age_group=filter_age_group,
            body_type=filter_body_type,
        )

        if max_samples:
            ground_truth = ground_truth[:max_samples]

        if not ground_truth:
            logger.warning("No ground truth data found")
            return self._empty_result()

        logger.info(f"Running validation on {len(ground_truth)} samples")

        # Get processor
        processor = self._get_processor()
        extractor_type = getattr(processor, '_extractor_type', 'unknown')

        # Reset metrics calculator
        self.metrics_calculator.reset()

        # Process each sample
        results: List[SingleValidationResult] = []
        successful = 0
        failed = 0

        for gt in ground_truth:
            result = self._validate_single(gt, processor)
            results.append(result)

            if result.success:
                successful += 1
                # Add to metrics calculator
                self.metrics_calculator.add_comparison(
                    predicted=result.predicted,
                    actual=result.actual,
                    metadata={
                        'id': gt.id,
                        'gender': gt.gender,
                        'age_group': gt.age_group,
                        'body_type': gt.body_type,
                    }
                )
            else:
                failed += 1

        # Calculate metrics
        metrics = self.metrics_calculator.calculate(extractor_type)
        accuracy_grade = calculate_accuracy_grade(metrics.mape)

        return ValidationResult(
            total_samples=len(ground_truth),
            successful_samples=successful,
            failed_samples=failed,
            metrics=metrics,
            accuracy_grade=accuracy_grade,
            results=results,
            validation_date=datetime.utcnow().isoformat(),
            extractor_type=extractor_type,
            dataset_path=str(self.dataset.data_dir),
        )

    def _validate_single(
        self,
        gt: GroundTruthMeasurement,
        processor
    ) -> SingleValidationResult:
        """Validate a single ground truth sample"""
        import time

        start_time = time.time()

        try:
            # Load image
            image_path = gt.image_path
            if not os.path.isabs(image_path):
                # Try relative to dataset images dir
                image_path = os.path.join(self.dataset.images_dir, image_path)

            if not os.path.exists(image_path):
                return SingleValidationResult(
                    id=gt.id,
                    image_path=gt.image_path,
                    success=False,
                    error_message=f"Image not found: {image_path}",
                    predicted={},
                    actual=gt.get_actual_measurements(),
                    errors={},
                    percentage_errors={},
                    extractor_type=getattr(processor, '_extractor_type', 'unknown'),
                    processing_time_ms=0,
                )

            image = cv2.imread(image_path)
            if image is None:
                return SingleValidationResult(
                    id=gt.id,
                    image_path=gt.image_path,
                    success=False,
                    error_message="Failed to load image",
                    predicted={},
                    actual=gt.get_actual_measurements(),
                    errors={},
                    percentage_errors={},
                    extractor_type=getattr(processor, '_extractor_type', 'unknown'),
                    processing_time_ms=0,
                )

            # Run prediction
            result = processor.process_image(image)

            # Get measurements from first valid person
            if not result.measurements or result.valid_people_count == 0:
                return SingleValidationResult(
                    id=gt.id,
                    image_path=gt.image_path,
                    success=False,
                    error_message="No valid person detected",
                    predicted={},
                    actual=gt.get_actual_measurements(),
                    errors={},
                    percentage_errors={},
                    extractor_type=getattr(processor, '_extractor_type', 'unknown'),
                    processing_time_ms=(time.time() - start_time) * 1000,
                )

            # Get first valid person's measurements
            person = None
            for m in result.measurements:
                if m.validation_result.is_valid and m.body_measurements:
                    person = m
                    break

            if not person or not person.body_measurements:
                return SingleValidationResult(
                    id=gt.id,
                    image_path=gt.image_path,
                    success=False,
                    error_message="No body measurements extracted",
                    predicted={},
                    actual=gt.get_actual_measurements(),
                    errors={},
                    percentage_errors={},
                    extractor_type=getattr(processor, '_extractor_type', 'unknown'),
                    processing_time_ms=(time.time() - start_time) * 1000,
                )

            bm = person.body_measurements

            # Extract predicted measurements
            predicted = {
                'chest_circumference': bm.chest_circumference,
                'waist_circumference': bm.waist_circumference,
                'hip_circumference': bm.hip_circumference,
                'shoulder_width': bm.shoulder_width,
                'inseam': bm.inseam,
                'arm_length': bm.arm_length,
                'height': bm.estimated_height_cm,
            }

            # Get actual measurements
            actual = gt.get_actual_measurements()

            # Calculate errors
            errors = {}
            percentage_errors = {}

            for name in predicted.keys():
                if name in actual and predicted[name] is not None and actual[name] is not None:
                    error = abs(predicted[name] - actual[name])
                    errors[name] = error
                    percentage_errors[name] = (error / actual[name]) * 100

            processing_time = (time.time() - start_time) * 1000

            return SingleValidationResult(
                id=gt.id,
                image_path=gt.image_path,
                success=True,
                error_message=None,
                predicted=predicted,
                actual=actual,
                errors=errors,
                percentage_errors=percentage_errors,
                extractor_type=getattr(processor, '_extractor_type', 'unknown'),
                processing_time_ms=processing_time,
            )

        except Exception as e:
            logger.error(f"Error validating {gt.id}: {e}")
            return SingleValidationResult(
                id=gt.id,
                image_path=gt.image_path,
                success=False,
                error_message=str(e),
                predicted={},
                actual=gt.get_actual_measurements(),
                errors={},
                percentage_errors={},
                extractor_type=getattr(processor, '_extractor_type', 'unknown'),
                processing_time_ms=(time.time() - start_time) * 1000,
            )

    def _empty_result(self) -> ValidationResult:
        """Return empty validation result"""
        return ValidationResult(
            total_samples=0,
            successful_samples=0,
            failed_samples=0,
            metrics=self.metrics_calculator.calculate(),
            accuracy_grade="N/A",
            results=[],
            validation_date=datetime.utcnow().isoformat(),
            extractor_type="unknown",
            dataset_path=str(self.dataset.data_dir),
        )

    def save_report(self, result: ValidationResult, output_path: str) -> bool:
        """
        Save validation report to file

        Args:
            result: Validation result to save
            output_path: Path to save report (JSON)

        Returns:
            True if saved successfully
        """
        try:
            # Convert to JSON-serializable format
            report = {
                'summary': {
                    'total_samples': result.total_samples,
                    'successful_samples': result.successful_samples,
                    'failed_samples': result.failed_samples,
                    'accuracy_grade': result.accuracy_grade,
                    'validation_date': result.validation_date,
                    'extractor_type': result.extractor_type,
                },
                'metrics': {
                    'mae': result.metrics.mae,
                    'mape': result.metrics.mape,
                    'rmse': result.metrics.rmse,
                    'correlation': result.metrics.correlation,
                    'error_percentiles': result.metrics.error_percentiles,
                    'within_tolerance': result.metrics.within_tolerance,
                },
                'per_measurement': {
                    name: {
                        'mae': m.mae,
                        'mape': m.mape,
                        'rmse': m.rmse,
                        'correlation': m.correlation,
                        'sample_count': m.sample_count,
                    }
                    for name, m in result.metrics.per_measurement.items()
                },
                'individual_results': [
                    {
                        'id': r.id,
                        'success': r.success,
                        'error_message': r.error_message,
                        'predicted': r.predicted,
                        'actual': r.actual,
                        'errors': r.errors,
                        'percentage_errors': r.percentage_errors,
                        'processing_time_ms': r.processing_time_ms,
                    }
                    for r in result.results
                ],
            }

            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)

            logger.info(f"Validation report saved to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving report: {e}")
            return False

    def compare_extractors(
        self,
        max_samples: Optional[int] = None,
    ) -> Dict[str, ValidationResult]:
        """
        Compare all available extractors against ground truth

        Returns:
            Dictionary of extractor_type -> ValidationResult
        """
        from app.ml.multi_person_processor_v3 import DepthBasedMultiPersonProcessor

        results = {}

        configurations = [
            ('midas_depth', True, True),
            ('ml_ratios', False, True),
            ('simple', False, False),
        ]

        for name, use_midas, use_ml in configurations:
            logger.info(f"Testing extractor: {name}")

            try:
                processor = DepthBasedMultiPersonProcessor(
                    use_midas_depth=use_midas,
                    use_ml_ratios=use_ml,
                )

                # Run validation
                self._processor = processor
                result = self.run_validation(max_samples=max_samples)
                results[name] = result

                logger.info(
                    f"  {name}: MAE={result.metrics.mae:.2f}cm, "
                    f"MAPE={result.metrics.mape:.2f}%, "
                    f"Grade={result.accuracy_grade}"
                )

            except Exception as e:
                logger.error(f"Error testing {name}: {e}")

        return results
