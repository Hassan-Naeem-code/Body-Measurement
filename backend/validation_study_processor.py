"""
Validation Study Processor
Processes validation photos and compares against ground truth measurements
"""

import cv2
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
from typing import Dict, List
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from app.ml.pose_detector import PoseDetector
from app.ml.circumference_extractor_ml import MLCircumferenceExtractor
from app.ml.depth_ratio_predictor import MLDepthRatioPredictor


class ValidationStudyProcessor:
    """Process validation study photos and generate accuracy metrics"""

    def __init__(self):
        self.pose_detector = PoseDetector()
        self.ml_extractor = MLCircumferenceExtractor(use_ml_ratios=True)
        self.ratio_predictor = MLDepthRatioPredictor()

    def process_participant(
        self,
        image_path: str,
        participant_id: str,
        ground_truth: Dict[str, float]
    ) -> Dict:
        """
        Process a single participant's photo and compare to ground truth

        Args:
            image_path: Path to participant photo
            participant_id: Unique participant ID
            ground_truth: Dictionary with actual measurements

        Returns:
            Dictionary with predictions, errors, and metadata
        """
        print(f"\nProcessing Participant {participant_id}...")

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {
                'participant_id': participant_id,
                'status': 'error',
                'error': 'Could not load image'
            }

        # Detect pose
        pose_result = self.pose_detector.detect_pose(image)
        if not pose_result.success:
            return {
                'participant_id': participant_id,
                'status': 'error',
                'error': 'Could not detect pose'
            }

        # Get ML predictions
        predicted_measurements = self.ml_extractor.extract_measurements(
            pose_result.pose_landmarks,
            image
        )

        # Calculate errors
        errors = {}
        for measure in ['chest_circumference', 'waist_circumference', 'hip_circumference']:
            if measure in ground_truth and ground_truth[measure] is not None:
                predicted = getattr(predicted_measurements, measure)
                actual = ground_truth[measure]
                error = predicted - actual
                absolute_error = abs(error)
                percent_error = (absolute_error / actual * 100) if actual != 0 else 0

                errors[measure] = {
                    'predicted': predicted,
                    'actual': actual,
                    'error': error,
                    'absolute_error': absolute_error,
                    'percent_error': percent_error
                }

        # Get ML metadata
        ml_metadata = {
            'body_shape': predicted_measurements.body_shape_category,
            'bmi_estimate': predicted_measurements.bmi_estimate,
            'confidence': predicted_measurements.predicted_ratios.confidence,
            'ratios': {
                'chest': predicted_measurements.predicted_ratios.chest_ratio,
                'waist': predicted_measurements.predicted_ratios.waist_ratio,
                'hip': predicted_measurements.predicted_ratios.hip_ratio,
            }
        }

        return {
            'participant_id': participant_id,
            'status': 'success',
            'predictions': {
                'chest_circumference': predicted_measurements.chest_circumference,
                'waist_circumference': predicted_measurements.waist_circumference,
                'hip_circumference': predicted_measurements.hip_circumference,
                'shoulder_width': predicted_measurements.shoulder_width,
                'inseam': predicted_measurements.inseam,
                'estimated_height': predicted_measurements.estimated_height_cm,
            },
            'ground_truth': ground_truth,
            'errors': errors,
            'ml_metadata': ml_metadata
        }

    def process_validation_dataset(
        self,
        csv_path: str,
        photos_dir: str
    ) -> pd.DataFrame:
        """
        Process entire validation dataset from CSV

        Args:
            csv_path: Path to validation data CSV
            photos_dir: Directory containing participant photos

        Returns:
            DataFrame with results and accuracy metrics
        """
        print(f"\n{'='*80}")
        print(f"VALIDATION STUDY PROCESSOR")
        print(f"{'='*80}\n")

        # Load validation data
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} participants from {csv_path}")

        results = []

        for idx, row in df.iterrows():
            participant_id = row['participant_id']
            photo_filename = row['photo_filename']
            photo_path = os.path.join(photos_dir, photo_filename)

            # Prepare ground truth
            ground_truth = {
                'chest_circumference': row.get('measured_chest_circ_cm'),
                'waist_circumference': row.get('measured_waist_circ_cm'),
                'hip_circumference': row.get('measured_hip_circ_cm'),
                'shoulder_width': row.get('measured_shoulder_width_cm'),
                'inseam': row.get('measured_inseam_cm'),
                'height': row.get('measured_height_cm'),
                'actual_shirt_size': row.get('actual_shirt_size'),
            }

            # Process participant
            result = self.process_participant(photo_path, participant_id, ground_truth)

            # Add to original row data
            result['original_data'] = row.to_dict()
            results.append(result)

        # Create results DataFrame
        results_df = self._create_results_dataframe(results)

        # Calculate and print accuracy metrics
        self._print_accuracy_metrics(results_df)

        return results_df

    def _create_results_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """Convert results to DataFrame"""
        rows = []

        for result in results:
            if result['status'] != 'success':
                continue

            row = {
                'participant_id': result['participant_id'],
            }

            # Add predictions
            for key, val in result['predictions'].items():
                row[f'predicted_{key}'] = val

            # Add ground truth
            for key, val in result['ground_truth'].items():
                row[f'actual_{key}'] = val

            # Add errors
            for measure, error_data in result['errors'].items():
                row[f'{measure}_error'] = error_data['error']
                row[f'{measure}_abs_error'] = error_data['absolute_error']
                row[f'{measure}_pct_error'] = error_data['percent_error']

            # Add ML metadata
            row['ml_body_shape'] = result['ml_metadata']['body_shape']
            row['ml_bmi_estimate'] = result['ml_metadata']['bmi_estimate']
            row['ml_confidence'] = result['ml_metadata']['confidence']
            row['ml_chest_ratio'] = result['ml_metadata']['ratios']['chest']
            row['ml_waist_ratio'] = result['ml_metadata']['ratios']['waist']
            row['ml_hip_ratio'] = result['ml_metadata']['ratios']['hip']

            # Add original data
            for key, val in result['original_data'].items():
                if key not in row:
                    row[key] = val

            rows.append(row)

        return pd.DataFrame(rows)

    def _print_accuracy_metrics(self, df: pd.DataFrame):
        """Print comprehensive accuracy metrics"""
        print(f"\n{'='*80}")
        print(f"ACCURACY METRICS")
        print(f"{'='*80}\n")

        # Overall metrics
        print(f"📊 OVERALL PERFORMANCE\n")
        print(f"Total Participants: {len(df)}")

        for measure in ['chest_circumference', 'waist_circumference', 'hip_circumference']:
            if f'{measure}_abs_error' in df.columns:
                mae = df[f'{measure}_abs_error'].mean()
                std = df[f'{measure}_abs_error'].std()
                max_error = df[f'{measure}_abs_error'].max()
                pct_within_5cm = (df[f'{measure}_abs_error'] <= 5).sum() / len(df) * 100
                pct_within_3cm = (df[f'{measure}_abs_error'] <= 3).sum() / len(df) * 100

                print(f"\n{measure.replace('_', ' ').title()}:")
                print(f"  Mean Absolute Error (MAE): {mae:.2f} cm")
                print(f"  Std Deviation: {std:.2f} cm")
                print(f"  Max Error: {max_error:.2f} cm")
                print(f"  Within 3cm: {pct_within_3cm:.1f}%")
                print(f"  Within 5cm: {pct_within_5cm:.1f}%")

        # By body type
        if 'ml_body_shape' in df.columns:
            print(f"\n📈 PERFORMANCE BY BODY SHAPE\n")
            for shape in df['ml_body_shape'].unique():
                if pd.isna(shape):
                    continue
                shape_df = df[df['ml_body_shape'] == shape]
                chest_mae = shape_df['chest_circumference_abs_error'].mean()
                print(f"{shape.replace('_', ' ').title():20s} ({len(shape_df):3d} people): "
                      f"Chest MAE = {chest_mae:.2f} cm")

        # By gender
        if 'gender' in df.columns:
            print(f"\n👥 PERFORMANCE BY GENDER\n")
            for gender in df['gender'].unique():
                if pd.isna(gender):
                    continue
                gender_df = df[df['gender'] == gender]
                chest_mae = gender_df['chest_circumference_abs_error'].mean()
                waist_mae = gender_df['waist_circumference_abs_error'].mean()
                print(f"{gender.upper():10s} ({len(gender_df):3d} people): "
                      f"Chest MAE = {chest_mae:.2f} cm, Waist MAE = {waist_mae:.2f} cm")

        # ML confidence correlation
        if 'ml_confidence' in df.columns:
            print(f"\n🎯 ML CONFIDENCE ANALYSIS\n")
            avg_confidence = df['ml_confidence'].mean()
            print(f"Average ML Confidence: {avg_confidence:.1%}")

            # Correlation between confidence and accuracy
            if 'chest_circumference_abs_error' in df.columns:
                high_conf = df[df['ml_confidence'] > 0.80]
                low_conf = df[df['ml_confidence'] <= 0.80]

                if len(high_conf) > 0 and len(low_conf) > 0:
                    high_conf_mae = high_conf['chest_circumference_abs_error'].mean()
                    low_conf_mae = low_conf['chest_circumference_abs_error'].mean()
                    print(f"High Confidence (>0.80): {high_conf_mae:.2f} cm MAE ({len(high_conf)} people)")
                    print(f"Low Confidence (<=0.80): {low_conf_mae:.2f} cm MAE ({len(low_conf)} people)")

        print(f"\n{'='*80}\n")


def main():
    if len(sys.argv) < 3:
        print("\n📊 Validation Study Processor")
        print("="*80)
        print("\nUsage:")
        print("  python validation_study_processor.py <validation_csv> <photos_directory>")
        print("\nExample:")
        print("  python validation_study_processor.py validation_data.csv validation_photos/")
        print("\nThis will:")
        print("  ✓ Process all participant photos")
        print("  ✓ Compare predictions vs ground truth")
        print("  ✓ Calculate accuracy metrics")
        print("  ✓ Save results to validation_results.csv")
        print("  ✓ Generate accuracy report")
        print("="*80)
        return

    csv_path = sys.argv[1]
    photos_dir = sys.argv[2]

    if not os.path.exists(csv_path):
        print(f"❌ Error: CSV file not found: {csv_path}")
        return

    if not os.path.exists(photos_dir):
        print(f"❌ Error: Photos directory not found: {photos_dir}")
        return

    # Process validation dataset
    processor = ValidationStudyProcessor()
    results_df = processor.process_validation_dataset(csv_path, photos_dir)

    # Save results
    output_path = 'validation_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"✅ Results saved to: {output_path}")

    # Save summary JSON
    summary = {
        'date': datetime.now().isoformat(),
        'total_participants': len(results_df),
        'metrics': {
            'chest_mae': results_df['chest_circumference_abs_error'].mean(),
            'waist_mae': results_df['waist_circumference_abs_error'].mean(),
            'hip_mae': results_df['hip_circumference_abs_error'].mean(),
            'avg_ml_confidence': results_df['ml_confidence'].mean(),
        },
        'by_body_shape': {},
        'by_gender': {}
    }

    # Add body shape breakdown
    for shape in results_df['ml_body_shape'].unique():
        if pd.notna(shape):
            shape_df = results_df[results_df['ml_body_shape'] == shape]
            summary['by_body_shape'][shape] = {
                'count': len(shape_df),
                'chest_mae': shape_df['chest_circumference_abs_error'].mean()
            }

    # Add gender breakdown
    if 'gender' in results_df.columns:
        for gender in results_df['gender'].unique():
            if pd.notna(gender):
                gender_df = results_df[results_df['gender'] == gender]
                summary['by_gender'][gender] = {
                    'count': len(gender_df),
                    'chest_mae': gender_df['chest_circumference_abs_error'].mean(),
                    'waist_mae': gender_df['waist_circumference_abs_error'].mean()
                }

    with open('validation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Summary saved to: validation_summary.json\n")


if __name__ == "__main__":
    main()
