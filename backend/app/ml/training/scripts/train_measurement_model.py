"""
Training Script for Body Measurement Prediction Model

This script:
1. Generates synthetic training data with ground truth measurements
2. Trains the measurement prediction neural network
3. Evaluates performance (MAE, MAPE per measurement)
4. Saves the trained model for production use
"""

import os
import sys
import logging
import argparse
import numpy as np

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.dirname(current_dir)
sys.path.insert(0, training_dir)

# Import from local training modules
from models.measurement_predictor import (
    MeasurementPredictorMLP,
    MeasurementPredictorTrainer,
    MeasurementDataset,
    MeasurementFeatureExtractor,
)
from scripts.synthetic_data_generator import (
    SyntheticDataGenerator,
    SyntheticBodyData,
)

# Torch imports
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_measurement_training_data(samples: list) -> tuple:
    """
    Prepare feature matrix and measurement labels for training

    Returns:
        Tuple of (features, measurements) where:
        - features: (N, 24) numpy array of pose features
        - measurements: (N, 6) numpy array of target measurements in cm
    """
    feature_extractor = MeasurementFeatureExtractor()

    features = []
    measurements = []

    for sample in samples:
        # Extract features using the feature extractor
        feature_vector = feature_extractor.extract_features_from_synthetic(sample)
        features.append(feature_vector)

        # Target measurements (in cm)
        measurement_vector = [
            sample.chest_circumference,
            sample.waist_circumference,
            sample.hip_circumference,
            sample.shoulder_width,
            sample.inseam,
            sample.arm_length,
        ]
        measurements.append(measurement_vector)

    return np.array(features, dtype=np.float32), np.array(measurements, dtype=np.float32)


def evaluate_detailed(model, dataloader, device) -> dict:
    """Detailed evaluation with per-measurement metrics"""
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            predictions = model(features)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    measurement_names = [
        'chest_circumference',
        'waist_circumference',
        'hip_circumference',
        'shoulder_width',
        'inseam',
        'arm_length',
    ]

    results = {}
    for i, name in enumerate(measurement_names):
        pred = predictions[:, i]
        true = targets[:, i]

        mae = np.abs(pred - true).mean()
        mape = (np.abs(pred - true) / (true + 1e-6) * 100).mean()
        rmse = np.sqrt(((pred - true) ** 2).mean())

        results[name] = {
            'mae_cm': round(mae, 2),
            'mape_percent': round(mape, 1),
            'rmse_cm': round(rmse, 2),
        }

    # Overall metrics
    overall_mae = np.abs(predictions - targets).mean()
    overall_mape = (np.abs(predictions - targets) / (targets + 1e-6) * 100).mean()

    results['overall'] = {
        'mae_cm': round(overall_mae, 2),
        'mape_percent': round(overall_mape, 1),
    }

    return results


def main(args):
    """Main training function"""

    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'synthetic')
    checkpoint_dir = os.path.join(base_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Step 1: Generate or load synthetic data
    logger.info("=" * 60)
    logger.info("STEP 1: Preparing Training Data")
    logger.info("=" * 60)

    generator = SyntheticDataGenerator(output_dir=data_dir)

    if args.regenerate_data or not os.path.exists(os.path.join(data_dir, 'synthetic_dataset.json')):
        logger.info(f"Generating {args.num_samples} synthetic samples...")
        samples = generator.generate_dataset(
            num_samples=args.num_samples,
            male_ratio=0.5,
            body_type_distribution={
                'slim': 0.20,
                'average': 0.40,
                'athletic': 0.25,
                'heavy': 0.15,
            }
        )
    else:
        logger.info("Loading existing synthetic dataset...")
        samples = generator.load_dataset()

    logger.info(f"Total samples: {len(samples)}")

    # Step 2: Prepare features and measurements
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Extracting Features")
    logger.info("=" * 60)

    features, measurements = prepare_measurement_training_data(samples)
    logger.info(f"Feature matrix shape: {features.shape}")
    logger.info(f"Measurement matrix shape: {measurements.shape}")

    # Print measurement statistics
    measurement_names = ['chest', 'waist', 'hip', 'shoulder', 'inseam', 'arm']
    logger.info("\nMeasurement statistics (cm):")
    for i, name in enumerate(measurement_names):
        values = measurements[:, i]
        logger.info(f"  {name:10s}: mean={values.mean():.1f}, std={values.std():.1f}, "
                   f"min={values.min():.1f}, max={values.max():.1f}")

    # Step 3: Split data
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Splitting Data")
    logger.info("=" * 60)

    X_train, X_temp, y_train, y_temp = train_test_split(
        features, measurements, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Test samples: {len(X_test)}")

    # Create datasets and dataloaders
    train_dataset = MeasurementDataset(X_train, y_train)
    val_dataset = MeasurementDataset(X_val, y_val)
    test_dataset = MeasurementDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Step 4: Create and train model
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Training Model")
    logger.info("=" * 60)

    model = MeasurementPredictorMLP(
        input_dim=features.shape[1],
        output_dim=measurements.shape[1],
        hidden_dims=[128, 256, 128, 64],
        dropout_rate=args.dropout,
    )

    trainer = MeasurementPredictorTrainer(
        model=model,
        learning_rate=args.learning_rate,
    )

    logger.info(f"Model architecture:")
    logger.info(model)
    logger.info(f"\nTraining for up to {args.epochs} epochs...")

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        early_stopping_patience=args.patience,
    )

    # Step 5: Detailed evaluation on test set
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Evaluating Model")
    logger.info("=" * 60)

    test_results = evaluate_detailed(model, test_loader, trainer.device)

    logger.info("\nPer-measurement results on test set:")
    logger.info("-" * 50)
    for name in ['chest_circumference', 'waist_circumference', 'hip_circumference',
                 'shoulder_width', 'inseam', 'arm_length']:
        r = test_results[name]
        logger.info(f"  {name:20s}: MAE={r['mae_cm']:.2f}cm, MAPE={r['mape_percent']:.1f}%, "
                   f"RMSE={r['rmse_cm']:.2f}cm")

    logger.info("-" * 50)
    logger.info(f"  {'OVERALL':20s}: MAE={test_results['overall']['mae_cm']:.2f}cm, "
               f"MAPE={test_results['overall']['mape_percent']:.1f}%")

    # Step 6: Save model
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Saving Model")
    logger.info("=" * 60)

    model_path = os.path.join(checkpoint_dir, 'measurement_predictor.pt')
    trainer.save_model(model_path)
    logger.info(f"Model saved to: {model_path}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Overall MAE: {test_results['overall']['mae_cm']:.2f} cm")
    logger.info(f"Overall MAPE: {test_results['overall']['mape_percent']:.1f}%")
    logger.info(f"Model saved to: {model_path}")

    # Quality assessment
    overall_mae = test_results['overall']['mae_cm']
    if overall_mae < 2.0:
        quality = "EXCELLENT - Ready for production"
    elif overall_mae < 3.5:
        quality = "GOOD - Acceptable for production"
    elif overall_mae < 5.0:
        quality = "MODERATE - May need more training data"
    else:
        quality = "NEEDS IMPROVEMENT - Consider more data or architecture changes"

    logger.info(f"Quality Assessment: {quality}")

    return test_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Body Measurement Prediction Model')

    parser.add_argument(
        '--num_samples', type=int, default=10000,
        help='Number of synthetic samples to generate'
    )
    parser.add_argument(
        '--regenerate_data', action='store_true',
        help='Regenerate synthetic data even if exists'
    )
    parser.add_argument(
        '--epochs', type=int, default=200,
        help='Maximum number of training epochs'
    )
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='Training batch size'
    )
    parser.add_argument(
        '--learning_rate', type=float, default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--dropout', type=float, default=0.2,
        help='Dropout rate'
    )
    parser.add_argument(
        '--patience', type=int, default=20,
        help='Early stopping patience'
    )

    args = parser.parse_args()
    main(args)
