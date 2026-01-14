"""
Training Script for Gender Classification Model

This script:
1. Generates synthetic training data (or loads existing)
2. Trains the gender classification model
3. Evaluates performance
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

# Import from local training modules (no app dependencies)
from models.gender_classifier import (
    GenderClassifierMLP,
    GenderClassifierTrainer,
    GenderDataset,
)
from scripts.synthetic_data_generator import (
    SyntheticDataGenerator,
    prepare_training_data,
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


def main(args):
    """Main training function"""

    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'synthetic')
    model_dir = os.path.join(base_dir, 'models', 'saved')
    os.makedirs(model_dir, exist_ok=True)

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
        )
    else:
        logger.info("Loading existing synthetic dataset...")
        samples = generator.load_dataset()

    logger.info(f"Total samples: {len(samples)}")

    # Step 2: Prepare features and labels
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Preparing Features")
    logger.info("=" * 60)

    features, labels = prepare_training_data(samples)
    logger.info(f"Feature matrix shape: {features.shape}")
    logger.info(f"Labels shape: {labels.shape}")
    logger.info(f"Male samples: {np.sum(labels == 1)}")
    logger.info(f"Female samples: {np.sum(labels == 0)}")

    # Step 3: Split data
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Splitting Data")
    logger.info("=" * 60)

    X_train, X_temp, y_train, y_temp = train_test_split(
        features, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Test samples: {len(X_test)}")

    # Create datasets and dataloaders
    train_dataset = GenderDataset(X_train, y_train)
    val_dataset = GenderDataset(X_val, y_val)
    test_dataset = GenderDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Step 4: Create and train model
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Training Model")
    logger.info("=" * 60)

    model = GenderClassifierMLP(
        input_dim=features.shape[1],
        hidden_dims=[64, 32, 16]
    )

    trainer = GenderClassifierTrainer(
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

    # Step 5: Evaluate on test set
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Evaluating Model")
    logger.info("=" * 60)

    test_loss, test_accuracy = trainer.evaluate(test_loader)
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_accuracy:.2%}")

    # Detailed evaluation
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features_batch, labels_batch in test_loader:
            features_batch = features_batch.to(trainer.device)
            outputs = model(features_batch)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_batch.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(all_labels, all_preds)
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"                Predicted")
    logger.info(f"              Female  Male")
    logger.info(f"Actual Female   {cm[0,0]:5d}  {cm[0,1]:5d}")
    logger.info(f"       Male     {cm[1,0]:5d}  {cm[1,1]:5d}")

    logger.info(f"\nClassification Report:")
    report = classification_report(all_labels, all_preds, target_names=['Female', 'Male'])
    logger.info(f"\n{report}")

    # Step 6: Save model
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Saving Model")
    logger.info("=" * 60)

    model_path = os.path.join(model_dir, 'gender_classifier.pth')
    trainer.save_model(model_path)
    logger.info(f"Model saved to: {model_path}")

    # Also save the model in the main ml directory for easy access
    production_model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'gender_model.pth'
    )
    trainer.save_model(production_model_path)
    logger.info(f"Production model saved to: {production_model_path}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Final Test Accuracy: {test_accuracy:.2%}")
    logger.info(f"Model saved to: {model_path}")

    return test_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Gender Classification Model')

    parser.add_argument(
        '--num_samples', type=int, default=10000,
        help='Number of synthetic samples to generate'
    )
    parser.add_argument(
        '--regenerate_data', action='store_true',
        help='Regenerate synthetic data even if exists'
    )
    parser.add_argument(
        '--epochs', type=int, default=100,
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
        '--patience', type=int, default=15,
        help='Early stopping patience'
    )

    args = parser.parse_args()
    main(args)
