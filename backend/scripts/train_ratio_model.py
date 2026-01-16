#!/usr/bin/env python3
"""
Training script for the depth ratio predictor model.

Usage:
    python scripts/train_ratio_model.py --epochs 100

Prerequisites:
    1. Add ground truth images to: app/data/validation/images/
    2. Edit ground truth measurements: app/data/validation/ground_truth.json
    3. Need at least 10 samples with real tape measurements

This trains a neural network to predict depth/width ratios based on
body proportions, replacing the rule-based heuristics.
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ml.trained_ratio_predictor import train_ratio_predictor


def main():
    parser = argparse.ArgumentParser(
        description='Train depth ratio predictor model'
    )
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=16,
        help='Training batch size (default: 16)'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=0.001,
        help='Initial learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--data-dir', type=str,
        default='app/data/validation',
        help='Directory containing ground_truth.json and images/'
    )
    parser.add_argument(
        '--output', type=str,
        default='app/ml/training/checkpoints/ratio_predictor.pt',
        help='Output path for trained model'
    )

    args = parser.parse_args()

    # Paths
    ground_truth_path = os.path.join(args.data_dir, 'ground_truth.json')
    images_dir = os.path.join(args.data_dir, 'images')

    # Check data exists
    if not os.path.exists(ground_truth_path):
        print(f"ERROR: Ground truth file not found: {ground_truth_path}")
        print("\nTo create training data:")
        print("1. Add images to: app/data/validation/images/")
        print("2. Edit: app/data/validation/ground_truth.json")
        print("3. Add real tape measurements for each image")
        sys.exit(1)

    if not os.path.exists(images_dir):
        print(f"ERROR: Images directory not found: {images_dir}")
        sys.exit(1)

    print("=" * 60)
    print("DEPTH RATIO PREDICTOR TRAINING")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Output: {args.output}")
    print("=" * 60)

    # Train
    metrics = train_ratio_predictor(
        ground_truth_path=ground_truth_path,
        images_dir=images_dir,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    if 'error' in metrics:
        print(f"\nERROR: {metrics['error']}")
        print("\nMake sure you have at least 10 samples with real measurements.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best validation loss: {metrics['best_val_loss']:.4f}")
    print(f"Training samples: {metrics['train_samples']}")
    print(f"Validation samples: {metrics['val_samples']}")
    print(f"\nModel saved to: {args.output}")
    print("\nTo use the trained model:")
    print("  Restart the backend server - it will automatically load the new model")


if __name__ == '__main__':
    main()
