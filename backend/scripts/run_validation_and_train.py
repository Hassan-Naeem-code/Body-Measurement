#!/usr/bin/env python3
"""
Validation Runner + Confidence Model Trainer.

1. Processes ground truth images through the ML pipeline
2. Compares predicted vs actual measurements
3. Generates synthetic training data from measurement-only samples
4. Trains the confidence prediction model

Usage (run inside Docker):
  python -m scripts.run_validation_and_train

Run from the backend/ directory.
"""

import os
import sys
import json
import logging
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.core.database import SessionLocal
from app.models.ground_truth import GroundTruth
from app.ml.confidence_predictor import ConfidenceNet, MEASUREMENT_NAMES, EXTRACTOR_QUALITY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "app", "ml", "training", "checkpoints"
)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

OUTPUT_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "confidence_predictor.pt")
VALIDATION_RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data", "validation", "validation_results.json"
)
os.makedirs(os.path.dirname(VALIDATION_RESULTS_PATH), exist_ok=True)


# Mapping from ground truth DB fields to MEASUREMENT_NAMES
GT_FIELD_MAP = {
    "chest_circumference": "actual_chest_circumference",
    "waist_circumference": "actual_waist_circumference",
    "hip_circumference": "actual_hip_circumference",
    "shoulder_width": "actual_shoulder_width",
    "inseam": "actual_inseam",
    "arm_length": "actual_arm_length",
}


def run_pipeline_on_image(image_path: str) -> dict:
    """Run the ML pipeline on a single image and return predictions + metadata."""
    from app.ml.pose_detector import PoseDetector
    from app.ml.measurement_extractor_v2 import EnhancedMeasurementExtractor
    from app.ml.body_validator import FullBodyValidator

    image = cv2.imread(image_path)
    if image is None:
        return None

    detector = PoseDetector()
    validator = FullBodyValidator()
    extractor = EnhancedMeasurementExtractor()

    # Detect pose
    landmarks = detector.detect_from_array(image)
    if landmarks is None:
        return None

    h, w = image.shape[:2]

    # Validate
    validation = validator.validate_full_body(landmarks)

    # Extract measurements
    measurements_obj = extractor.extract_measurements(landmarks, image)
    measurements = vars(measurements_obj) if hasattr(measurements_obj, '__dict__') else measurements_obj

    # Compute visibility scores
    visibility = {}
    for i, lm in enumerate(landmarks.landmarks):
        visibility[f"landmark_{i}"] = lm.get("visibility", 0.5)

    avg_vis = np.mean(list(visibility.values()))

    # Body-to-frame ratio
    xs = [lm["x"] for lm in landmarks.landmarks]
    ys = [lm["y"] for lm in landmarks.landmarks]
    body_width = (max(xs) - min(xs)) / w
    body_height = (max(ys) - min(ys)) / h
    frame_ratio = body_width * body_height

    return {
        "measurements": measurements,
        "visibility_scores": visibility,
        "body_to_frame_ratio": float(frame_ratio),
        "pose_angle_degrees": float(measurements.get("pose_angle_degrees", 0.0)),
        "avg_visibility": float(avg_vis),
        "extractor_method": "ml_anthropometric",
    }


def process_real_images():
    """Process ground truth samples that have real images."""
    db = SessionLocal()
    samples = []

    real_gts = db.query(GroundTruth).filter(
        GroundTruth.measured_by == "trainingdatapro_dataset"
    ).all()

    logger.info("Processing %d real images...", len(real_gts))

    for i, gt in enumerate(real_gts):
        if not gt.image_path or not os.path.exists(gt.image_path):
            continue

        logger.info("  [%d/%d] %s", i + 1, len(real_gts), os.path.basename(gt.image_path))

        result = run_pipeline_on_image(gt.image_path)
        if result is None:
            logger.warning("    Pipeline failed, skipping")
            continue

        sample = {
            "image_path": gt.image_path,
            "source": "trainingdatapro",
            "visibility_scores": result["visibility_scores"],
            "body_to_frame_ratio": result["body_to_frame_ratio"],
            "pose_angle_degrees": result["pose_angle_degrees"],
            "extractor_method": result["extractor_method"],
        }

        # Add predicted and actual measurements
        pred = result["measurements"]
        for name in MEASUREMENT_NAMES:
            gt_field = GT_FIELD_MAP[name]
            actual = getattr(gt, gt_field, None)
            predicted = pred.get(name, 0) or pred.get(name + "_cm", 0)

            sample[f"predicted_{name}"] = float(predicted) if predicted else 0
            sample[f"actual_{name}"] = float(actual) if actual else 0

        samples.append(sample)

    db.close()
    logger.info("Processed %d real images successfully", len(samples))
    return samples


def generate_synthetic_training_data(n_samples: int = 2000):
    """
    Generate synthetic training data from BodyM measurement-only samples.

    Since BodyM doesn't have images we can run through our pipeline, we simulate
    different quality conditions and compute what the expected errors would be
    for each extractor type and quality level.
    """
    db = SessionLocal()
    bodym_gts = db.query(GroundTruth).filter(
        GroundTruth.measured_by == "bodym_dataset"
    ).limit(n_samples).all()

    logger.info("Generating synthetic training data from %d BodyM samples...", len(bodym_gts))

    samples = []
    extractor_types = list(EXTRACTOR_QUALITY.keys())
    rng = np.random.RandomState(42)

    for gt in bodym_gts:
        # For each ground truth sample, simulate multiple quality conditions
        for _ in range(1):  # 1 synthetic sample per GT sample
            # Simulate visibility scores (higher = better quality)
            quality_level = rng.choice(["high", "medium", "low"], p=[0.3, 0.5, 0.2])

            if quality_level == "high":
                vis_mean, vis_std = 0.85, 0.08
                frame_ratio = rng.uniform(0.5, 0.8)
                angle = rng.uniform(0, 15)
                extractor = rng.choice(["smpl_mesh_slicing", "midas_actual", "trained_neural_network"])
                error_scale = rng.uniform(0.5, 2.0)  # cm
            elif quality_level == "medium":
                vis_mean, vis_std = 0.65, 0.15
                frame_ratio = rng.uniform(0.3, 0.6)
                angle = rng.uniform(10, 40)
                extractor = rng.choice(["ml_anthropometric", "fallback_mesh", "trained_neural_network"])
                error_scale = rng.uniform(1.5, 4.0)
            else:
                vis_mean, vis_std = 0.45, 0.2
                frame_ratio = rng.uniform(0.15, 0.4)
                angle = rng.uniform(30, 70)
                extractor = rng.choice(["simple_fallback", "fixed_anthropometric"])
                error_scale = rng.uniform(3.0, 7.0)

            # Generate 33 visibility scores
            vis_scores = {}
            for j in range(33):
                vis_scores[f"landmark_{j}"] = float(np.clip(rng.normal(vis_mean, vis_std), 0, 1))

            sample = {
                "source": "bodym_synthetic",
                "visibility_scores": vis_scores,
                "body_to_frame_ratio": float(frame_ratio),
                "pose_angle_degrees": float(angle),
                "extractor_method": extractor,
            }

            # Generate predicted measurements by adding realistic errors to actuals
            for name in MEASUREMENT_NAMES:
                gt_field = GT_FIELD_MAP[name]
                actual = getattr(gt, gt_field, None)
                if actual and actual > 0:
                    # Error is proportional to quality level
                    # Circumferences have higher error than linear measurements
                    mult = 1.3 if "circumference" in name else 1.0
                    error = rng.normal(0, error_scale * mult)
                    predicted = actual + error
                    sample[f"predicted_{name}"] = float(predicted)
                    sample[f"actual_{name}"] = float(actual)
                else:
                    sample[f"predicted_{name}"] = 0
                    sample[f"actual_{name}"] = 0

            samples.append(sample)

    db.close()
    logger.info("Generated %d synthetic training samples", len(samples))
    return samples


def save_validation_results(samples: list):
    """Save validation results to JSON for the training script."""
    results = {
        "total_samples": len(samples),
        "real_image_samples": sum(1 for s in samples if s.get("source") != "bodym_synthetic"),
        "synthetic_samples": sum(1 for s in samples if s.get("source") == "bodym_synthetic"),
        "samples": samples,
    }

    with open(VALIDATION_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved %d validation results to %s", len(samples), VALIDATION_RESULTS_PATH)


def train_confidence_model(samples: list, epochs: int = 150, lr: float = 0.001, batch_size: int = 64):
    """Train the confidence prediction model."""

    extractor_types = list(EXTRACTOR_QUALITY.keys())
    features_list = []
    labels_list = []

    for sample in samples:
        # Build feature vector (same as ConfidencePredictor._build_features)
        vis = np.full(33, 0.5)
        vis_scores = sample.get("visibility_scores", {})
        for i, v in enumerate(vis_scores.values()):
            if i < 33:
                vis[i] = v

        frame_ratio = sample.get("body_to_frame_ratio", 0.7)
        angle = sample.get("pose_angle_degrees", 0.0) / 90.0
        scalars = np.array([frame_ratio, angle])

        method = sample.get("extractor_method", "simple_fallback")
        one_hot = np.zeros(len(extractor_types))
        if method in extractor_types:
            one_hot[extractor_types.index(method)] = 1.0
        else:
            one_hot[-1] = 1.0

        features = np.concatenate([vis, scalars, one_hot])
        features_list.append(features)

        # Labels: absolute error per measurement (cm)
        errors = []
        for name in MEASUREMENT_NAMES:
            predicted = sample.get(f"predicted_{name}", 0)
            actual = sample.get(f"actual_{name}", 0)
            if predicted and actual:
                errors.append(abs(predicted - actual))
            else:
                errors.append(3.0)
        labels_list.append(errors)

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.float32)

    logger.info("Training data: %d samples, %d features", X.shape[0], X.shape[1])
    logger.info("Error stats: mean=%.2f cm, median=%.2f cm, max=%.2f cm",
                y.mean(), np.median(y), y.max())

    # Convert errors to confidence: conf = 1 - clip(error / 10, 0, 1)
    y_conf = np.clip(1.0 - y / 10.0, 0.1, 0.98).astype(np.float32)
    y_error = y.astype(np.float32)

    # Split 80/20
    n = len(X)
    split = int(n * 0.8)
    indices = np.random.permutation(n)
    train_idx, val_idx = indices[:split], indices[split:]

    train_ds = TensorDataset(
        torch.tensor(X[train_idx]),
        torch.tensor(y_conf[train_idx]),
        torch.tensor(y_error[train_idx]),
    )
    val_ds = TensorDataset(
        torch.tensor(X[val_idx]),
        torch.tensor(y_conf[val_idx]),
        torch.tensor(y_error[val_idx]),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = ConfidenceNet(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    conf_loss_fn = nn.MSELoss()
    error_loss_fn = nn.L1Loss()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x_batch, y_conf_batch, y_err_batch in train_loader:
            optimizer.zero_grad()
            pred_conf, pred_err = model(x_batch)
            loss = conf_loss_fn(pred_conf, y_conf_batch) + 0.5 * error_loss_fn(pred_err, y_err_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_conf_batch, y_err_batch in val_loader:
                pred_conf, pred_err = model(x_batch)
                loss = conf_loss_fn(pred_conf, y_conf_batch) + 0.5 * error_loss_fn(pred_err, y_err_batch)
                val_loss += loss.item()

        avg_train = train_loss / max(len(train_loader), 1)
        avg_val = val_loss / max(len(val_loader), 1)
        scheduler.step(avg_val)

        if (epoch + 1) % 20 == 0:
            logger.info("Epoch %d/%d  train=%.4f  val=%.4f  lr=%.6f",
                        epoch + 1, epochs, avg_train, avg_val,
                        optimizer.param_groups[0]["lr"])

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > 30:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    logger.info("Training complete! Best val_loss=%.4f", best_val_loss)
    logger.info("Model saved to %s", OUTPUT_MODEL_PATH)

    # Quick test
    model.load_state_dict(torch.load(OUTPUT_MODEL_PATH, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        test_x = torch.tensor(X[:5])
        pred_conf, pred_err = model(test_x)
        logger.info("Sample predictions:")
        for i in range(min(3, len(pred_conf))):
            conf_vals = {name: f"{pred_conf[i][j]:.2f}" for j, name in enumerate(MEASUREMENT_NAMES)}
            err_vals = {name: f"{pred_err[i][j]:.1f}cm" for j, name in enumerate(MEASUREMENT_NAMES)}
            logger.info("  Sample %d confidence: %s", i, conf_vals)
            logger.info("  Sample %d exp_error: %s", i, err_vals)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-images", action="store_true",
                        help="Skip real image processing (use synthetic data only)")
    args = parser.parse_args()

    real_samples = []
    if not args.skip_images:
        logger.info("=" * 60)
        logger.info("STEP 1: Process real images through ML pipeline")
        logger.info("=" * 60)
        real_samples = process_real_images()
    else:
        logger.info("Skipping real image processing (--skip-images)")

    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2: Generate synthetic training data from BodyM")
    logger.info("=" * 60)
    synthetic_samples = generate_synthetic_training_data(n_samples=2000)

    all_samples = real_samples + synthetic_samples
    logger.info("")
    logger.info("Total training samples: %d (real: %d, synthetic: %d)",
                len(all_samples), len(real_samples), len(synthetic_samples))

    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3: Save validation results")
    logger.info("=" * 60)
    save_validation_results(all_samples)

    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4: Train confidence prediction model")
    logger.info("=" * 60)
    train_confidence_model(all_samples)

    logger.info("")
    logger.info("=" * 60)
    logger.info("ALL DONE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
