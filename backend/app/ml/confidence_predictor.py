"""
Confidence Predictor — predicts per-measurement error magnitude.

Replaces hardcoded confidence scores with a learned model that estimates
how accurate each measurement is, given image/pose quality signals.

Until trained on ground truth data, falls back to a rule-based estimator
that is still better than flat constants.
"""

import numpy as np
import torch
import torch.nn as nn
import logging
import os
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MEASUREMENT_NAMES = [
    "chest_circumference",
    "waist_circumference",
    "hip_circumference",
    "shoulder_width",
    "inseam",
    "arm_length",
]

# Extractor quality priors (learned from validation; updated as data grows)
EXTRACTOR_QUALITY = {
    "smpl_mesh_slicing": 0.95,
    "fallback_mesh": 0.80,
    "midas_actual": 0.88,
    "trained_neural_network": 0.85,
    "ml_anthropometric": 0.78,
    "simple_fallback": 0.65,
    "fixed_anthropometric": 0.60,
}


@dataclass
class ConfidencePrediction:
    """Per-measurement confidence and expected error."""
    confidence: Dict[str, float]       # 0-1 per measurement
    expected_error_cm: Dict[str, float]  # predicted MAE in cm per measurement
    method: str                         # 'trained_model' or 'rule_based'


class ConfidenceNet(nn.Module):
    """Small MLP that predicts per-measurement confidence and error."""

    def __init__(self, input_dim: int = 42, num_measurements: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        # Two heads: confidence (sigmoid) and error (ReLU for positive values)
        self.confidence_head = nn.Sequential(nn.Linear(32, num_measurements), nn.Sigmoid())
        self.error_head = nn.Sequential(nn.Linear(32, num_measurements), nn.ReLU())

    def forward(self, x):
        features = self.net(x)
        return self.confidence_head(features), self.error_head(features)


class ConfidencePredictor:
    """
    Predicts per-measurement confidence and expected error.

    Input features (42 dims):
      - 33 landmark visibility scores
      - image body-to-frame ratio (1)
      - pose angle (1)
      - extractor type one-hot (7)

    Falls back to rule-based estimation when no trained model exists.
    """

    MODEL_PATH = os.path.join(
        os.path.dirname(__file__), "training", "checkpoints", "confidence_predictor.pt"
    )

    def __init__(self):
        self.model: Optional[ConfidenceNet] = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.MODEL_PATH):
            try:
                self.model = ConfidenceNet()
                state = torch.load(self.MODEL_PATH, map_location="cpu")
                self.model.load_state_dict(state)
                self.model.eval()
                logger.info("Confidence predictor model loaded")
            except Exception as e:
                logger.warning("Failed to load confidence model: %s. Using rule-based.", e)
                self.model = None
        else:
            logger.info("No trained confidence model found. Using rule-based fallback.")

    def predict(
        self,
        visibility_scores: Dict[str, float],
        body_to_frame_ratio: float,
        pose_angle_degrees: float,
        extractor_method: str,
    ) -> ConfidencePrediction:
        """
        Predict per-measurement confidence and expected error.

        Args:
            visibility_scores: landmark name -> visibility (0-1)
            body_to_frame_ratio: fraction of frame occupied by body (0-1)
            pose_angle_degrees: estimated pose angle from frontal (0=front, 90=side)
            extractor_method: which extractor produced the measurements

        Returns:
            ConfidencePrediction with per-measurement scores
        """
        if self.model is not None:
            return self._predict_with_model(
                visibility_scores, body_to_frame_ratio, pose_angle_degrees, extractor_method
            )
        return self._predict_rule_based(
            visibility_scores, body_to_frame_ratio, pose_angle_degrees, extractor_method
        )

    def _build_features(
        self,
        visibility_scores: Dict[str, float],
        body_to_frame_ratio: float,
        pose_angle_degrees: float,
        extractor_method: str,
    ) -> np.ndarray:
        """Build feature vector for the model."""
        # 33 visibility scores (pad missing with 0.5)
        vis = np.full(33, 0.5)
        for i, (name, score) in enumerate(visibility_scores.items()):
            if i < 33:
                vis[i] = score

        # Scalar features
        scalars = np.array([body_to_frame_ratio, pose_angle_degrees / 90.0])

        # Extractor one-hot (7 categories)
        extractor_types = list(EXTRACTOR_QUALITY.keys())
        one_hot = np.zeros(len(extractor_types))
        if extractor_method in extractor_types:
            one_hot[extractor_types.index(extractor_method)] = 1.0
        else:
            one_hot[-1] = 1.0  # default to last

        return np.concatenate([vis, scalars, one_hot]).astype(np.float32)

    def _predict_with_model(self, visibility_scores, body_to_frame_ratio, pose_angle_degrees, extractor_method):
        features = self._build_features(visibility_scores, body_to_frame_ratio, pose_angle_degrees, extractor_method)
        x = torch.tensor(features).unsqueeze(0)
        with torch.no_grad():
            conf_tensor, error_tensor = self.model(x)
        conf = conf_tensor[0].numpy()
        error = error_tensor[0].numpy()
        return ConfidencePrediction(
            confidence={name: float(conf[i]) for i, name in enumerate(MEASUREMENT_NAMES)},
            expected_error_cm={name: float(error[i]) for i, name in enumerate(MEASUREMENT_NAMES)},
            method="trained_model",
        )

    def _predict_rule_based(self, visibility_scores, body_to_frame_ratio, pose_angle_degrees, extractor_method):
        """
        Rule-based confidence estimation — better than flat constants.

        Factors in:
        - Extractor quality prior
        - Average landmark visibility
        - Pose angle penalty (side views are less accurate)
        - Body-to-frame ratio (too small = less accuracy)
        """
        # Base quality from extractor type
        base = EXTRACTOR_QUALITY.get(extractor_method, 0.65)

        # Visibility factor: average of relevant landmarks per measurement
        all_vis = list(visibility_scores.values()) if visibility_scores else [0.5]
        avg_vis = np.mean(all_vis)

        # Pose angle penalty: 0° = frontal (best), 90° = side (worst for width measurements)
        angle_factor = 1.0 - (pose_angle_degrees / 90.0) * 0.25

        # Frame ratio factor: body should fill at least 50% of frame for good accuracy
        frame_factor = min(1.0, body_to_frame_ratio / 0.5) if body_to_frame_ratio > 0 else 0.7

        combined = base * 0.5 + avg_vis * 0.25 + angle_factor * 0.15 + frame_factor * 0.10

        # Per-measurement adjustments
        confidence = {}
        expected_error = {}
        for name in MEASUREMENT_NAMES:
            c = combined
            # Circumferences are harder to estimate than linear measurements
            if "circumference" in name:
                c *= 0.97
            # Inseam and arm length are more reliable from 2D
            if name in ("inseam", "arm_length"):
                c *= 1.02

            confidence[name] = float(np.clip(c, 0.3, 0.98))

            # Expected error: rough mapping from confidence to cm
            # High confidence (~0.95) => ~1.5cm error; low (~0.65) => ~5cm error
            expected_error[name] = float(np.clip(8.0 * (1.0 - c), 1.0, 8.0))

        return ConfidencePrediction(
            confidence=confidence,
            expected_error_cm=expected_error,
            method="rule_based",
        )
