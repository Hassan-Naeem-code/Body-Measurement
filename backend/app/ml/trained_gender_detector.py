"""
Production Gender Detector using Trained ML Model

This module provides a drop-in replacement for the rule-based gender detection,
using the trained neural network model when available.
"""

import os
import logging
from typing import Tuple, Optional

import torch
import numpy as np

logger = logging.getLogger(__name__)


class MLGenderDetector:
    """
    ML-powered gender detector that uses a trained neural network

    Falls back to rule-based detection if model is not available.
    """

    def __init__(self, model_path: str = None):
        """
        Initialize the detector

        Args:
            model_path: Path to trained model file. If None, searches default locations.
        """
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Search for model in default locations
        if model_path is None:
            possible_paths = [
                os.path.join(os.path.dirname(__file__), 'gender_model.pth'),
                os.path.join(os.path.dirname(__file__), 'training', 'models', 'saved', 'gender_classifier.pth'),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break

        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            logger.warning(
                "No trained gender model found. Will use rule-based detection. "
                "Run training script to create a model."
            )

    def _load_model(self, path: str):
        """Load the trained model"""
        try:
            # Import here to avoid circular imports
            from app.ml.training.models.gender_classifier import GenderClassifierMLP

            self.model = GenderClassifierMLP(input_dim=11, hidden_dims=[64, 32, 16])
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Loaded trained gender model from {path}")

        except Exception as e:
            logger.error(f"Failed to load gender model: {e}")
            self.model = None

    def extract_features(self, pose_landmarks=None, measurements=None) -> np.ndarray:
        """
        Extract features for gender prediction

        Args:
            pose_landmarks: PoseLandmarks object
            measurements: CircumferenceMeasurements object

        Returns:
            Feature vector as numpy array
        """
        if measurements is not None:
            return self._features_from_measurements(measurements)
        elif pose_landmarks is not None:
            return self._features_from_landmarks(pose_landmarks)
        else:
            raise ValueError("Must provide either pose_landmarks or measurements")

    def _features_from_landmarks(self, pose_landmarks) -> np.ndarray:
        """
        Extract features from pose landmarks

        Note: From landmarks alone, we can directly measure:
        - shoulder width (distance between shoulders)
        - hip width (distance between hips)
        - body proportions

        But we CANNOT measure:
        - waist/chest circumferences (need measurements object for that)

        For features that require circumferences, we use typical values.
        This is less accurate but avoids wildly wrong estimates.
        """
        landmarks = pose_landmarks.landmarks
        img_width = pose_landmarks.image_width
        img_height = pose_landmarks.image_height

        # Get key landmarks
        nose = landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]

        # Calculate normalized distances (landmarks are in 0-1 range)
        shoulder_width_norm = abs(left_shoulder['x'] - right_shoulder['x'])
        hip_width_norm = abs(left_hip['x'] - right_hip['x'])

        avg_shoulder_y = (left_shoulder['y'] + right_shoulder['y']) / 2
        avg_hip_y = (left_hip['y'] + right_hip['y']) / 2
        avg_ankle_y = (left_ankle['y'] + right_ankle['y']) / 2

        torso_length = abs(avg_hip_y - avg_shoulder_y)
        leg_length = abs(avg_ankle_y - avg_hip_y)
        body_height = abs(avg_ankle_y - nose['y'])

        # Calculate shoulder-hip ratio from landmarks
        # This should give values in the 2.0-3.5 range matching training data
        shoulder_hip_ratio = shoulder_width_norm / max(hip_width_norm, 1e-6)

        # Body aspect ratio
        body_aspect_ratio = body_height / max(shoulder_width_norm, 1e-6)

        # Torso-leg ratio from actual landmark positions
        torso_leg_ratio = torso_length / max(leg_length, 1e-6)

        # For circumference-based features, use typical values since
        # we can't measure them from landmarks alone.
        # These are averages from anthropometric data.
        waist_hip_ratio = 0.80  # Typical value
        chest_waist_ratio = 1.15  # Typical value

        # Upper/lower body mass indicators
        upper_body_mass = shoulder_width_norm * chest_waist_ratio
        lower_body_mass = hip_width_norm / max(shoulder_width_norm, 1e-6)

        # Shoulder and hip slopes (asymmetry indicators)
        shoulder_slope = abs(left_shoulder['y'] - right_shoulder['y']) / max(shoulder_width_norm, 1e-6)
        hip_slope = abs(left_hip['y'] - right_hip['y']) / max(hip_width_norm, 1e-6)

        features = [
            shoulder_hip_ratio,     # Primary gender indicator (males ~2.9-3.2, females ~2.1-2.6)
            shoulder_width_norm,    # Normalized 0-0.35
            hip_width_norm,         # Normalized 0-0.15
            waist_hip_ratio,        # Typical default
            chest_waist_ratio,      # Typical default
            torso_leg_ratio,        # From landmarks
            body_aspect_ratio,      # Height/shoulder ratio
            upper_body_mass,        # Combined indicator
            lower_body_mass,        # Hip prominence indicator
            shoulder_slope,         # Asymmetry
            hip_slope,              # Asymmetry
        ]

        return np.array(features, dtype=np.float32)

    def _features_from_measurements(self, measurements) -> np.ndarray:
        """
        Extract features from measurements object

        Features must match exactly what the model was trained on:
        - shoulder_hip_ratio: shoulder_width / (hip_circumference / π / 2)
        - shoulder_width_norm: normalized shoulder width (0-1, simulating landmark distance)
        - hip_width_norm: normalized hip width (0-1, simulating landmark distance)
        - etc.
        """
        # Calculate hip diameter from circumference (matches training data calculation)
        hip_diameter = measurements.hip_circumference / np.pi / 2

        # Normalize widths to approximate landmark distances (0-1 range)
        # In training, these are normalized x-coordinates, roughly shoulder/height
        shoulder_width_norm = measurements.shoulder_width / max(measurements.estimated_height_cm, 1e-6)
        hip_width_norm = hip_diameter / max(measurements.estimated_height_cm, 1e-6)

        # Shoulder-hip ratio as used in training: shoulder_width / hip_diameter
        shoulder_hip_ratio = measurements.shoulder_width / max(hip_diameter, 1e-6)

        # Other ratios
        waist_hip_ratio = measurements.waist_circumference / max(measurements.hip_circumference, 1e-6)
        chest_waist_ratio = measurements.chest_circumference / max(measurements.waist_circumference, 1e-6)
        body_aspect_ratio = measurements.estimated_height_cm / max(measurements.shoulder_width, 1e-6)

        # Upper and lower body mass indicators
        upper_body_mass = shoulder_width_norm * chest_waist_ratio
        lower_body_mass = hip_width_norm / max(shoulder_width_norm, 1e-6)

        features = [
            shoulder_hip_ratio,    # Males: ~2.9-3.2, Females: ~2.1-2.6
            shoulder_width_norm,   # Normalized 0-0.3
            hip_width_norm,        # Normalized 0-0.15
            waist_hip_ratio,       # Typically 0.7-0.9
            chest_waist_ratio,     # Typically 1.0-1.4
            0.65,                  # torso_leg_ratio (constant)
            body_aspect_ratio,     # Typically 3.3-4.1
            upper_body_mass,       # Combined indicator
            lower_body_mass,       # Hip prominence indicator
            0.05,                  # shoulder_slope (constant)
            0.05,                  # hip_slope (constant)
        ]

        return np.array(features, dtype=np.float32)

    def predict(
        self,
        pose_landmarks=None,
        measurements=None
    ) -> Tuple[str, float]:
        """
        Predict gender

        Args:
            pose_landmarks: PoseLandmarks object
            measurements: CircumferenceMeasurements object

        Returns:
            Tuple of (gender: str, confidence: float)
        """
        features = self.extract_features(pose_landmarks, measurements)

        if self.model is not None:
            return self._predict_with_model(features)
        else:
            return self._predict_rule_based(features)

    def _predict_with_model(self, features: np.ndarray) -> Tuple[str, float]:
        """Predict using trained neural network"""
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)

            female_prob = probs[0, 0].item()
            male_prob = probs[0, 1].item()

            if male_prob > female_prob:
                return 'male', male_prob
            else:
                return 'female', female_prob

    def _predict_rule_based(self, features: np.ndarray) -> Tuple[str, float]:
        """Fallback rule-based prediction"""
        shoulder_hip_ratio = features[0]  # Most important feature
        chest_waist_ratio = features[4]
        upper_body_mass = features[7]

        male_score = 0.0
        female_score = 0.0

        # Shoulder-hip ratio (primary indicator)
        if shoulder_hip_ratio >= 1.25:
            male_score += 3.0
        elif shoulder_hip_ratio >= 1.15:
            male_score += 2.0
        elif shoulder_hip_ratio >= 1.05:
            male_score += 0.5
        elif shoulder_hip_ratio <= 0.95:
            female_score += 2.5
        elif shoulder_hip_ratio <= 1.00:
            female_score += 1.5
        else:
            female_score += 0.5

        # Chest-waist ratio
        if chest_waist_ratio >= 1.35:
            male_score += 1.5
        elif chest_waist_ratio >= 1.20:
            male_score += 0.8

        # Upper body mass
        if upper_body_mass > 0.15:
            male_score += 0.5

        # Determine result
        total_score = male_score + female_score
        if total_score < 1.0:
            return 'female', 0.55

        if male_score > female_score:
            confidence = min(0.95, 0.55 + (male_score / total_score) * 0.40)
            return 'male', confidence
        else:
            confidence = min(0.95, 0.55 + (female_score / total_score) * 0.40)
            return 'female', confidence

    @property
    def is_model_loaded(self) -> bool:
        """Check if trained model is loaded"""
        return self.model is not None


# Singleton instance for easy import
_detector_instance: Optional[MLGenderDetector] = None


def get_gender_detector() -> MLGenderDetector:
    """Get or create the singleton gender detector instance"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = MLGenderDetector()
    return _detector_instance
