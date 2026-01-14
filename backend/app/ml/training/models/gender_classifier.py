"""
Gender Classification Model
Two approaches:
1. Feature-based: Uses pose landmarks + body proportions (fast, lightweight)
2. CNN-based: Uses body image crops (more accurate, requires more compute)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict
import pickle
import os
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# APPROACH 1: Feature-Based Gender Classifier (Lightweight)
# ============================================================================

class GenderFeatureExtractor:
    """
    Extract gender-predictive features from pose landmarks

    Features extracted:
    - Shoulder-to-hip ratio
    - Shoulder width normalized
    - Hip width normalized
    - Waist-to-hip ratio
    - Chest-to-waist ratio
    - Torso-to-leg ratio
    - Body aspect ratio
    - Upper body mass indicator
    - Lower body mass indicator
    """

    FEATURE_NAMES = [
        'shoulder_hip_ratio',
        'shoulder_width_norm',
        'hip_width_norm',
        'waist_hip_ratio',
        'chest_waist_ratio',
        'torso_leg_ratio',
        'body_aspect_ratio',
        'upper_body_mass',
        'lower_body_mass',
        'shoulder_slope',
        'hip_slope',
    ]

    def extract_features(self, pose_landmarks) -> np.ndarray:
        """
        Extract features from pose landmarks

        Args:
            pose_landmarks: PoseLandmarks object

        Returns:
            Feature vector (numpy array)
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

        # Calculate measurements
        shoulder_width = abs(left_shoulder['x'] - right_shoulder['x'])
        hip_width = abs(left_hip['x'] - right_hip['x'])

        avg_shoulder_y = (left_shoulder['y'] + right_shoulder['y']) / 2
        avg_hip_y = (left_hip['y'] + right_hip['y']) / 2
        avg_ankle_y = (left_ankle['y'] + right_ankle['y']) / 2

        torso_length = abs(avg_hip_y - avg_shoulder_y)
        leg_length = abs(avg_ankle_y - avg_hip_y)
        body_height = abs(avg_ankle_y - nose['y'])

        # Estimate waist and chest widths
        waist_width = hip_width * 0.85
        chest_width = shoulder_width * 0.90

        # Calculate features
        features = []

        # 1. Shoulder-to-hip ratio (key gender indicator)
        shoulder_hip_ratio = shoulder_width / max(hip_width, 1e-6)
        features.append(shoulder_hip_ratio)

        # 2. Shoulder width normalized to body height
        shoulder_width_norm = shoulder_width / max(body_height, 1e-6)
        features.append(shoulder_width_norm)

        # 3. Hip width normalized
        hip_width_norm = hip_width / max(body_height, 1e-6)
        features.append(hip_width_norm)

        # 4. Waist-to-hip ratio
        waist_hip_ratio = waist_width / max(hip_width, 1e-6)
        features.append(waist_hip_ratio)

        # 5. Chest-to-waist ratio
        chest_waist_ratio = chest_width / max(waist_width, 1e-6)
        features.append(chest_waist_ratio)

        # 6. Torso-to-leg ratio
        torso_leg_ratio = torso_length / max(leg_length, 1e-6)
        features.append(torso_leg_ratio)

        # 7. Body aspect ratio (height/width)
        body_aspect_ratio = body_height / max(shoulder_width, 1e-6)
        features.append(body_aspect_ratio)

        # 8. Upper body mass indicator
        upper_body_mass = shoulder_width_norm * chest_waist_ratio
        features.append(upper_body_mass)

        # 9. Lower body mass indicator
        lower_body_mass = hip_width_norm / max(shoulder_width_norm, 1e-6)
        features.append(lower_body_mass)

        # 10. Shoulder slope (difference in shoulder heights)
        shoulder_slope = abs(left_shoulder['y'] - right_shoulder['y']) / max(shoulder_width, 1e-6)
        features.append(shoulder_slope)

        # 11. Hip slope
        hip_slope = abs(left_hip['y'] - right_hip['y']) / max(hip_width, 1e-6)
        features.append(hip_slope)

        return np.array(features, dtype=np.float32)

    def extract_features_from_measurements(self, measurements) -> np.ndarray:
        """
        Extract features from CircumferenceMeasurements object
        """
        features = []

        # 1. Shoulder-to-hip ratio
        shoulder_hip_ratio = measurements.shoulder_width / max(measurements.hip_width, 1e-6)
        features.append(shoulder_hip_ratio)

        # 2. Shoulder width normalized (assume ~170cm height)
        shoulder_width_norm = measurements.shoulder_width / max(measurements.estimated_height_cm, 1e-6)
        features.append(shoulder_width_norm)

        # 3. Hip width normalized
        hip_width_norm = measurements.hip_width / max(measurements.estimated_height_cm, 1e-6)
        features.append(hip_width_norm)

        # 4. Waist-to-hip ratio (using circumferences)
        waist_hip_ratio = measurements.waist_circumference / max(measurements.hip_circumference, 1e-6)
        features.append(waist_hip_ratio)

        # 5. Chest-to-waist ratio
        chest_waist_ratio = measurements.chest_circumference / max(measurements.waist_circumference, 1e-6)
        features.append(chest_waist_ratio)

        # 6. Torso-to-leg ratio (estimate from inseam)
        torso_estimate = measurements.estimated_height_cm * 0.30
        torso_leg_ratio = torso_estimate / max(measurements.inseam, 1e-6)
        features.append(torso_leg_ratio)

        # 7. Body aspect ratio
        body_aspect_ratio = measurements.estimated_height_cm / max(measurements.shoulder_width, 1e-6)
        features.append(body_aspect_ratio)

        # 8. Upper body mass
        upper_body_mass = shoulder_width_norm * chest_waist_ratio
        features.append(upper_body_mass)

        # 9. Lower body mass
        lower_body_mass = hip_width_norm / max(shoulder_width_norm, 1e-6)
        features.append(lower_body_mass)

        # 10, 11. Slopes (not available from measurements, use defaults)
        features.append(0.05)  # shoulder_slope
        features.append(0.05)  # hip_slope

        return np.array(features, dtype=np.float32)


class GenderDataset(Dataset):
    """Dataset for gender classification training"""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Args:
            features: (N, num_features) feature matrix
            labels: (N,) labels (0=female, 1=male)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class GenderClassifierMLP(nn.Module):
    """
    MLP-based gender classifier using body features
    """

    def __init__(self, input_dim: int = 11, hidden_dims: List[int] = [64, 32, 16]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 2))  # 2 classes: male, female

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def predict_proba(self, x):
        """Get probability scores"""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs


class GenderClassifierTrainer:
    """Trainer for the gender classification model"""

    def __init__(
        self,
        model: GenderClassifierMLP,
        learning_rate: float = 0.001,
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0

        for features, labels in dataloader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Evaluate model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in dataloader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0

        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 10
    ) -> Dict:
        """Full training loop with early stopping"""
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_accuracy = self.evaluate(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)

            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Accuracy: {val_accuracy:.2%}"
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return self.history

    def save_model(self, path: str):
        """Save model to disk"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model from disk"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', {})
        logger.info(f"Model loaded from {path}")


class TrainedGenderClassifier:
    """
    Production-ready gender classifier that uses the trained model
    Falls back to rule-based if model not available
    """

    def __init__(self, model_path: str = None):
        self.feature_extractor = GenderFeatureExtractor()
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            logger.warning("No trained model found. Using rule-based fallback.")

    def _load_model(self, path: str):
        """Load trained model"""
        try:
            self.model = GenderClassifierMLP()
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded trained gender model from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def predict(self, pose_landmarks=None, measurements=None) -> Tuple[str, float]:
        """
        Predict gender from pose landmarks or measurements

        Returns:
            Tuple of (gender: str, confidence: float)
        """
        # Extract features
        if pose_landmarks is not None:
            features = self.feature_extractor.extract_features(pose_landmarks)
        elif measurements is not None:
            features = self.feature_extractor.extract_features_from_measurements(measurements)
        else:
            raise ValueError("Must provide either pose_landmarks or measurements")

        # Use trained model if available
        if self.model is not None:
            return self._predict_with_model(features)
        else:
            return self._predict_rule_based(features)

    def _predict_with_model(self, features: np.ndarray) -> Tuple[str, float]:
        """Predict using trained model"""
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            probs = self.model.predict_proba(x)

            # Index 0 = female, 1 = male
            female_prob = probs[0, 0].item()
            male_prob = probs[0, 1].item()

            if male_prob > female_prob:
                return 'male', male_prob
            else:
                return 'female', female_prob

    def _predict_rule_based(self, features: np.ndarray) -> Tuple[str, float]:
        """Fallback rule-based prediction"""
        shoulder_hip_ratio = features[0]
        chest_waist_ratio = features[4]

        male_score = 0.0

        if shoulder_hip_ratio > 1.15:
            male_score += 2.0
        elif shoulder_hip_ratio > 1.05:
            male_score += 1.0
        elif shoulder_hip_ratio < 0.95:
            male_score -= 2.0
        elif shoulder_hip_ratio < 1.00:
            male_score -= 1.0

        if chest_waist_ratio > 1.20:
            male_score += 1.0

        if male_score > 0:
            confidence = min(0.95, 0.60 + male_score * 0.10)
            return 'male', confidence
        else:
            confidence = min(0.95, 0.60 + abs(male_score) * 0.10)
            return 'female', confidence
