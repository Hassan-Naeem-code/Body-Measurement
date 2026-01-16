"""
Neural Network-Based Depth Ratio Predictor
Replaces rule-based heuristics with learned model

This model is trained on actual ground truth data (real tape measurements)
to learn the optimal depth/width ratios for different body types.

Target: 95-98% accuracy for circumference estimation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from app.ml.pose_detector import PoseLandmarks
from app.ml.depth_ratio_predictor import BodyFeatures, DepthRatios

logger = logging.getLogger(__name__)


class RatioFeatureExtractor:
    """
    Extract numerical features from pose landmarks for neural network input

    Features designed to capture:
    - Body proportions
    - Body shape indicators
    - Pose quality metrics
    """

    # MediaPipe landmark indices
    LANDMARKS = {
        "NOSE": 0, "LEFT_EYE": 2, "RIGHT_EYE": 5,
        "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
        "LEFT_ELBOW": 13, "RIGHT_ELBOW": 14,
        "LEFT_WRIST": 15, "RIGHT_WRIST": 16,
        "LEFT_HIP": 23, "RIGHT_HIP": 24,
        "LEFT_KNEE": 25, "RIGHT_KNEE": 26,
        "LEFT_ANKLE": 27, "RIGHT_ANKLE": 28,
    }

    # Number of output features
    NUM_FEATURES = 20

    def extract(self, pose_landmarks: PoseLandmarks) -> np.ndarray:
        """
        Extract feature vector from pose landmarks

        Args:
            pose_landmarks: MediaPipe pose landmarks

        Returns:
            Feature vector of shape (NUM_FEATURES,)
        """
        landmarks = pose_landmarks.landmarks
        img_w = pose_landmarks.image_width
        img_h = pose_landmarks.image_height

        def get_lm(name):
            return landmarks[self.LANDMARKS[name]]

        # Get key landmarks
        nose = get_lm("NOSE")
        left_shoulder = get_lm("LEFT_SHOULDER")
        right_shoulder = get_lm("RIGHT_SHOULDER")
        left_hip = get_lm("LEFT_HIP")
        right_hip = get_lm("RIGHT_HIP")
        left_ankle = get_lm("LEFT_ANKLE")
        right_ankle = get_lm("RIGHT_ANKLE")
        left_knee = get_lm("LEFT_KNEE")
        right_knee = get_lm("RIGHT_KNEE")
        left_elbow = get_lm("LEFT_ELBOW")
        right_elbow = get_lm("RIGHT_ELBOW")

        # Calculate measurements (normalized to image size)
        shoulder_width = abs(left_shoulder['x'] - right_shoulder['x']) / img_w
        hip_width = abs(left_hip['x'] - right_hip['x']) / img_w

        shoulder_y = (left_shoulder['y'] + right_shoulder['y']) / 2 / img_h
        hip_y = (left_hip['y'] + right_hip['y']) / 2 / img_h
        ankle_y = (left_ankle['y'] + right_ankle['y']) / 2 / img_h
        knee_y = (left_knee['y'] + right_knee['y']) / 2 / img_h

        torso_length = abs(hip_y - shoulder_y)
        leg_length = abs(ankle_y - hip_y)
        upper_leg = abs(knee_y - hip_y)
        lower_leg = abs(ankle_y - knee_y)
        body_height = abs(ankle_y - nose['y'] / img_h)

        # Feature vector
        features = [
            # 1-5: Basic proportions
            shoulder_width,
            hip_width,
            shoulder_width / max(hip_width, 0.01),  # Shoulder-to-hip ratio
            torso_length,
            leg_length,

            # 6-10: Body shape indicators
            torso_length / max(leg_length, 0.01),  # Torso-to-leg ratio
            upper_leg / max(lower_leg, 0.01),  # Upper-to-lower leg ratio
            (shoulder_width + hip_width) / 2,  # Average width (body mass proxy)
            shoulder_width * 0.88 / max(hip_width * 0.8, 0.01),  # Chest-to-waist ratio estimate
            body_height,

            # 11-15: Symmetry and pose quality
            abs(left_shoulder['x'] - (img_w - right_shoulder['x'])) / img_w,  # Shoulder symmetry
            abs(left_hip['x'] - (img_w - right_hip['x'])) / img_w,  # Hip symmetry
            (left_shoulder.get('visibility', 0.5) + right_shoulder.get('visibility', 0.5)) / 2,
            (left_hip.get('visibility', 0.5) + right_hip.get('visibility', 0.5)) / 2,
            (left_ankle.get('visibility', 0.5) + right_ankle.get('visibility', 0.5)) / 2,

            # 16-20: Derived features
            shoulder_width / max(body_height, 0.01),  # Width-to-height ratio
            hip_width / max(body_height, 0.01),
            torso_length / max(body_height, 0.01),
            leg_length / max(body_height, 0.01),
            (shoulder_width - hip_width),  # Width difference (body shape)
        ]

        return np.array(features, dtype=np.float32)


class DepthRatioNet(nn.Module):
    """
    Neural network for predicting depth/width ratios

    Architecture:
    - Input: 20 body features
    - Hidden layers with batch norm and dropout
    - Output: 3 ratios (chest, waist, hip) + confidence

    The network learns to map body proportions to optimal depth ratios
    based on ground truth data.
    """

    def __init__(self, input_dim: int = 20, hidden_dims: List[int] = [64, 128, 64]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.2 if i < len(hidden_dims) - 1 else 0.1),
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Output heads
        self.ratio_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # chest, waist, hip ratios
            nn.Sigmoid(),  # Output in (0, 1) range
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(prev_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # Scale and shift for ratio output
        # Ratios should be in range [0.4, 0.85]
        self.register_buffer('ratio_min', torch.tensor([0.40, 0.40, 0.40]))
        self.register_buffer('ratio_range', torch.tensor([0.45, 0.45, 0.40]))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input features (batch_size, input_dim)

        Returns:
            ratios: Predicted depth ratios (batch_size, 3)
            confidence: Prediction confidence (batch_size, 1)
        """
        encoded = self.encoder(x)

        # Predict ratios (scaled to valid range)
        raw_ratios = self.ratio_head(encoded)
        ratios = self.ratio_min + raw_ratios * self.ratio_range

        # Predict confidence
        confidence = self.confidence_head(encoded)

        return ratios, confidence


class TrainedRatioPredictor:
    """
    Trained neural network predictor for depth ratios

    Uses the trained DepthRatioNet model to predict personalized
    depth/width ratios from pose landmarks.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize predictor

        Args:
            model_path: Path to trained model checkpoint
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = RatioFeatureExtractor()
        self.model = None
        self.is_model_loaded = False

        # Default model path
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__),
                'training', 'checkpoints', 'ratio_predictor.pt'
            )

        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load trained model"""
        try:
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)

                self.model = DepthRatioNet(
                    input_dim=checkpoint.get('input_dim', 20),
                    hidden_dims=checkpoint.get('hidden_dims', [64, 128, 64])
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()

                self.is_model_loaded = True
                logger.info(f"Loaded trained ratio predictor from {model_path}")

                # Log model metrics if available
                if 'metrics' in checkpoint:
                    m = checkpoint['metrics']
                    logger.info(
                        f"Model metrics - MAE: {m.get('mae', 'N/A')}, "
                        f"Accuracy: {m.get('accuracy', 'N/A')}"
                    )
            else:
                logger.warning(f"Model not found at {model_path}. Will use fallback.")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.is_model_loaded = False

    def predict_from_pose(self, pose_landmarks: PoseLandmarks) -> DepthRatios:
        """
        Predict depth ratios from pose landmarks

        Args:
            pose_landmarks: MediaPipe pose landmarks

        Returns:
            DepthRatios with predictions
        """
        if not self.is_model_loaded or self.model is None:
            return self._fallback_prediction(pose_landmarks)

        try:
            # Extract features
            features = self.feature_extractor.extract(pose_landmarks)

            # Convert to tensor
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Predict
            with torch.no_grad():
                ratios, confidence = self.model(x)

            # Convert to DepthRatios
            ratios = ratios.cpu().numpy()[0]
            confidence = confidence.cpu().numpy()[0, 0]

            return DepthRatios(
                chest_ratio=float(ratios[0]),
                waist_ratio=float(ratios[1]),
                hip_ratio=float(ratios[2]),
                shoulder_ratio=float(ratios[0] * 0.95),
                confidence=float(confidence),
                method='trained_neural_network'
            )

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._fallback_prediction(pose_landmarks)

    def _fallback_prediction(self, pose_landmarks: PoseLandmarks) -> DepthRatios:
        """Fallback to rule-based prediction"""
        from app.ml.depth_ratio_predictor import MLDepthRatioPredictor

        fallback = MLDepthRatioPredictor()
        result = fallback.predict_from_pose(pose_landmarks)
        result.method = 'rule_based_fallback'
        return result


class RatioTrainingDataset(Dataset):
    """
    Dataset for training the ratio predictor

    Loads ground truth data and pairs it with pose features.
    """

    def __init__(
        self,
        ground_truth_path: str,
        images_dir: str,
        transform=None
    ):
        """
        Initialize dataset

        Args:
            ground_truth_path: Path to ground_truth.json
            images_dir: Directory containing images
            transform: Optional transform for data augmentation
        """
        self.images_dir = images_dir
        self.transform = transform
        self.samples = []

        self._load_data(ground_truth_path)

    def _load_data(self, ground_truth_path: str):
        """Load and process ground truth data"""
        import cv2
        from app.ml.pose_detector import PoseDetector

        with open(ground_truth_path, 'r') as f:
            data = json.load(f)

        detector = PoseDetector()
        feature_extractor = RatioFeatureExtractor()

        for entry in data.get('measurements', []):
            # Skip incomplete entries
            if not all([
                entry.get('actual_chest_circumference'),
                entry.get('actual_waist_circumference'),
                entry.get('actual_hip_circumference'),
            ]):
                continue

            # Load image
            image_path = os.path.join(self.images_dir, entry['image_path'])
            if not os.path.exists(image_path):
                continue

            image = cv2.imread(image_path)
            if image is None:
                continue

            # Detect pose
            pose = detector.detect_from_array(image)
            if pose is None:
                continue

            # Extract features
            features = feature_extractor.extract(pose)

            # Calculate target ratios from actual measurements
            # Actual circumference = π * (a + b) * correction
            # where a = width/2, b = depth/2
            # So ratio = depth/width = circumference / (π * width) - 1 roughly

            # For now, use estimated ratios based on proportions
            # In a full implementation, we'd need front and side images
            # to calculate actual depth

            # Estimate ratios from body proportions and actual measurements
            # This is a simplification - ideally we'd have 3D ground truth
            chest_ratio = self._estimate_ratio_from_circumference(
                entry['actual_chest_circumference'],
                entry.get('actual_shoulder_width', 45) * 0.88  # Chest width estimate
            )
            waist_ratio = self._estimate_ratio_from_circumference(
                entry['actual_waist_circumference'],
                entry.get('actual_hip_circumference', 95) * 0.33  # Waist width estimate
            )
            hip_ratio = self._estimate_ratio_from_circumference(
                entry['actual_hip_circumference'],
                entry.get('actual_hip_circumference', 95) * 0.35  # Hip width estimate
            )

            target_ratios = np.array([
                np.clip(chest_ratio, 0.45, 0.85),
                np.clip(waist_ratio, 0.40, 0.80),
                np.clip(hip_ratio, 0.42, 0.78),
            ], dtype=np.float32)

            self.samples.append({
                'features': features,
                'ratios': target_ratios,
                'metadata': {
                    'id': entry['id'],
                    'gender': entry.get('gender'),
                    'body_type': entry.get('body_type'),
                }
            })

        logger.info(f"Loaded {len(self.samples)} training samples")

    def _estimate_ratio_from_circumference(
        self,
        circumference: float,
        estimated_width: float
    ) -> float:
        """
        Estimate depth/width ratio from circumference and width

        Uses inverse of ellipse circumference formula:
        C ≈ π * (a + b) * correction where a = width/2, b = depth/2
        """
        if estimated_width <= 0:
            return 0.60

        # Simplified: C ≈ 2 * π * sqrt((a² + b²) / 2)
        # Or C ≈ π * (a + b) for rough estimate

        # From C = π * (a + b):
        # a + b = C / π
        # a = width / 2
        # b = (C / π) - a
        # ratio = b / a = (2 * C / π / width) - 1

        a = estimated_width / 2
        sum_ab = circumference / np.pi
        b = sum_ab - a

        if a <= 0:
            return 0.60

        ratio = b / a

        return ratio

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'features': torch.tensor(sample['features'], dtype=torch.float32),
            'ratios': torch.tensor(sample['ratios'], dtype=torch.float32),
        }


def train_ratio_predictor(
    ground_truth_path: str,
    images_dir: str,
    output_path: str,
    epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 0.001,
) -> Dict:
    """
    Train the ratio predictor model

    Args:
        ground_truth_path: Path to ground_truth.json
        images_dir: Directory containing images
        output_path: Path to save trained model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate

    Returns:
        Training metrics
    """
    # Create dataset
    dataset = RatioTrainingDataset(ground_truth_path, images_dir)

    if len(dataset) < 10:
        logger.warning("Not enough training samples. Need at least 10.")
        return {'error': 'Insufficient training data'}

    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DepthRatioNet(input_dim=20)
    model.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            features = batch['features'].to(device)
            targets = batch['ratios'].to(device)

            optimizer.zero_grad()
            ratios, confidence = model(features)
            loss = criterion(ratios, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                targets = batch['ratios'].to(device)

                ratios, confidence = model(features)
                loss = criterion(ratios, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    # Save model
    if best_model_state is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        torch.save({
            'model_state_dict': best_model_state,
            'input_dim': 20,
            'hidden_dims': [64, 128, 64],
            'metrics': {
                'final_train_loss': train_losses[-1],
                'final_val_loss': val_losses[-1],
                'best_val_loss': best_val_loss,
            },
            'train_samples': train_size,
            'val_samples': val_size,
        }, output_path)

        logger.info(f"Model saved to {output_path}")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'train_samples': train_size,
        'val_samples': val_size,
    }
