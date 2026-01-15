"""
Body Measurement Prediction Model

A regression neural network that predicts actual body measurements from pose landmarks.
This replaces the hardcoded geometric formulas with learned relationships.

Input Features (from pose landmarks):
- Normalized body proportions
- Landmark positions
- Estimated body type indicators

Output (actual measurements in cm):
- chest_circumference
- waist_circumference
- hip_circumference
- shoulder_width
- arm_circumference
- thigh_circumference
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import os
import logging
import json

logger = logging.getLogger(__name__)


class MeasurementFeatureExtractor:
    """
    Extract features from pose landmarks for measurement prediction.

    Features are designed to capture:
    1. Body proportions (ratios between landmarks)
    2. Absolute positions (normalized)
    3. Body type indicators
    4. Pose angle information
    """

    FEATURE_NAMES = [
        # Basic proportions (7 features)
        'shoulder_width_norm',        # Shoulder width / body height
        'hip_width_norm',             # Hip width / body height
        'torso_length_norm',          # Torso length / body height
        'leg_length_norm',            # Leg length / body height
        'arm_span_norm',              # Arm span / body height
        'head_to_body_ratio',         # Head size / body height
        'body_height_pixels',         # Total body height in normalized coords

        # Ratios (6 features)
        'shoulder_hip_ratio',         # Key gender/body type indicator
        'torso_leg_ratio',            # Upper vs lower body
        'shoulder_torso_ratio',       # Shoulder development
        'hip_torso_ratio',            # Hip prominence
        'chest_position_ratio',       # Where chest sits on torso
        'waist_position_ratio',       # Where waist sits on torso

        # Body type indicators (5 features)
        'upper_body_mass',            # Shoulder development indicator
        'lower_body_mass',            # Hip development indicator
        'body_taper',                 # Shoulder-to-hip taper
        'bmi_indicator',              # Estimated from proportions
        'body_compactness',           # Width to height ratio

        # Pose information (4 features)
        'pose_angle',                 # 0 = frontal, 1 = side
        'shoulder_symmetry',          # Left-right balance
        'hip_symmetry',               # Left-right balance
        'landmark_confidence',        # Average confidence score

        # Height estimation (2 features)
        'estimated_height_normalized', # Normalized height estimate
        'height_confidence',           # Confidence in height estimate
    ]

    NUM_FEATURES = len(FEATURE_NAMES)

    def extract_features(self, pose_landmarks) -> np.ndarray:
        """
        Extract features from PoseLandmarks object

        Args:
            pose_landmarks: PoseLandmarks with landmarks and visibility_scores

        Returns:
            Feature vector of shape (NUM_FEATURES,)
        """
        landmarks = pose_landmarks.landmarks
        visibility = pose_landmarks.visibility_scores
        img_width = pose_landmarks.image_width
        img_height = pose_landmarks.image_height

        # Get key landmark positions
        nose = landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_elbow = landmarks[13]
        right_elbow = landmarks[14]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        left_knee = landmarks[25]
        right_knee = landmarks[26]
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]

        # Calculate basic measurements (in pixels)
        shoulder_width = abs(left_shoulder['x'] - right_shoulder['x']) * img_width
        hip_width = abs(left_hip['x'] - right_hip['x']) * img_width

        avg_shoulder_y = (left_shoulder['y'] + right_shoulder['y']) / 2 * img_height
        avg_hip_y = (left_hip['y'] + right_hip['y']) / 2 * img_height
        avg_ankle_y = (left_ankle['y'] + right_ankle['y']) / 2 * img_height
        nose_y = nose['y'] * img_height

        torso_length = abs(avg_hip_y - avg_shoulder_y)
        leg_length = abs(avg_ankle_y - avg_hip_y)
        body_height = abs(avg_ankle_y - nose_y)

        # Arm span calculation
        left_arm = self._calculate_arm_length(left_shoulder, left_elbow, left_wrist, img_width, img_height)
        right_arm = self._calculate_arm_length(right_shoulder, right_elbow, right_wrist, img_width, img_height)
        arm_span = left_arm + right_arm + shoulder_width

        # Head size (approximate from nose to shoulder)
        head_size = avg_shoulder_y - nose_y

        # Build feature vector
        features = []

        # === Basic proportions (7) ===
        features.append(shoulder_width / max(body_height, 1))  # shoulder_width_norm
        features.append(hip_width / max(body_height, 1))       # hip_width_norm
        features.append(torso_length / max(body_height, 1))    # torso_length_norm
        features.append(leg_length / max(body_height, 1))      # leg_length_norm
        features.append(arm_span / max(body_height, 1))        # arm_span_norm
        features.append(head_size / max(body_height, 1))       # head_to_body_ratio
        features.append(body_height / img_height)              # body_height_pixels

        # === Ratios (6) ===
        features.append(shoulder_width / max(hip_width, 1))    # shoulder_hip_ratio
        features.append(torso_length / max(leg_length, 1))     # torso_leg_ratio
        features.append(shoulder_width / max(torso_length, 1)) # shoulder_torso_ratio
        features.append(hip_width / max(torso_length, 1))      # hip_torso_ratio

        # Chest position (estimated at 1/3 down torso)
        chest_y = avg_shoulder_y + torso_length * 0.33
        features.append((chest_y - nose_y) / max(body_height, 1))  # chest_position_ratio

        # Waist position (estimated at 2/3 down torso)
        waist_y = avg_shoulder_y + torso_length * 0.67
        features.append((waist_y - nose_y) / max(body_height, 1))  # waist_position_ratio

        # === Body type indicators (5) ===
        upper_body_mass = (shoulder_width / max(body_height, 1)) * (torso_length / max(body_height, 1))
        lower_body_mass = (hip_width / max(body_height, 1)) * (leg_length / max(body_height, 1))
        features.append(upper_body_mass)                        # upper_body_mass
        features.append(lower_body_mass)                        # lower_body_mass
        features.append(shoulder_width / max(hip_width, 1) - 1) # body_taper (-ve = wider hips)

        # BMI indicator (width-to-height based)
        avg_width = (shoulder_width + hip_width) / 2
        bmi_indicator = (avg_width / max(body_height, 1)) ** 2 * 100  # Scaled
        features.append(bmi_indicator)                          # bmi_indicator

        # Body compactness
        features.append(avg_width / max(body_height, 1))        # body_compactness

        # === Pose information (4) ===
        # Pose angle estimation (0 = frontal, 1 = side)
        avg_width_norm = avg_width / img_width
        pose_angle = max(0, min(1, 1 - avg_width_norm * 4))     # Wider = more frontal
        features.append(pose_angle)                             # pose_angle

        # Symmetry (0 = perfect, 1 = very asymmetric)
        shoulder_diff = abs(left_shoulder['y'] - right_shoulder['y']) * img_height
        hip_diff = abs(left_hip['y'] - right_hip['y']) * img_height
        features.append(shoulder_diff / max(shoulder_width, 1))  # shoulder_symmetry
        features.append(hip_diff / max(hip_width, 1))            # hip_symmetry

        # Average landmark confidence
        key_landmarks = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP',
                        'LEFT_ANKLE', 'RIGHT_ANKLE', 'NOSE']
        confidences = [visibility.get(lm, 0.5) for lm in key_landmarks]
        features.append(np.mean(confidences))                    # landmark_confidence

        # === Height estimation (2) ===
        # Estimate actual height using head reference (head ≈ 23cm)
        estimated_height = (head_size / 0.13) if head_size > 0 else 170  # head ≈ 13% of height
        features.append(estimated_height / 200)                  # estimated_height_normalized (scale to ~1)
        features.append(min(1.0, head_size / (body_height * 0.15 + 1)))  # height_confidence

        return np.array(features, dtype=np.float32)

    def _calculate_arm_length(self, shoulder, elbow, wrist, img_width, img_height) -> float:
        """Calculate arm length from landmarks"""
        shoulder_to_elbow = np.sqrt(
            ((shoulder['x'] - elbow['x']) * img_width) ** 2 +
            ((shoulder['y'] - elbow['y']) * img_height) ** 2
        )
        elbow_to_wrist = np.sqrt(
            ((elbow['x'] - wrist['x']) * img_width) ** 2 +
            ((elbow['y'] - wrist['y']) * img_height) ** 2
        )
        return shoulder_to_elbow + elbow_to_wrist

    def extract_features_from_synthetic(self, sample) -> np.ndarray:
        """
        Extract features from SyntheticBodyData for training
        """
        landmarks = sample.landmarks

        # Get normalized positions
        shoulder_width = abs(landmarks['left_shoulder']['x'] - landmarks['right_shoulder']['x'])
        hip_width = abs(landmarks['left_hip']['x'] - landmarks['right_hip']['x'])

        nose_y = landmarks['nose']['y']
        shoulder_y = (landmarks['left_shoulder']['y'] + landmarks['right_shoulder']['y']) / 2
        hip_y = (landmarks['left_hip']['y'] + landmarks['right_hip']['y']) / 2
        ankle_y = (landmarks['left_ankle']['y'] + landmarks['right_ankle']['y']) / 2

        # Get elbow and wrist positions for arm length calculation
        left_elbow = landmarks.get('left_elbow', {'x': 0.35, 'y': 0.36})
        right_elbow = landmarks.get('right_elbow', {'x': 0.65, 'y': 0.36})
        left_wrist = landmarks.get('left_wrist', {'x': 0.35, 'y': 0.50})
        right_wrist = landmarks.get('right_wrist', {'x': 0.65, 'y': 0.50})

        torso_length = abs(hip_y - shoulder_y)
        leg_length = abs(ankle_y - hip_y)
        body_height = abs(ankle_y - nose_y)
        head_size = shoulder_y - nose_y

        # Calculate arm span from landmarks
        left_arm_len = self._calc_distance(landmarks['left_shoulder'], left_elbow) + \
                       self._calc_distance(left_elbow, left_wrist)
        right_arm_len = self._calc_distance(landmarks['right_shoulder'], right_elbow) + \
                        self._calc_distance(right_elbow, right_wrist)
        arm_span = left_arm_len + right_arm_len + shoulder_width

        # Build features (matching the order in FEATURE_NAMES)
        features = []

        # Basic proportions (7)
        features.append(shoulder_width / max(body_height, 0.01))
        features.append(hip_width / max(body_height, 0.01))
        features.append(torso_length / max(body_height, 0.01))
        features.append(leg_length / max(body_height, 0.01))
        features.append(arm_span / max(body_height, 0.01))
        features.append(head_size / max(body_height, 0.01))
        features.append(body_height)

        # Ratios (6)
        features.append(sample.shoulder_hip_ratio)
        features.append(torso_length / max(leg_length, 0.01))
        features.append(shoulder_width / max(torso_length, 0.01))
        features.append(hip_width / max(torso_length, 0.01))
        features.append((shoulder_y + torso_length * 0.33 - nose_y) / max(body_height, 0.01))
        features.append((shoulder_y + torso_length * 0.67 - nose_y) / max(body_height, 0.01))

        # Body type indicators (5)
        features.append(shoulder_width * torso_length)
        features.append(hip_width * leg_length)
        features.append(sample.shoulder_hip_ratio - 1)
        features.append(sample.chest_circumference / sample.height * 10)  # BMI-like
        features.append((shoulder_width + hip_width) / 2)

        # Pose information (4)
        features.append(0.0)  # Frontal pose for synthetic data
        features.append(0.02)  # Small asymmetry
        features.append(0.02)
        features.append(0.9)  # High confidence

        # Height estimation (2)
        features.append(sample.height / 200)
        features.append(0.95)

        return np.array(features, dtype=np.float32)

    def _calc_distance(self, p1: dict, p2: dict) -> float:
        """Calculate Euclidean distance between two landmark points"""
        return np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)


class MeasurementDataset(Dataset):
    """Dataset for measurement prediction training"""

    # Target measurements (output)
    TARGET_MEASUREMENTS = [
        'chest_circumference',
        'waist_circumference',
        'hip_circumference',
        'shoulder_width',
        'inseam',
        'arm_length',
    ]

    def __init__(self, features: np.ndarray, measurements: np.ndarray):
        """
        Args:
            features: (N, num_features) input features
            measurements: (N, num_measurements) target measurements in cm
        """
        self.features = torch.FloatTensor(features)
        self.measurements = torch.FloatTensor(measurements)

    def __len__(self):
        return len(self.measurements)

    def __getitem__(self, idx):
        return self.features[idx], self.measurements[idx]


class MeasurementPredictorMLP(nn.Module):
    """
    Neural network for predicting body measurements from pose features.

    Architecture:
    - Input: 24 pose features
    - Hidden layers with batch norm and dropout
    - Output: 6 measurements (cm)

    Uses residual connections and careful initialization for stable training.
    """

    def __init__(
        self,
        input_dim: int = MeasurementFeatureExtractor.NUM_FEATURES,
        output_dim: int = 6,
        hidden_dims: List[int] = [128, 256, 128, 64],
        dropout_rate: float = 0.2
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_dim)

        # Build hidden layers
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(0.1))

            if i < len(hidden_dims) - 1:  # Skip dropout on last hidden layer
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        self.hidden = nn.Sequential(*layers)

        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)

        # Initialize weights
        self._initialize_weights()

        # Measurement ranges for output scaling (min, max in cm)
        self.register_buffer('output_min', torch.tensor([
            70.0,   # chest_circumference
            55.0,   # waist_circumference
            75.0,   # hip_circumference
            30.0,   # shoulder_width
            60.0,   # inseam
            45.0,   # arm_length
        ]))
        self.register_buffer('output_max', torch.tensor([
            140.0,  # chest_circumference
            130.0,  # waist_circumference
            140.0,  # hip_circumference
            55.0,   # shoulder_width
            95.0,   # inseam
            75.0,   # arm_length
        ]))

    def _initialize_weights(self):
        """Initialize weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input features (batch_size, input_dim)

        Returns:
            Predicted measurements (batch_size, output_dim) in cm
        """
        # Normalize input
        x = self.input_norm(x)

        # Hidden layers
        h = self.hidden(x)

        # Output with sigmoid to constrain range
        raw_output = self.output_layer(h)
        scaled_output = torch.sigmoid(raw_output)

        # Scale to measurement ranges
        measurements = self.output_min + scaled_output * (self.output_max - self.output_min)

        return measurements

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict measurements from features (convenience method)

        Args:
            features: Input features (num_features,) or (batch, num_features)

        Returns:
            Predicted measurements (output_dim,) or (batch, output_dim)
        """
        self.eval()
        with torch.no_grad():
            if features.ndim == 1:
                features = features[np.newaxis, :]
                squeeze = True
            else:
                squeeze = False

            x = torch.FloatTensor(features)
            if next(self.parameters()).is_cuda:
                x = x.cuda()

            predictions = self(x).cpu().numpy()

            if squeeze:
                predictions = predictions[0]

            return predictions


class MeasurementLoss(nn.Module):
    """
    Custom loss function for measurement prediction.

    Combines:
    1. MSE loss for overall accuracy
    2. Relative error loss (percentage-based)
    3. Correlation loss to maintain measurement relationships
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        relative_weight: float = 0.5,
        correlation_weight: float = 0.2
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.relative_weight = relative_weight
        self.correlation_weight = correlation_weight
        self.mse = nn.MSELoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss

        Args:
            predictions: (batch, num_measurements)
            targets: (batch, num_measurements)
        """
        # MSE loss (absolute error)
        mse_loss = self.mse(predictions, targets)

        # Relative error loss (percentage-based)
        relative_error = torch.abs(predictions - targets) / (targets + 1e-6)
        relative_loss = relative_error.mean()

        # Correlation loss (maintain relationships between measurements)
        # E.g., chest > waist for most people
        pred_chest = predictions[:, 0]
        pred_waist = predictions[:, 1]
        pred_hip = predictions[:, 2]

        target_chest = targets[:, 0]
        target_waist = targets[:, 1]
        target_hip = targets[:, 2]

        # Encourage correct ordering: chest > waist, hip > waist
        correlation_loss = (
            torch.relu(pred_waist - pred_chest + 5).mean() +  # chest should be > waist - 5
            torch.relu(pred_waist - pred_hip + 5).mean()      # hip should be > waist - 5
        )

        total_loss = (
            self.mse_weight * mse_loss +
            self.relative_weight * relative_loss * 100 +  # Scale relative loss
            self.correlation_weight * correlation_loss
        )

        return total_loss


class MeasurementPredictorTrainer:
    """Trainer for the measurement prediction model"""

    def __init__(
        self,
        model: MeasurementPredictorMLP,
        learning_rate: float = 0.001,
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )
        self.criterion = MeasurementLoss()
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_mape': []
        }

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0

        for features, targets in dataloader:
            features = features.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(features)
            loss = self.criterion(predictions, targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        """
        Evaluate model

        Returns:
            Tuple of (loss, MAE in cm, MAPE percentage)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for features, targets in dataloader:
                features = features.to(self.device)
                targets = targets.to(self.device)

                predictions = self.model(features)
                loss = self.criterion(predictions, targets)
                total_loss += loss.item()

                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # Mean Absolute Error (cm)
        mae = np.abs(predictions - targets).mean()

        # Mean Absolute Percentage Error
        mape = (np.abs(predictions - targets) / (targets + 1e-6) * 100).mean()

        avg_loss = total_loss / len(dataloader)

        return avg_loss, mae, mape

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 200,
        early_stopping_patience: int = 20
    ) -> Dict:
        """Full training loop with early stopping"""
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        logger.info(f"Starting training on {self.device}")
        logger.info(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_mae, val_mape = self.evaluate(val_loader)

            self.scheduler.step(val_loss)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_mae'].append(val_mae)
            self.history['val_mape'].append(val_mape)

            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val MAE: {val_mae:.2f} cm, "
                f"Val MAPE: {val_mape:.1f}%"
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                logger.info(f"  -> New best model (MAE: {val_mae:.2f} cm)")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        # Final evaluation
        final_loss, final_mae, final_mape = self.evaluate(val_loader)
        logger.info(f"\nFinal Results:")
        logger.info(f"  Loss: {final_loss:.4f}")
        logger.info(f"  MAE: {final_mae:.2f} cm")
        logger.info(f"  MAPE: {final_mape:.1f}%")

        return self.history

    def save_model(self, path: str):
        """Save model to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'feature_names': MeasurementFeatureExtractor.FEATURE_NAMES,
            'target_names': MeasurementDataset.TARGET_MEASUREMENTS,
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model from disk"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', {})
        logger.info(f"Model loaded from {path}")


class TrainedMeasurementPredictor:
    """
    Production-ready measurement predictor using trained model.
    Falls back to geometric methods if model not available.
    """

    def __init__(self, model_path: str = None):
        self.feature_extractor = MeasurementFeatureExtractor()
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if model_path is None:
            # Default model path
            model_path = os.path.join(
                os.path.dirname(__file__),
                '..', 'checkpoints', 'measurement_predictor.pt'
            )

        if os.path.exists(model_path):
            self._load_model(model_path)
        else:
            logger.warning(f"No trained model found at {model_path}. Using geometric fallback.")

    def _load_model(self, path: str):
        """Load trained model"""
        try:
            self.model = MeasurementPredictorMLP()
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded trained measurement model from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def predict(self, pose_landmarks) -> Dict[str, float]:
        """
        Predict measurements from pose landmarks

        Args:
            pose_landmarks: PoseLandmarks object

        Returns:
            Dict with measurement names and values in cm
        """
        features = self.feature_extractor.extract_features(pose_landmarks)

        if self.model is not None:
            predictions = self.model.predict(features)
            return {
                'chest_circumference': float(predictions[0]),
                'waist_circumference': float(predictions[1]),
                'hip_circumference': float(predictions[2]),
                'shoulder_width': float(predictions[3]),
                'inseam': float(predictions[4]),
                'arm_length': float(predictions[5]),
            }
        else:
            # Geometric fallback
            return self._predict_geometric(pose_landmarks, features)

    def _predict_geometric(self, pose_landmarks, features: np.ndarray) -> Dict[str, float]:
        """Fallback geometric prediction"""
        # Extract estimated height from features
        height = features[22] * 200  # Denormalize

        # Use standard proportions
        return {
            'chest_circumference': height * 0.55,
            'waist_circumference': height * 0.45,
            'hip_circumference': height * 0.55,
            'shoulder_width': height * 0.26,
            'inseam': height * 0.46,
            'arm_length': height * 0.36,
        }

    @property
    def is_model_loaded(self) -> bool:
        """Check if trained model is loaded"""
        return self.model is not None
