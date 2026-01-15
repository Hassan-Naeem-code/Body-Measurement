"""
Synthetic Data Generator for Body Measurement Training

Generates synthetic training data with known ground truth measurements.
Two approaches:
1. Mathematical body model (no external dependencies)
2. SMPL-based (requires smplx library - more realistic)

The mathematical approach creates body proportions based on anthropometric
research data, allowing us to generate thousands of training samples with
known measurements.
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import random
import logging

logger = logging.getLogger(__name__)


@dataclass
class SyntheticBodyData:
    """Synthetic body data with ground truth measurements"""
    # Identity
    sample_id: str
    gender: str  # 'male' or 'female'
    age_group: str  # 'adult', 'teen', 'child'
    body_type: str  # 'slim', 'average', 'athletic', 'heavy'

    # Ground truth measurements (in cm)
    height: float
    shoulder_width: float
    chest_circumference: float
    waist_circumference: float
    hip_circumference: float
    inseam: float
    arm_length: float

    # Derived ratios (for training)
    shoulder_hip_ratio: float
    waist_hip_ratio: float
    chest_waist_ratio: float

    # Pose landmarks (normalized 0-1)
    landmarks: Dict[str, Dict[str, float]]

    # Recommended size (based on measurements)
    recommended_size: str


class AnthropometricModel:
    """
    Mathematical model for generating realistic body proportions
    Based on anthropometric research data (CAESAR, ANSUR datasets)
    """

    # Average measurements by gender (in cm) - from anthropometric studies
    MALE_AVERAGES = {
        'height': 175.0,
        'shoulder_width': 46.0,
        'chest_circumference': 102.0,
        'waist_circumference': 88.0,
        'hip_circumference': 100.0,
        'inseam': 81.0,
        'arm_length': 60.0,
    }

    FEMALE_AVERAGES = {
        'height': 162.0,
        'shoulder_width': 38.0,
        'chest_circumference': 92.0,
        'waist_circumference': 74.0,
        'hip_circumference': 100.0,
        'inseam': 74.0,
        'arm_length': 54.0,
    }

    # Standard deviations (for generating variety)
    MALE_STD = {
        'height': 7.0,
        'shoulder_width': 3.0,
        'chest_circumference': 8.0,
        'waist_circumference': 10.0,
        'hip_circumference': 7.0,
        'inseam': 5.0,
        'arm_length': 4.0,
    }

    FEMALE_STD = {
        'height': 6.5,
        'shoulder_width': 2.5,
        'chest_circumference': 7.0,
        'waist_circumference': 9.0,
        'hip_circumference': 8.0,
        'inseam': 5.0,
        'arm_length': 3.5,
    }

    # Body type modifiers
    BODY_TYPE_MODIFIERS = {
        'slim': {
            'chest_circumference': -0.15,
            'waist_circumference': -0.20,
            'hip_circumference': -0.10,
        },
        'average': {
            'chest_circumference': 0.0,
            'waist_circumference': 0.0,
            'hip_circumference': 0.0,
        },
        'athletic': {
            'chest_circumference': 0.10,
            'waist_circumference': -0.10,
            'hip_circumference': 0.05,
            'shoulder_width': 0.10,
        },
        'heavy': {
            'chest_circumference': 0.20,
            'waist_circumference': 0.30,
            'hip_circumference': 0.20,
        },
    }

    # Size charts (chest circumference ranges)
    SIZE_CHARTS = {
        'male': {
            'XS': (82, 88),
            'S': (88, 94),
            'M': (94, 102),
            'L': (102, 110),
            'XL': (110, 118),
            'XXL': (118, 130),
        },
        'female': {
            'XS': (76, 82),
            'S': (82, 88),
            'M': (88, 94),
            'L': (94, 102),
            'XL': (102, 110),
            'XXL': (110, 120),
        },
    }

    def generate_sample(
        self,
        gender: str = None,
        body_type: str = None,
        age_group: str = 'adult'
    ) -> SyntheticBodyData:
        """
        Generate a single synthetic body sample

        Args:
            gender: 'male' or 'female' (random if None)
            body_type: 'slim', 'average', 'athletic', 'heavy' (random if None)
            age_group: 'adult', 'teen', 'child'

        Returns:
            SyntheticBodyData with all measurements and landmarks
        """
        # Random selection if not specified
        if gender is None:
            gender = random.choice(['male', 'female'])
        if body_type is None:
            body_type = random.choice(['slim', 'average', 'athletic', 'heavy'])

        # Get base averages and std
        if gender == 'male':
            averages = self.MALE_AVERAGES.copy()
            stds = self.MALE_STD
        else:
            averages = self.FEMALE_AVERAGES.copy()
            stds = self.FEMALE_STD

        # Apply body type modifiers
        modifiers = self.BODY_TYPE_MODIFIERS[body_type]
        for key, modifier in modifiers.items():
            if key in averages:
                averages[key] *= (1 + modifier)

        # Apply age group modifiers
        if age_group == 'teen':
            averages['height'] *= 0.92
            averages['shoulder_width'] *= 0.90
            averages['chest_circumference'] *= 0.88
            averages['waist_circumference'] *= 0.85
            averages['hip_circumference'] *= 0.88
        elif age_group == 'child':
            averages['height'] *= 0.75
            averages['shoulder_width'] *= 0.70
            averages['chest_circumference'] *= 0.65
            averages['waist_circumference'] *= 0.60
            averages['hip_circumference'] *= 0.65

        # Generate measurements with normal distribution
        measurements = {}
        for key in averages:
            mean = averages[key]
            std = stds.get(key, mean * 0.05)
            measurements[key] = max(mean * 0.5, np.random.normal(mean, std))

        # Ensure body proportions are realistic
        measurements = self._enforce_realistic_proportions(measurements, gender)

        # Calculate derived ratios
        shoulder_hip_ratio = measurements['shoulder_width'] / (measurements['hip_circumference'] / np.pi / 2)
        waist_hip_ratio = measurements['waist_circumference'] / measurements['hip_circumference']
        chest_waist_ratio = measurements['chest_circumference'] / measurements['waist_circumference']

        # Generate pose landmarks
        landmarks = self._generate_landmarks(measurements)

        # Determine size
        recommended_size = self._get_size(measurements['chest_circumference'], gender)

        # Create sample ID
        sample_id = f"{gender[0]}_{body_type[:3]}_{random.randint(10000, 99999)}"

        return SyntheticBodyData(
            sample_id=sample_id,
            gender=gender,
            age_group=age_group,
            body_type=body_type,
            height=round(measurements['height'], 1),
            shoulder_width=round(measurements['shoulder_width'], 1),
            chest_circumference=round(measurements['chest_circumference'], 1),
            waist_circumference=round(measurements['waist_circumference'], 1),
            hip_circumference=round(measurements['hip_circumference'], 1),
            inseam=round(measurements['inseam'], 1),
            arm_length=round(measurements['arm_length'], 1),
            shoulder_hip_ratio=round(shoulder_hip_ratio, 3),
            waist_hip_ratio=round(waist_hip_ratio, 3),
            chest_waist_ratio=round(chest_waist_ratio, 3),
            landmarks=landmarks,
            recommended_size=recommended_size,
        )

    def _enforce_realistic_proportions(
        self,
        measurements: Dict[str, float],
        gender: str
    ) -> Dict[str, float]:
        """Ensure body proportions are anatomically realistic"""

        # Inseam should be ~45-48% of height
        expected_inseam = measurements['height'] * random.uniform(0.45, 0.48)
        measurements['inseam'] = (measurements['inseam'] + expected_inseam) / 2

        # Arm length should be ~35-38% of height
        expected_arm = measurements['height'] * random.uniform(0.35, 0.38)
        measurements['arm_length'] = (measurements['arm_length'] + expected_arm) / 2

        # Shoulder width constraints
        if gender == 'male':
            min_shoulder = measurements['height'] * 0.24
            max_shoulder = measurements['height'] * 0.30
        else:
            min_shoulder = measurements['height'] * 0.21
            max_shoulder = measurements['height'] * 0.26

        measurements['shoulder_width'] = np.clip(
            measurements['shoulder_width'],
            min_shoulder,
            max_shoulder
        )

        # Hip circumference should be reasonable relative to height
        min_hip = measurements['height'] * 0.52
        max_hip = measurements['height'] * 0.72
        measurements['hip_circumference'] = np.clip(
            measurements['hip_circumference'],
            min_hip,
            max_hip
        )

        # Waist should be less than hips (in most cases)
        if measurements['waist_circumference'] > measurements['hip_circumference'] * 1.05:
            measurements['waist_circumference'] = measurements['hip_circumference'] * random.uniform(0.85, 1.02)

        return measurements

    def _generate_landmarks(self, measurements: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Generate normalized (0-1) pose landmarks based on measurements

        This simulates what MediaPipe would detect from a full-body image
        """
        height = measurements['height']
        shoulder_width = measurements['shoulder_width']
        hip_width = measurements['hip_circumference'] / np.pi / 2  # Approximate width from circumference
        arm_length = measurements['arm_length']

        # Add some noise to simulate real detection
        noise = lambda: random.uniform(-0.02, 0.02)

        # Vertical positions (normalized to body height)
        nose_y = 0.05 + noise()
        shoulder_y = 0.18 + noise()
        elbow_y = 0.36 + noise()  # Elbows at ~36% of height
        wrist_y = 0.50 + noise()  # Wrists at ~50% of height
        hip_y = 0.52 + noise()
        knee_y = 0.75 + noise()
        ankle_y = 0.95 + noise()

        # Horizontal positions (centered at 0.5)
        center_x = 0.5
        shoulder_offset = (shoulder_width / height) / 2
        hip_offset = (hip_width / height) / 2

        # Arm positions - slightly outside shoulders, hanging naturally
        arm_offset = shoulder_offset + 0.02  # Arms slightly outside shoulders
        elbow_offset = shoulder_offset + 0.05
        wrist_offset = shoulder_offset + 0.03

        landmarks = {
            'nose': {'x': center_x + noise(), 'y': nose_y, 'visibility': 0.95 + noise() * 0.5},
            'left_shoulder': {'x': center_x - shoulder_offset + noise(), 'y': shoulder_y, 'visibility': 0.92 + noise() * 0.5},
            'right_shoulder': {'x': center_x + shoulder_offset + noise(), 'y': shoulder_y, 'visibility': 0.92 + noise() * 0.5},
            'left_elbow': {'x': center_x - elbow_offset + noise(), 'y': elbow_y, 'visibility': 0.88 + noise() * 0.5},
            'right_elbow': {'x': center_x + elbow_offset + noise(), 'y': elbow_y, 'visibility': 0.88 + noise() * 0.5},
            'left_wrist': {'x': center_x - wrist_offset + noise(), 'y': wrist_y, 'visibility': 0.85 + noise() * 0.5},
            'right_wrist': {'x': center_x + wrist_offset + noise(), 'y': wrist_y, 'visibility': 0.85 + noise() * 0.5},
            'left_hip': {'x': center_x - hip_offset + noise(), 'y': hip_y, 'visibility': 0.88 + noise() * 0.5},
            'right_hip': {'x': center_x + hip_offset + noise(), 'y': hip_y, 'visibility': 0.88 + noise() * 0.5},
            'left_knee': {'x': center_x - hip_offset * 0.9 + noise(), 'y': knee_y, 'visibility': 0.85 + noise() * 0.5},
            'right_knee': {'x': center_x + hip_offset * 0.9 + noise(), 'y': knee_y, 'visibility': 0.85 + noise() * 0.5},
            'left_ankle': {'x': center_x - hip_offset * 0.8 + noise(), 'y': ankle_y, 'visibility': 0.80 + noise() * 0.5},
            'right_ankle': {'x': center_x + hip_offset * 0.8 + noise(), 'y': ankle_y, 'visibility': 0.80 + noise() * 0.5},
        }

        return landmarks

    def _get_size(self, chest_circumference: float, gender: str) -> str:
        """Determine clothing size based on chest circumference"""
        chart = self.SIZE_CHARTS.get(gender, self.SIZE_CHARTS['male'])

        for size, (min_chest, max_chest) in chart.items():
            if min_chest <= chest_circumference < max_chest:
                return size

        # Edge cases
        if chest_circumference < list(chart.values())[0][0]:
            return 'XS'
        return 'XXL'


class SyntheticDataGenerator:
    """
    Main generator class for creating training datasets
    """

    def __init__(self, output_dir: str = None):
        self.model = AnthropometricModel()
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(__file__),
            '..', 'data', 'synthetic'
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_dataset(
        self,
        num_samples: int = 10000,
        male_ratio: float = 0.5,
        body_type_distribution: Dict[str, float] = None,
        save: bool = True
    ) -> List[SyntheticBodyData]:
        """
        Generate a full training dataset

        Args:
            num_samples: Number of samples to generate
            male_ratio: Proportion of male samples (0-1)
            body_type_distribution: Dict of body_type -> proportion
            save: Whether to save to disk

        Returns:
            List of SyntheticBodyData samples
        """
        if body_type_distribution is None:
            body_type_distribution = {
                'slim': 0.20,
                'average': 0.40,
                'athletic': 0.25,
                'heavy': 0.15,
            }

        samples = []
        num_male = int(num_samples * male_ratio)
        num_female = num_samples - num_male

        logger.info(f"Generating {num_samples} synthetic samples...")
        logger.info(f"  Male: {num_male}, Female: {num_female}")

        # Generate male samples
        for i in range(num_male):
            body_type = self._sample_body_type(body_type_distribution)
            sample = self.model.generate_sample(gender='male', body_type=body_type)
            samples.append(sample)

            if (i + 1) % 1000 == 0:
                logger.info(f"  Generated {i + 1}/{num_male} male samples")

        # Generate female samples
        for i in range(num_female):
            body_type = self._sample_body_type(body_type_distribution)
            sample = self.model.generate_sample(gender='female', body_type=body_type)
            samples.append(sample)

            if (i + 1) % 1000 == 0:
                logger.info(f"  Generated {i + 1}/{num_female} female samples")

        # Shuffle
        random.shuffle(samples)

        if save:
            self._save_dataset(samples)

        logger.info(f"Generated {len(samples)} samples")
        return samples

    def _sample_body_type(self, distribution: Dict[str, float]) -> str:
        """Sample body type based on distribution"""
        types = list(distribution.keys())
        weights = list(distribution.values())
        return random.choices(types, weights=weights, k=1)[0]

    def _save_dataset(self, samples: List[SyntheticBodyData]):
        """Save dataset to disk"""
        # Save as JSON
        json_path = os.path.join(self.output_dir, 'synthetic_dataset.json')
        data = [asdict(s) for s in samples]

        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved dataset to {json_path}")

        # Save summary statistics
        self._save_statistics(samples)

    def _save_statistics(self, samples: List[SyntheticBodyData]):
        """Save dataset statistics"""
        stats = {
            'total_samples': len(samples),
            'gender_distribution': {},
            'body_type_distribution': {},
            'size_distribution': {},
            'measurement_stats': {},
        }

        # Count distributions
        for sample in samples:
            stats['gender_distribution'][sample.gender] = \
                stats['gender_distribution'].get(sample.gender, 0) + 1
            stats['body_type_distribution'][sample.body_type] = \
                stats['body_type_distribution'].get(sample.body_type, 0) + 1
            stats['size_distribution'][sample.recommended_size] = \
                stats['size_distribution'].get(sample.recommended_size, 0) + 1

        # Measurement statistics
        measurements = ['height', 'shoulder_width', 'chest_circumference',
                       'waist_circumference', 'hip_circumference']

        for measure in measurements:
            values = [getattr(s, measure) for s in samples]
            stats['measurement_stats'][measure] = {
                'mean': round(np.mean(values), 2),
                'std': round(np.std(values), 2),
                'min': round(np.min(values), 2),
                'max': round(np.max(values), 2),
            }

        stats_path = os.path.join(self.output_dir, 'dataset_statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved statistics to {stats_path}")

    def load_dataset(self) -> List[SyntheticBodyData]:
        """Load dataset from disk"""
        json_path = os.path.join(self.output_dir, 'synthetic_dataset.json')

        with open(json_path, 'r') as f:
            data = json.load(f)

        samples = [SyntheticBodyData(**d) for d in data]
        logger.info(f"Loaded {len(samples)} samples from {json_path}")

        return samples


def prepare_training_data(samples: List[SyntheticBodyData]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare feature matrix and labels for training

    Returns:
        Tuple of (features, labels) where:
        - features: (N, num_features) numpy array
        - labels: (N,) numpy array (0=female, 1=male)
    """
    features = []
    labels = []

    for sample in samples:
        # Extract features from landmarks
        landmarks = sample.landmarks
        shoulder_width_norm = abs(
            landmarks['left_shoulder']['x'] - landmarks['right_shoulder']['x']
        )
        hip_width_norm = abs(
            landmarks['left_hip']['x'] - landmarks['right_hip']['x']
        )

        feature_vector = [
            sample.shoulder_hip_ratio,
            shoulder_width_norm,
            hip_width_norm,
            sample.waist_hip_ratio,
            sample.chest_waist_ratio,
            0.65,  # torso_leg_ratio (estimated)
            sample.height / sample.shoulder_width,  # body_aspect_ratio
            shoulder_width_norm * sample.chest_waist_ratio,  # upper_body_mass
            hip_width_norm / max(shoulder_width_norm, 0.01),  # lower_body_mass
            0.05,  # shoulder_slope
            0.05,  # hip_slope
        ]

        features.append(feature_vector)
        labels.append(1 if sample.gender == 'male' else 0)

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64)


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    generator = SyntheticDataGenerator()
    samples = generator.generate_dataset(num_samples=10000)

    print(f"\nGenerated {len(samples)} samples")
    print(f"Example sample: {samples[0]}")
