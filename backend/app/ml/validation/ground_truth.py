"""
Ground Truth Data Schema and Storage for Validation

Stores real tape measurements to compare against ML predictions.
This is CRITICAL for knowing actual accuracy.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class GroundTruthMeasurement:
    """
    Ground truth measurements from real tape measurements

    All measurements in cm for consistency.
    """
    # Identification
    id: str  # Unique identifier
    image_path: str  # Path to the image file

    # Actual measurements (from tape measure)
    actual_chest_circumference: Optional[float] = None
    actual_waist_circumference: Optional[float] = None
    actual_hip_circumference: Optional[float] = None
    actual_shoulder_width: Optional[float] = None
    actual_inseam: Optional[float] = None
    actual_arm_length: Optional[float] = None
    actual_height: Optional[float] = None  # Person's actual height

    # Demographics (optional but helpful for analysis)
    gender: Optional[str] = None  # 'male', 'female', 'other'
    age_group: Optional[str] = None  # 'child', 'teen', 'adult'
    body_type: Optional[str] = None  # 'slim', 'average', 'athletic', 'plus'

    # Image metadata
    pose_type: Optional[str] = None  # 'front', 'side', 'angled'
    image_quality: Optional[str] = None  # 'good', 'average', 'poor'

    # Measurement metadata
    measured_by: Optional[str] = None  # Who took the measurements
    measurement_date: Optional[str] = None  # When measurements were taken
    notes: Optional[str] = None  # Any additional notes

    # Timestamp
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'GroundTruthMeasurement':
        """Create from dictionary"""
        return cls(**data)

    def get_actual_measurements(self) -> Dict[str, float]:
        """Get all actual measurements as a dictionary"""
        measurements = {}
        if self.actual_chest_circumference is not None:
            measurements['chest_circumference'] = self.actual_chest_circumference
        if self.actual_waist_circumference is not None:
            measurements['waist_circumference'] = self.actual_waist_circumference
        if self.actual_hip_circumference is not None:
            measurements['hip_circumference'] = self.actual_hip_circumference
        if self.actual_shoulder_width is not None:
            measurements['shoulder_width'] = self.actual_shoulder_width
        if self.actual_inseam is not None:
            measurements['inseam'] = self.actual_inseam
        if self.actual_arm_length is not None:
            measurements['arm_length'] = self.actual_arm_length
        if self.actual_height is not None:
            measurements['height'] = self.actual_height
        return measurements


class ValidationDataset:
    """
    Manages a collection of ground truth measurements for validation

    Stores data in JSON format for easy editing and version control.
    """

    def __init__(self, data_dir: str = None):
        """
        Initialize validation dataset

        Args:
            data_dir: Directory to store validation data
        """
        if data_dir is None:
            # Default to backend/app/data/validation
            data_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'data', 'validation'
            )

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_file = self.data_dir / 'ground_truth.json'
        self.images_dir = self.data_dir / 'images'
        self.images_dir.mkdir(exist_ok=True)

        self._measurements: List[GroundTruthMeasurement] = []
        self._load()

    def _load(self):
        """Load dataset from file"""
        if self.dataset_file.exists():
            try:
                with open(self.dataset_file, 'r') as f:
                    data = json.load(f)
                    self._measurements = [
                        GroundTruthMeasurement.from_dict(m)
                        for m in data.get('measurements', [])
                    ]
                logger.info(f"Loaded {len(self._measurements)} ground truth measurements")
            except Exception as e:
                logger.error(f"Error loading validation dataset: {e}")
                self._measurements = []
        else:
            self._measurements = []

    def _save(self):
        """Save dataset to file"""
        try:
            data = {
                'version': '1.0',
                'updated_at': datetime.utcnow().isoformat(),
                'total_samples': len(self._measurements),
                'measurements': [m.to_dict() for m in self._measurements]
            }
            with open(self.dataset_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self._measurements)} ground truth measurements")
        except Exception as e:
            logger.error(f"Error saving validation dataset: {e}")

    def add(self, measurement: GroundTruthMeasurement) -> str:
        """
        Add a new ground truth measurement

        Args:
            measurement: Ground truth measurement to add

        Returns:
            ID of the added measurement
        """
        # Generate ID if not provided
        if not measurement.id:
            measurement.id = f"gt_{len(self._measurements) + 1:04d}"

        self._measurements.append(measurement)
        self._save()
        return measurement.id

    def get(self, id: str) -> Optional[GroundTruthMeasurement]:
        """Get measurement by ID"""
        for m in self._measurements:
            if m.id == id:
                return m
        return None

    def get_all(self) -> List[GroundTruthMeasurement]:
        """Get all measurements"""
        return self._measurements.copy()

    def update(self, id: str, measurement: GroundTruthMeasurement) -> bool:
        """Update an existing measurement"""
        for i, m in enumerate(self._measurements):
            if m.id == id:
                measurement.id = id
                self._measurements[i] = measurement
                self._save()
                return True
        return False

    def delete(self, id: str) -> bool:
        """Delete a measurement"""
        for i, m in enumerate(self._measurements):
            if m.id == id:
                del self._measurements[i]
                self._save()
                return True
        return False

    def filter_by(
        self,
        gender: Optional[str] = None,
        age_group: Optional[str] = None,
        body_type: Optional[str] = None,
        pose_type: Optional[str] = None,
    ) -> List[GroundTruthMeasurement]:
        """Filter measurements by criteria"""
        results = self._measurements

        if gender:
            results = [m for m in results if m.gender == gender]
        if age_group:
            results = [m for m in results if m.age_group == age_group]
        if body_type:
            results = [m for m in results if m.body_type == body_type]
        if pose_type:
            results = [m for m in results if m.pose_type == pose_type]

        return results

    def get_statistics(self) -> dict:
        """Get dataset statistics"""
        stats = {
            'total_samples': len(self._measurements),
            'by_gender': {},
            'by_age_group': {},
            'by_body_type': {},
            'by_pose_type': {},
            'measurements_coverage': {},
        }

        for m in self._measurements:
            # Count by gender
            if m.gender:
                stats['by_gender'][m.gender] = stats['by_gender'].get(m.gender, 0) + 1

            # Count by age group
            if m.age_group:
                stats['by_age_group'][m.age_group] = stats['by_age_group'].get(m.age_group, 0) + 1

            # Count by body type
            if m.body_type:
                stats['by_body_type'][m.body_type] = stats['by_body_type'].get(m.body_type, 0) + 1

            # Count by pose type
            if m.pose_type:
                stats['by_pose_type'][m.pose_type] = stats['by_pose_type'].get(m.pose_type, 0) + 1

            # Count measurement coverage
            for field in ['chest_circumference', 'waist_circumference', 'hip_circumference',
                         'shoulder_width', 'inseam', 'arm_length', 'height']:
                actual_field = f'actual_{field}'
                if getattr(m, actual_field, None) is not None:
                    stats['measurements_coverage'][field] = \
                        stats['measurements_coverage'].get(field, 0) + 1

        return stats

    def export_to_csv(self, filepath: str) -> bool:
        """Export dataset to CSV"""
        try:
            import csv

            fieldnames = [
                'id', 'image_path', 'actual_chest_circumference',
                'actual_waist_circumference', 'actual_hip_circumference',
                'actual_shoulder_width', 'actual_inseam', 'actual_arm_length',
                'actual_height', 'gender', 'age_group', 'body_type', 'pose_type'
            ]

            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for m in self._measurements:
                    row = {k: getattr(m, k, None) for k in fieldnames}
                    writer.writerow(row)

            return True
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False

    def import_from_csv(self, filepath: str) -> int:
        """
        Import ground truth data from CSV

        CSV should have columns matching GroundTruthMeasurement fields.

        Returns:
            Number of measurements imported
        """
        try:
            import csv

            count = 0
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert empty strings to None
                    cleaned_row = {}
                    for k, v in row.items():
                        if v == '' or v is None:
                            cleaned_row[k] = None
                        elif k.startswith('actual_') and v:
                            cleaned_row[k] = float(v)
                        else:
                            cleaned_row[k] = v

                    measurement = GroundTruthMeasurement.from_dict(cleaned_row)
                    self.add(measurement)
                    count += 1

            return count
        except Exception as e:
            logger.error(f"Error importing from CSV: {e}")
            return 0
