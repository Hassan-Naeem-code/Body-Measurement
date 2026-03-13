#!/usr/bin/env python3
"""
Import ground truth measurements from a CSV file into the database.

CSV format:
  image_path,actual_chest_circumference,actual_waist_circumference,actual_hip_circumference,
  actual_shoulder_width,actual_inseam,actual_arm_length,actual_height,
  gender,age_group,body_type,pose_type,measured_by,notes

Usage:
  python scripts/import_ground_truth.py path/to/measurements.csv
"""

import csv
import sys
import os

# Add parent dir to path so we can import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import SessionLocal
from app.models.ground_truth import GroundTruth

FLOAT_FIELDS = {
    "actual_chest_circumference",
    "actual_waist_circumference",
    "actual_hip_circumference",
    "actual_shoulder_width",
    "actual_inseam",
    "actual_arm_length",
    "actual_height",
}

STRING_FIELDS = {
    "image_path",
    "gender",
    "age_group",
    "body_type",
    "pose_type",
    "image_quality",
    "measured_by",
    "measurement_date",
    "notes",
}


def import_csv(filepath: str) -> int:
    db = SessionLocal()
    count = 0

    try:
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                kwargs = {}
                for field in FLOAT_FIELDS:
                    val = row.get(field, "").strip()
                    kwargs[field] = float(val) if val else None
                for field in STRING_FIELDS:
                    val = row.get(field, "").strip()
                    kwargs[field] = val if val else None

                if not kwargs.get("image_path"):
                    print(f"Skipping row {count + 1}: missing image_path")
                    continue

                gt = GroundTruth(**kwargs)
                db.add(gt)
                count += 1

        db.commit()
        print(f"Imported {count} ground truth samples.")
    except Exception as e:
        db.rollback()
        print(f"Error: {e}")
        raise
    finally:
        db.close()

    return count


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/import_ground_truth.py <csv_file>")
        sys.exit(1)

    import_csv(sys.argv[1])
