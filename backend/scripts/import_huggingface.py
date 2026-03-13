#!/usr/bin/env python3
"""
Import TrainingDataPro body-measurements-dataset from Hugging Face.

21 subjects with front + side photos, and detailed measurements JSON including:
chest, waist, hips, shoulder width, arm length, thigh, height, weight, gender, age.

Usage:
  pip install huggingface_hub
  python -m scripts.import_huggingface

Run from the backend/ directory.
"""

import os
import sys
import csv
import json
import shutil
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from huggingface_hub import hf_hub_download
from app.core.database import SessionLocal
from app.models.ground_truth import GroundTruth

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPO_ID = "TrainingDataPro/body-measurements-dataset"
GROUND_TRUTH_IMAGES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data", "validation", "images"
)
os.makedirs(GROUND_TRUTH_IMAGES_DIR, exist_ok=True)


def safe_float(val) -> float:
    """Parse measurement value. Handles strings like '70.0_tbr' (to be reviewed)."""
    if val is None:
        return None
    val = str(val).replace("_tbr", "").strip()
    try:
        return round(float(val), 2)
    except (ValueError, TypeError):
        return None


def age_to_group(age_str: str) -> str:
    """Convert age string to age group."""
    try:
        age = int(age_str)
        if age < 25:
            return "18-24"
        elif age < 35:
            return "25-34"
        elif age < 45:
            return "35-44"
        elif age < 55:
            return "45-54"
        else:
            return "55+"
    except (ValueError, TypeError):
        return None


def import_huggingface():
    """Import TrainingDataPro body-measurements-dataset."""

    logger.info("Downloading dataset index from Hugging Face...")
    csv_path = hf_hub_download(repo_id=REPO_ID, filename="body.csv", repo_type="dataset")

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    logger.info("Found %d subjects in dataset", len(rows))

    db = SessionLocal()
    imported = 0

    try:
        for i, row in enumerate(rows):
            logger.info("Processing subject %d/%d ...", i + 1, len(rows))

            # Download measurement JSON
            try:
                meas_path = hf_hub_download(
                    repo_id=REPO_ID, filename=row["measurements"], repo_type="dataset"
                )
                with open(meas_path) as f:
                    meas = json.load(f)
            except Exception as e:
                logger.warning("Could not load measurements for %s: %s", row["measurements"], e)
                continue

            # Download front image
            try:
                front_path = hf_hub_download(
                    repo_id=REPO_ID, filename=row["front"], repo_type="dataset"
                )
                img_filename = f"hf_front_{i:03d}.jpg"
                local_img = os.path.join(GROUND_TRUTH_IMAGES_DIR, img_filename)
                shutil.copy2(front_path, local_img)
            except Exception as e:
                logger.warning("Could not download front image for subject %d: %s", i, e)
                continue

            # Also download side image for dual-view training
            side_img_path = ""
            try:
                side_path = hf_hub_download(
                    repo_id=REPO_ID, filename=row["side"], repo_type="dataset"
                )
                side_filename = f"hf_side_{i:03d}.jpg"
                side_local = os.path.join(GROUND_TRUTH_IMAGES_DIR, side_filename)
                shutil.copy2(side_path, side_local)
                side_img_path = side_local
            except Exception:
                pass

            # Map fields
            # Dataset fields: arm_length_cm, chest_circumference_cm, front_build_cm,
            #   hips_circumference_cm, leg_length_cm, neck_circumference_cm,
            #   neck_waist_length_front_cm, pelvis_circumference_cm, shoulder_width_cm,
            #   thigh_circumference_cm, under_chest_circumference_cm,
            #   waist_circumference_cm, height, weight, age, gender, race, profession

            gender_val = meas.get("gender", "").lower()
            if gender_val not in ("male", "female"):
                gender_val = "other"

            notes_parts = [
                "HuggingFace TrainingDataPro dataset.",
                f"Weight: {meas.get('weight')}kg.",
                f"Neck: {meas.get('neck_circumference_cm')}cm.",
                f"Thigh: {meas.get('thigh_circumference_cm')}cm.",
                f"Race: {meas.get('race')}.",
            ]
            if side_img_path:
                notes_parts.append(f"Side image: {side_img_path}")

            gt = GroundTruth(
                image_path=local_img,
                actual_chest_circumference=safe_float(meas.get("chest_circumference_cm")),
                actual_waist_circumference=safe_float(meas.get("waist_circumference_cm")),
                actual_hip_circumference=safe_float(
                    meas.get("hips_circumference_cm") or meas.get("pelvis_circumference_cm")
                ),
                actual_shoulder_width=safe_float(meas.get("shoulder_width_cm")),
                actual_inseam=safe_float(meas.get("leg_length_cm")),
                actual_arm_length=safe_float(meas.get("arm_length_cm")),
                actual_height=safe_float(meas.get("height")),
                gender=gender_val,
                age_group=age_to_group(meas.get("age")),
                body_type=None,
                pose_type="front",
                measured_by="trainingdatapro_dataset",
                notes=" ".join(notes_parts),
            )
            db.add(gt)
            imported += 1

        db.commit()
        logger.info("Done! Imported %d subjects from HuggingFace.", imported)

    except Exception as e:
        db.rollback()
        logger.error("Import failed: %s", e)
        raise
    finally:
        db.close()


if __name__ == "__main__":
    import_huggingface()
