#!/usr/bin/env python3
"""
Import BodyM dataset from AWS S3 into the ground truth database.

BodyM: 2,018 subjects with 14 body measurements + silhouette images.
Source: s3://amazon-bodym (no auth required)

Usage:
  pip install awscli
  python -m scripts.import_bodym [--limit 100] [--skip-images]

Run from the backend/ directory.
"""

import os
import sys
import csv
import subprocess
import argparse
import logging
import tempfile
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.core.database import SessionLocal
from app.models.ground_truth import GroundTruth

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

S3_BUCKET = "s3://amazon-bodym"
GROUND_TRUTH_IMAGES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data", "validation", "images"
)
os.makedirs(GROUND_TRUTH_IMAGES_DIR, exist_ok=True)


def download_s3_file(s3_path: str, local_path: str) -> bool:
    """Download a file from S3 without authentication."""
    try:
        subprocess.run(
            ["aws", "s3", "cp", "--no-sign-request", s3_path, local_path],
            check=True, capture_output=True, text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.warning("Failed to download %s: %s", s3_path, e.stderr)
        return False


def load_measurements(csv_path: str) -> dict:
    """Load measurements CSV into a dict keyed by subject_id."""
    data = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["subject_id"]] = row
    return data


def load_metadata(csv_path: str) -> dict:
    """Load height/weight/gender metadata."""
    data = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["subject_id"]] = row
    return data


def load_photo_map(csv_path: str) -> dict:
    """Load subject -> photo_id mapping. Returns first photo per subject."""
    data = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["subject_id"]
            if sid not in data:
                data[sid] = row["photo_id"]
    return data


def import_bodym(limit: int = None, skip_images: bool = False):
    """Import BodyM dataset into ground truth database."""

    # Download CSV files to temp directory
    tmp = tempfile.mkdtemp(prefix="bodym_")
    logger.info("Downloading BodyM metadata to %s ...", tmp)

    measurements_csv = os.path.join(tmp, "measurements.csv")
    hwg_csv = os.path.join(tmp, "hwg_metadata.csv")
    photos_csv = os.path.join(tmp, "subject_to_photo_map.csv")

    for name, local in [
        ("train/measurements.csv", measurements_csv),
        ("train/hwg_metadata.csv", hwg_csv),
        ("train/subject_to_photo_map.csv", photos_csv),
    ]:
        if not download_s3_file(f"{S3_BUCKET}/{name}", local):
            logger.error("Could not download %s. Is AWS CLI installed?", name)
            return

    # Parse data
    measurements = load_measurements(measurements_csv)
    metadata = load_metadata(hwg_csv)
    photo_map = load_photo_map(photos_csv)

    logger.info("Loaded %d subjects with measurements", len(measurements))
    logger.info("Loaded %d subjects with metadata", len(metadata))
    logger.info("Loaded %d subjects with photos", len(photo_map))

    # Find subjects that have both measurements and metadata
    common_ids = set(measurements.keys()) & set(metadata.keys())
    logger.info("Subjects with both measurements and metadata: %d", len(common_ids))

    if limit:
        common_ids = list(common_ids)[:limit]
        logger.info("Limiting to %d subjects", limit)

    db = SessionLocal()
    imported = 0
    skipped = 0

    try:
        for sid in common_ids:
            m = measurements[sid]
            meta = metadata[sid]

            # Download silhouette image if available
            image_path = ""
            if not skip_images and sid in photo_map:
                photo_id = photo_map[sid]
                img_filename = f"bodym_{photo_id}.png"
                local_img = os.path.join(GROUND_TRUTH_IMAGES_DIR, img_filename)

                if not os.path.exists(local_img):
                    s3_img = f"{S3_BUCKET}/train/mask/{photo_id}.png"
                    if download_s3_file(s3_img, local_img):
                        image_path = local_img
                    else:
                        # Try mask_left
                        s3_img = f"{S3_BUCKET}/train/mask_left/{photo_id}.png"
                        download_s3_file(s3_img, local_img)
                        image_path = local_img if os.path.exists(local_img) else ""
                else:
                    image_path = local_img

            if not image_path and not skip_images:
                skipped += 1
                continue

            if skip_images:
                image_path = f"bodym_silhouette_{sid[:12]}"

            # Map BodyM columns to our ground truth fields
            # BodyM: ankle, arm-length, bicep, calf, chest, forearm, height, hip,
            #        leg-length, shoulder-breadth, shoulder-to-crotch, thigh, waist, wrist
            gt = GroundTruth(
                image_path=image_path,
                actual_chest_circumference=safe_float(m.get("chest")),
                actual_waist_circumference=safe_float(m.get("waist")),
                actual_hip_circumference=safe_float(m.get("hip")),
                actual_shoulder_width=safe_float(m.get("shoulder-breadth")),
                actual_inseam=safe_float(m.get("leg-length")),
                actual_arm_length=safe_float(m.get("arm-length")),
                actual_height=safe_float(meta.get("height_cm")),
                gender=meta.get("gender", "").lower() or None,
                body_type=None,
                pose_type="front",
                measured_by="bodym_dataset",
                notes=f"BodyM dataset. Weight: {meta.get('weight_kg')}kg. "
                      f"Thigh: {m.get('thigh')}cm. Bicep: {m.get('bicep')}cm.",
            )
            db.add(gt)
            imported += 1

            if imported % 100 == 0:
                db.commit()
                logger.info("Imported %d / %d subjects...", imported, len(common_ids))

        db.commit()
        logger.info("Done! Imported %d subjects, skipped %d (no image).", imported, skipped)

    except Exception as e:
        db.rollback()
        logger.error("Import failed: %s", e)
        raise
    finally:
        db.close()


def safe_float(val) -> float:
    """Convert to float, return None if invalid."""
    if val is None:
        return None
    try:
        return round(float(val), 2)
    except (ValueError, TypeError):
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import BodyM dataset")
    parser.add_argument("--limit", type=int, default=None, help="Max subjects to import")
    parser.add_argument("--skip-images", action="store_true", help="Skip downloading silhouette images")
    args = parser.parse_args()
    import_bodym(limit=args.limit, skip_images=args.skip_images)
