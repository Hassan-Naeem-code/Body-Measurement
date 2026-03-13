#!/usr/bin/env python3
"""
Import 3DPatBody dataset anthropometric measurements.

3DPatBody: 299 subjects with waist, hip, height, weight, BMI.
Source: https://ri.conicet.gov.ar/handle/11336/161809

NOTE: This dataset must be manually downloaded because the CONICET repository
blocks automated downloads. Steps:
  1. Visit https://ri.conicet.gov.ar/handle/11336/161809
  2. Download "anthropometric data" ZIP file (~211KB)
  3. Unzip it and place the CSV file at:
     backend/data/external/3dpatbody_anthropometric.csv
  4. Run: python -m scripts.import_3dpatbody

This dataset has no photos, so measurements are imported as "measurement-only"
ground truth (useful for validation ranges and statistical modeling even without images).

Run from the backend/ directory.
"""

import os
import sys
import csv
import logging
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.core.database import SessionLocal
from app.models.ground_truth import GroundTruth

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXTERNAL_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data", "external"
)
os.makedirs(EXTERNAL_DATA_DIR, exist_ok=True)


def safe_float(val) -> float:
    """Convert to float, return None if invalid."""
    if val is None or str(val).strip() == "":
        return None
    try:
        return round(float(str(val).replace(",", ".")), 2)
    except (ValueError, TypeError):
        return None


def find_csv():
    """Find the anthropometric CSV in the external data dir."""
    patterns = [
        os.path.join(EXTERNAL_DATA_DIR, "3dpatbody*.csv"),
        os.path.join(EXTERNAL_DATA_DIR, "anthropometric*.csv"),
        os.path.join(EXTERNAL_DATA_DIR, "*.csv"),
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None


def import_3dpatbody():
    """Import 3DPatBody anthropometric measurements."""

    csv_path = find_csv()
    if not csv_path:
        logger.error(
            "No CSV found in %s.\n"
            "Please download the 3DPatBody dataset manually:\n"
            "  1. Visit https://ri.conicet.gov.ar/handle/11336/161809\n"
            "  2. Download 'anthropometric data' ZIP\n"
            "  3. Unzip and place CSV in: %s/",
            EXTERNAL_DATA_DIR, EXTERNAL_DATA_DIR
        )
        return

    logger.info("Loading CSV from %s", csv_path)

    # Read and detect encoding / delimiter
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        sample = f.read(2000)

    delimiter = ";" if ";" in sample else ","

    with open(csv_path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        rows = list(reader)

    logger.info("Found %d rows with columns: %s", len(rows), list(rows[0].keys()) if rows else "none")

    db = SessionLocal()
    imported = 0

    # Try to map columns flexibly (3DPatBody uses Spanish/English mix)
    col_map = {}
    for row in rows[:1]:
        cols = {c.lower().strip(): c for c in row.keys()}
        # Height
        for key in ["height", "altura", "talla", "height (cm)", "height_cm"]:
            if key in cols:
                col_map["height"] = cols[key]
                break
        # Weight/mass
        for key in ["mass", "weight", "peso", "mass (kg)", "weight_kg", "mass_kg"]:
            if key in cols:
                col_map["weight"] = cols[key]
                break
        # Waist
        for key in ["waist", "cintura", "waist (cm)", "waist_cm"]:
            if key in cols:
                col_map["waist"] = cols[key]
                break
        # Hip
        for key in ["hip", "cadera", "hip (cm)", "hip_cm"]:
            if key in cols:
                col_map["hip"] = cols[key]
                break
        # Gender/sex
        for key in ["gender", "sex", "sexo", "género"]:
            if key in cols:
                col_map["gender"] = cols[key]
                break
        # Age
        for key in ["age", "edad"]:
            if key in cols:
                col_map["age"] = cols[key]
                break

    logger.info("Column mapping: %s", col_map)

    try:
        for row in rows:
            height = safe_float(row.get(col_map.get("height", "")))
            waist = safe_float(row.get(col_map.get("waist", "")))
            hip = safe_float(row.get(col_map.get("hip", "")))
            weight = safe_float(row.get(col_map.get("weight", "")))
            gender_raw = row.get(col_map.get("gender", ""), "").strip().lower()
            age = row.get(col_map.get("age", ""), "")

            # Need at least one body measurement
            if not any([waist, hip]):
                continue

            gender = None
            if gender_raw in ("m", "male", "masculino", "1"):
                gender = "male"
            elif gender_raw in ("f", "female", "femenino", "2"):
                gender = "female"
            else:
                gender = "other"

            gt = GroundTruth(
                image_path="no_image_3dpatbody",
                actual_chest_circumference=None,
                actual_waist_circumference=waist,
                actual_hip_circumference=hip,
                actual_shoulder_width=None,
                actual_inseam=None,
                actual_arm_length=None,
                actual_height=height,
                gender=gender,
                body_type=None,
                pose_type=None,
                measured_by="3dpatbody_dataset",
                notes=f"3DPatBody dataset. Weight: {weight}kg. Age: {age}.",
            )
            db.add(gt)
            imported += 1

        db.commit()
        logger.info("Done! Imported %d subjects from 3DPatBody.", imported)

    except Exception as e:
        db.rollback()
        logger.error("Import failed: %s", e)
        raise
    finally:
        db.close()


if __name__ == "__main__":
    import_3dpatbody()
