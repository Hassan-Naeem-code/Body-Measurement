"""
Debug script to see exactly what body parts are detected and which are missing
"""

import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.ml.pose_detector import PoseDetector
from app.ml.body_validator import FullBodyValidator

def debug_image(image_path: str):
    """Debug what's being detected in an image"""
    print(f"\n{'='*80}")
    print(f"DEBUGGING IMAGE: {image_path}")
    print(f"{'='*80}\n")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return

    print(f"✅ Image loaded: {image.shape[1]}x{image.shape[0]} pixels\n")

    # Detect pose
    detector = PoseDetector()
    result = detector.detect_pose(image)

    if not result.success:
        print(f"❌ Pose detection failed!")
        return

    print(f"✅ Pose detected successfully\n")

    # Validate body parts (use same thresholds as backend)
    from app.core.config import settings
    validator = FullBodyValidator(custom_thresholds={
        'head': settings.BODY_VALIDATION_HEAD_THRESHOLD,
        'shoulders': settings.BODY_VALIDATION_SHOULDERS_THRESHOLD,
        'elbows': settings.BODY_VALIDATION_ELBOWS_THRESHOLD,
        'hands': settings.BODY_VALIDATION_HANDS_THRESHOLD,
        'torso': settings.BODY_VALIDATION_TORSO_THRESHOLD,
        'legs': settings.BODY_VALIDATION_LEGS_THRESHOLD,
        'feet': settings.BODY_VALIDATION_FEET_THRESHOLD,
        'overall_min': settings.BODY_VALIDATION_OVERALL_MIN,
    })
    validation = validator.validate_full_body(result.pose_landmarks)

    print(f"{'='*80}")
    print(f"VALIDATION RESULT")
    print(f"{'='*80}\n")

    print(f"Overall Valid: {'✅ YES' if validation.is_valid else '❌ NO'}")
    print(f"Overall Confidence: {validation.overall_confidence:.2%} (threshold: {settings.BODY_VALIDATION_OVERALL_MIN:.2%})\n")

    if validation.missing_parts:
        print(f"❌ MISSING PARTS ({len(validation.missing_parts)}):")
        for part in validation.missing_parts:
            print(f"   • {part}")
        print()

    print(f"📊 BODY PART CONFIDENCE SCORES:")
    print(f"{'-'*80}")

    # Get threshold values from config (already imported above)
    thresholds = {
        'head': settings.BODY_VALIDATION_HEAD_THRESHOLD,
        'left_shoulder': settings.BODY_VALIDATION_SHOULDERS_THRESHOLD,
        'right_shoulder': settings.BODY_VALIDATION_SHOULDERS_THRESHOLD,
        'left_elbow': settings.BODY_VALIDATION_ELBOWS_THRESHOLD,
        'right_elbow': settings.BODY_VALIDATION_ELBOWS_THRESHOLD,
        'left_wrist': settings.BODY_VALIDATION_HANDS_THRESHOLD,
        'right_wrist': settings.BODY_VALIDATION_HANDS_THRESHOLD,
        'torso': settings.BODY_VALIDATION_TORSO_THRESHOLD,
        'left_hip': settings.BODY_VALIDATION_TORSO_THRESHOLD,
        'right_hip': settings.BODY_VALIDATION_TORSO_THRESHOLD,
        'left_knee': settings.BODY_VALIDATION_LEGS_THRESHOLD,
        'right_knee': settings.BODY_VALIDATION_LEGS_THRESHOLD,
        'left_ankle': settings.BODY_VALIDATION_FEET_THRESHOLD,
        'right_ankle': settings.BODY_VALIDATION_FEET_THRESHOLD,
    }

    # Print all confidence scores with status
    for part, score in sorted(validation.confidence_scores.items()):
        threshold = thresholds.get(part, 0.5)
        status = "✅" if score >= threshold else "❌"
        print(f"{status} {part:20s}: {score:.2f} (threshold: {threshold:.2f})")

    print(f"\n{'='*80}")
    print(f"VALIDATION DETAILS")
    print(f"{'='*80}\n")

    for key, value in validation.validation_details.items():
        print(f"{key}: {value}")

    # Check if human
    print(f"\n{'='*80}")
    print(f"HUMANITY CHECK")
    print(f"{'='*80}\n")

    is_human = validator.is_human(result.pose_landmarks)
    print(f"Is Human: {'✅ YES' if is_human else '❌ NO (might be mannequin/drawing/animal)'}")

    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python debug_validation.py <image_path>")
        print("\nExample:")
        print("  python debug_validation.py test_image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    debug_image(image_path)
