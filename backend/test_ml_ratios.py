"""
Test script to compare ML-based ratios vs Fixed ratios
Demonstrates the improvement in personalization
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.ml.pose_detector import PoseDetector
from app.ml.circumference_extractor_simple import SimpleCircumferenceExtractor
from app.ml.circumference_extractor_ml import MLCircumferenceExtractor
from app.ml.depth_ratio_predictor import MLDepthRatioPredictor


def test_on_image(image_path: str):
    """Test both extractors on an image and compare results"""
    print(f"\n{'='*80}")
    print(f"Testing on: {image_path}")
    print(f"{'='*80}\n")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    # Initialize detectors
    pose_detector = PoseDetector()
    fixed_extractor = SimpleCircumferenceExtractor()
    ml_extractor = MLCircumferenceExtractor(use_ml_ratios=True)
    ratio_predictor = MLDepthRatioPredictor()

    # Detect pose
    pose_result = pose_detector.detect_pose(image)
    if not pose_result.success:
        print("Error: Could not detect pose")
        return

    # Get ML ratio analysis
    print("🧠 ML RATIO ANALYSIS")
    print("-" * 80)
    stats = ratio_predictor.get_comparison_stats(pose_result.pose_landmarks)

    features = stats['features']
    print(f"Body Characteristics:")
    print(f"  • Gender Estimate: {features.estimated_gender.upper()}")
    print(f"  • Body Shape: {features.body_shape_category.replace('_', ' ').title()}")
    print(f"  • BMI Estimate: {features.bmi_estimate:.1f}")
    print(f"  • Shoulder/Hip Ratio: {features.shoulder_to_hip_ratio:.2f}")
    print(f"  • Chest/Waist Taper: {features.chest_to_waist_ratio:.2f}")
    print(f"  • Pose Angle: {features.pose_angle:.1f}°")
    print(f"  • Confidence: {features.landmark_confidence:.2f}")

    print(f"\n📊 DEPTH RATIO COMPARISON")
    print("-" * 80)
    ml_ratios = stats['ml_ratios']
    fixed_ratios = stats['fixed_ratios']
    diffs = stats['differences']

    print(f"{'Measurement':<20} {'Fixed':>10} {'ML-Based':>10} {'Difference':>12} {'Change':>10}")
    print("-" * 80)
    print(f"{'Chest Ratio':<20} {fixed_ratios.chest_ratio:>10.3f} {ml_ratios.chest_ratio:>10.3f} "
          f"{diffs['chest']:>+12.3f} {(diffs['chest']/fixed_ratios.chest_ratio*100):>+9.1f}%")
    print(f"{'Waist Ratio':<20} {fixed_ratios.waist_ratio:>10.3f} {ml_ratios.waist_ratio:>10.3f} "
          f"{diffs['waist']:>+12.3f} {(diffs['waist']/fixed_ratios.waist_ratio*100):>+9.1f}%")
    print(f"{'Hip Ratio':<20} {fixed_ratios.hip_ratio:>10.3f} {ml_ratios.hip_ratio:>10.3f} "
          f"{diffs['hip']:>+12.3f} {(diffs['hip']/fixed_ratios.hip_ratio*100):>+9.1f}%")

    # Extract measurements with both methods
    print(f"\n📏 CIRCUMFERENCE MEASUREMENTS COMPARISON")
    print("-" * 80)

    fixed_measurements = fixed_extractor.extract_measurements(pose_result.pose_landmarks, image)
    ml_measurements = ml_extractor.extract_measurements(pose_result.pose_landmarks, image)

    print(f"{'Measurement':<20} {'Fixed (cm)':>12} {'ML-Based (cm)':>15} {'Difference':>12}")
    print("-" * 80)

    measurements = [
        ('Chest Circumference', 'chest_circumference'),
        ('Waist Circumference', 'waist_circumference'),
        ('Hip Circumference', 'hip_circumference'),
        ('Shoulder Width', 'shoulder_width'),
    ]

    for name, attr in measurements:
        fixed_val = getattr(fixed_measurements, attr)
        ml_val = getattr(ml_measurements, attr)
        diff = ml_val - fixed_val
        print(f"{name:<20} {fixed_val:>12.1f} {ml_val:>15.1f} {diff:>+12.1f} cm")

    # Interpret results
    print(f"\n💡 INTERPRETATION")
    print("-" * 80)

    if abs(diffs['chest']) > 0.05 or abs(diffs['waist']) > 0.05:
        print("✅ ML model detected significant body type differences!")
        print("   The personalized ratios will provide more accurate measurements.")

        if features.body_shape_category == 'inverted_triangle':
            print("   👔 Athletic/V-shape build detected - increased chest depth ratio")
        elif features.body_shape_category == 'triangle':
            print("   🍐 Pear shape detected - increased hip depth ratio")
        elif features.body_shape_category == 'hourglass':
            print("   ⏳ Hourglass shape detected - balanced with deeper waist")

        if features.bmi_estimate > 27:
            print(f"   📈 Higher BMI ({features.bmi_estimate:.1f}) - all depth ratios increased")
        elif features.bmi_estimate < 20:
            print(f"   📉 Lower BMI ({features.bmi_estimate:.1f}) - athletic build with selective ratios")
    else:
        print("ℹ️  Body type close to average - ML ratios similar to fixed ratios")
        print("   Both methods will give comparable results for this person.")

    print(f"\n🎯 CONFIDENCE & METHOD")
    print("-" * 80)
    print(f"ML Prediction Confidence: {ml_ratios.confidence:.1%}")
    print(f"Method Used: {ml_ratios.method}")

    if ml_ratios.confidence < 0.7:
        print("⚠️  Warning: Lower confidence - consider retaking photo with better pose/lighting")

    print()


def test_synthetic_body_types():
    """Test on synthetic/simulated body types to demonstrate adaptation"""
    print("\n" + "="*80)
    print("SYNTHETIC BODY TYPE TESTING")
    print("="*80)
    print("Demonstrating how ML ratios adapt to different body types\n")

    ratio_predictor = MLDepthRatioPredictor()

    # Simulate different body types with features
    test_cases = [
        {
            'name': 'Average Male (Athletic)',
            'gender': 'male',
            'shoulder_hip': 1.25,
            'chest_waist': 1.35,
            'bmi': 24,
        },
        {
            'name': 'Average Female (Hourglass)',
            'gender': 'female',
            'shoulder_hip': 0.92,
            'chest_waist': 1.22,
            'bmi': 22,
        },
        {
            'name': 'Male Bodybuilder',
            'gender': 'male',
            'shoulder_hip': 1.45,
            'chest_waist': 1.50,
            'bmi': 28,
        },
        {
            'name': 'Female Pear Shape',
            'gender': 'female',
            'shoulder_hip': 0.85,
            'chest_waist': 1.10,
            'bmi': 23,
        },
        {
            'name': 'Overweight Male',
            'gender': 'male',
            'shoulder_hip': 1.10,
            'chest_waist': 1.05,
            'bmi': 32,
        },
        {
            'name': 'Thin/Slender Build',
            'gender': 'neutral',
            'shoulder_hip': 1.05,
            'chest_waist': 1.15,
            'bmi': 18,
        },
    ]

    from app.ml.depth_ratio_predictor import BodyFeatures

    for case in test_cases:
        # Create synthetic features
        features = BodyFeatures(
            shoulder_to_hip_ratio=case['shoulder_hip'],
            torso_length_ratio=0.52,
            bmi_estimate=case['bmi'],
            shoulder_width_normalized=0.20,
            hip_width_normalized=0.20 / case['shoulder_hip'],
            waist_width_normalized=0.18,
            chest_to_waist_ratio=case['chest_waist'],
            upper_body_mass_indicator=0.25,
            lower_body_mass_indicator=1.0 / case['shoulder_hip'],
            body_shape_category=ratio_predictor._classify_body_shape(
                case['shoulder_hip'], case['chest_waist']
            ),
            estimated_gender=case['gender'],
            pose_angle=0.0,
            landmark_confidence=0.95
        )

        ratios = ratio_predictor.predict_ratios(features)

        print(f"\n{case['name']}:")
        print(f"  Shape: {features.body_shape_category.replace('_', ' ').title()}")
        print(f"  Ratios → Chest: {ratios.chest_ratio:.3f}, "
              f"Waist: {ratios.waist_ratio:.3f}, Hip: {ratios.hip_ratio:.3f}")

        # Compare to fixed
        chest_diff = ((ratios.chest_ratio - 0.62) / 0.62) * 100
        waist_diff = ((ratios.waist_ratio - 0.58) / 0.58) * 100
        hip_diff = ((ratios.hip_ratio - 0.55) / 0.55) * 100

        print(f"  vs Fixed → Chest: {chest_diff:+.1f}%, "
              f"Waist: {waist_diff:+.1f}%, Hip: {hip_diff:+.1f}%")


if __name__ == "__main__":
    print("\n🧪 ML DEPTH RATIO PREDICTOR - TEST SUITE")
    print("="*80)

    # Test synthetic body types first
    test_synthetic_body_types()

    # Test on real images if provided
    if len(sys.argv) > 1:
        for image_path in sys.argv[1:]:
            try:
                test_on_image(image_path)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("\n" + "="*80)
        print("ℹ️  To test on real images, run:")
        print("   python test_ml_ratios.py path/to/image1.jpg path/to/image2.jpg")
        print("="*80)

    print("\n✅ Testing complete!")
