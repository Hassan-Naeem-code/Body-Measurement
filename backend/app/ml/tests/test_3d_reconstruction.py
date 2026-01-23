"""
Test suite for 3D Body Mesh Reconstruction

Tests the SMPL-based 3D reconstruction and mesh slicing for accurate
body circumference measurements.
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))


class TestMeshSlicer:
    """Test the mesh slicing algorithm"""

    def test_slice_simple_cylinder(self):
        """Test slicing a simple cylinder shape"""
        from app.ml.body_mesh_reconstructor import MeshSlicer

        slicer = MeshSlicer()

        # Create a simple cylinder mesh
        num_slices = 20
        num_radial = 32
        radius = 0.15  # 15cm radius
        height = 1.7  # 170cm height (in meters)

        vertices = []
        faces = []

        # Generate cylinder vertices
        for i in range(num_slices):
            y = i / (num_slices - 1) * height
            for j in range(num_radial):
                theta = 2 * np.pi * j / num_radial
                x = radius * np.cos(theta)
                z = radius * np.sin(theta)
                vertices.append([x, y, z])

        vertices = np.array(vertices)

        # Generate faces
        for i in range(num_slices - 1):
            for j in range(num_radial):
                base = i * num_radial + j
                next_j = (j + 1) % num_radial
                next_base = base + num_radial

                faces.append([base, next_base, base + next_j - j])
                faces.append([base + next_j - j, next_base, next_base + next_j - j])

        faces = np.array(faces)

        # Slice at middle (50% height)
        contour = slicer.slice_mesh_at_level(vertices, faces, 0.5)

        assert len(contour) > 0, "Should find intersection points"

        # Calculate circumference
        circumference = slicer.calculate_circumference(contour)

        # Expected circumference = 2 * pi * radius
        expected_circumference = 2 * np.pi * radius

        # Should be close to expected (within 10%)
        assert abs(circumference - expected_circumference) / expected_circumference < 0.15, \
            f"Circumference {circumference} should be close to {expected_circumference}"

    def test_measurement_levels(self):
        """Test that measurement levels are correctly defined"""
        from app.ml.body_mesh_reconstructor import MeshSlicer

        slicer = MeshSlicer()

        # Check measurement levels exist
        assert 'chest' in slicer.MEASUREMENT_LEVELS
        assert 'waist' in slicer.MEASUREMENT_LEVELS
        assert 'hip' in slicer.MEASUREMENT_LEVELS

        # Check values are reasonable (as fraction of height)
        assert 0.6 < slicer.MEASUREMENT_LEVELS['chest'] < 0.8
        assert 0.5 < slicer.MEASUREMENT_LEVELS['waist'] < 0.7
        assert 0.4 < slicer.MEASUREMENT_LEVELS['hip'] < 0.6


class TestHMRRegressor:
    """Test the Human Mesh Recovery regressor"""

    def test_regressor_forward_pass(self):
        """Test that HMR regressor produces valid output shapes"""
        import torch
        from app.ml.body_mesh_reconstructor import HMRRegressor

        regressor = HMRRegressor(num_keypoints=33)
        regressor.eval()

        # Create dummy keypoints
        batch_size = 2
        keypoints = torch.randn(batch_size, 33, 4)

        with torch.no_grad():
            betas, pose, orient = regressor(keypoints)

        # Check output shapes
        assert betas.shape == (batch_size, 10), f"Betas shape should be ({batch_size}, 10)"
        assert pose.shape == (batch_size, 72), f"Pose shape should be ({batch_size}, 72)"
        assert orient.shape == (batch_size, 3), f"Orient shape should be ({batch_size}, 3)"

    def test_regressor_deterministic(self):
        """Test that regressor gives same output for same input"""
        import torch
        from app.ml.body_mesh_reconstructor import HMRRegressor

        regressor = HMRRegressor()
        regressor.eval()

        keypoints = torch.randn(1, 33, 4)

        with torch.no_grad():
            betas1, _, _ = regressor(keypoints)
            betas2, _, _ = regressor(keypoints)

        assert torch.allclose(betas1, betas2), "Should produce same output for same input"


class TestBodyMeshReconstructor:
    """Test the full body mesh reconstructor"""

    def test_reconstructor_initialization(self):
        """Test that reconstructor initializes correctly"""
        from app.ml.body_mesh_reconstructor import BodyMeshReconstructor

        reconstructor = BodyMeshReconstructor(
            smpl_model_path=None,  # Will use fallback
            device='cpu',
            use_optimization=False
        )

        assert reconstructor.hmr_regressor is not None
        assert reconstructor.mesh_slicer is not None

    def test_keypoint_conversion(self):
        """Test MediaPipe keypoint conversion"""
        from app.ml.body_mesh_reconstructor import BodyMeshReconstructor
        from dataclasses import dataclass
        from typing import Dict, List

        # Create mock PoseLandmarks
        @dataclass
        class MockPoseLandmarks:
            landmarks: List[Dict]
            visibility_scores: Dict[str, float]
            image_width: int = 640
            image_height: int = 480

        # Create dummy landmarks
        landmarks = [{'x': i * 10, 'y': i * 15, 'z': 0.0} for i in range(33)]
        visibility = {f'LANDMARK_{i}': 0.9 for i in range(33)}

        pose_landmarks = MockPoseLandmarks(
            landmarks=landmarks,
            visibility_scores=visibility
        )

        reconstructor = BodyMeshReconstructor(device='cpu', use_optimization=False)
        keypoints = reconstructor.convert_mediapipe_to_array(pose_landmarks)

        assert keypoints.shape == (33, 4), f"Shape should be (33, 4), got {keypoints.shape}"

    def test_fallback_mesh_generation(self):
        """Test fallback mesh generation when SMPL not available"""
        from app.ml.body_mesh_reconstructor import BodyMeshReconstructor, SMPLParams
        import numpy as np

        reconstructor = BodyMeshReconstructor(
            smpl_model_path=None,
            device='cpu',
            use_optimization=False
        )

        # Create dummy keypoints
        keypoints = np.zeros((33, 4))
        # Set some key body points
        keypoints[0] = [0.5, 0.9, 0, 0.9]   # Nose
        keypoints[11] = [0.3, 0.75, 0, 0.9]  # Left shoulder
        keypoints[12] = [0.7, 0.75, 0, 0.9]  # Right shoulder
        keypoints[23] = [0.35, 0.5, 0, 0.9]  # Left hip
        keypoints[24] = [0.65, 0.5, 0, 0.9]  # Right hip
        keypoints[27] = [0.35, 0.1, 0, 0.9]  # Left ankle
        keypoints[28] = [0.65, 0.1, 0, 0.9]  # Right ankle

        params = SMPLParams(
            betas=np.zeros(10),
            body_pose=np.zeros(69),
            global_orient=np.zeros(3),
            transl=np.zeros(3),
            gender='neutral'
        )

        vertices, faces = reconstructor._generate_fallback_mesh(keypoints, params)

        assert len(vertices) > 0, "Should generate some vertices"
        assert vertices.shape[1] == 3, "Vertices should have 3 dimensions"


class TestCircumference3DExtractor:
    """Test the 3D circumference extractor"""

    def test_extractor_initialization(self):
        """Test that 3D extractor initializes"""
        from app.ml.circumference_extractor_3d import Circumference3DExtractor

        extractor = Circumference3DExtractor(
            smpl_model_path=None,
            use_gpu=False,
            fallback_to_2d=True
        )

        assert extractor.mesh_slicer is not None

    def test_pose_to_keypoints_array(self):
        """Test pose landmark conversion"""
        from app.ml.circumference_extractor_3d import Circumference3DExtractor
        from dataclasses import dataclass
        from typing import Dict, List

        @dataclass
        class MockPoseLandmarks:
            landmarks: List[Dict]
            visibility_scores: Dict[str, float]
            image_width: int = 640
            image_height: int = 480

        landmarks = [{'x': 320 + i, 'y': 240 + i, 'z': 0.0} for i in range(33)]
        visibility = {f'POINT_{i}': 0.8 for i in range(33)}

        pose = MockPoseLandmarks(landmarks=landmarks, visibility_scores=visibility)

        extractor = Circumference3DExtractor(use_gpu=False)
        keypoints = extractor._pose_to_keypoints_array(pose)

        assert keypoints.shape == (33, 4)
        # Check normalization
        assert 0 <= keypoints[0, 0] <= 1  # x should be normalized
        assert 0 <= keypoints[0, 1] <= 1  # y should be normalized


class TestIntegration:
    """Integration tests for the full pipeline"""

    def test_full_pipeline_with_mock_data(self):
        """Test the full measurement pipeline with mock data"""
        from app.ml.circumference_extractor_3d import Circumference3DExtractor
        from dataclasses import dataclass
        from typing import Dict, List
        import numpy as np

        @dataclass
        class MockPoseLandmarks:
            landmarks: List[Dict]
            visibility_scores: Dict[str, float]
            image_width: int = 640
            image_height: int = 480

        # Create realistic mock pose
        landmarks = []
        # Head
        landmarks.append({'x': 320, 'y': 50, 'z': 0})   # 0: Nose
        landmarks.append({'x': 310, 'y': 40, 'z': 0})   # 1: Left eye inner
        landmarks.append({'x': 315, 'y': 38, 'z': 0})   # 2: Left eye
        landmarks.append({'x': 300, 'y': 40, 'z': 0})   # 3: Left eye outer
        landmarks.append({'x': 330, 'y': 40, 'z': 0})   # 4: Right eye inner
        landmarks.append({'x': 325, 'y': 38, 'z': 0})   # 5: Right eye
        landmarks.append({'x': 340, 'y': 40, 'z': 0})   # 6: Right eye outer
        landmarks.append({'x': 295, 'y': 45, 'z': 0})   # 7: Left ear
        landmarks.append({'x': 345, 'y': 45, 'z': 0})   # 8: Right ear
        landmarks.append({'x': 310, 'y': 60, 'z': 0})   # 9: Mouth left
        landmarks.append({'x': 330, 'y': 60, 'z': 0})   # 10: Mouth right

        # Upper body
        landmarks.append({'x': 250, 'y': 120, 'z': 0})  # 11: Left shoulder
        landmarks.append({'x': 390, 'y': 120, 'z': 0})  # 12: Right shoulder
        landmarks.append({'x': 220, 'y': 200, 'z': 0})  # 13: Left elbow
        landmarks.append({'x': 420, 'y': 200, 'z': 0})  # 14: Right elbow
        landmarks.append({'x': 200, 'y': 280, 'z': 0})  # 15: Left wrist
        landmarks.append({'x': 440, 'y': 280, 'z': 0})  # 16: Right wrist
        landmarks.append({'x': 195, 'y': 295, 'z': 0})  # 17: Left pinky
        landmarks.append({'x': 445, 'y': 295, 'z': 0})  # 18: Right pinky
        landmarks.append({'x': 190, 'y': 290, 'z': 0})  # 19: Left index
        landmarks.append({'x': 450, 'y': 290, 'z': 0})  # 20: Right index
        landmarks.append({'x': 185, 'y': 285, 'z': 0})  # 21: Left thumb
        landmarks.append({'x': 455, 'y': 285, 'z': 0})  # 22: Right thumb

        # Lower body
        landmarks.append({'x': 280, 'y': 280, 'z': 0})  # 23: Left hip
        landmarks.append({'x': 360, 'y': 280, 'z': 0})  # 24: Right hip
        landmarks.append({'x': 275, 'y': 380, 'z': 0})  # 25: Left knee
        landmarks.append({'x': 365, 'y': 380, 'z': 0})  # 26: Right knee
        landmarks.append({'x': 270, 'y': 470, 'z': 0})  # 27: Left ankle
        landmarks.append({'x': 370, 'y': 470, 'z': 0})  # 28: Right ankle
        landmarks.append({'x': 265, 'y': 478, 'z': 0})  # 29: Left heel
        landmarks.append({'x': 375, 'y': 478, 'z': 0})  # 30: Right heel
        landmarks.append({'x': 260, 'y': 478, 'z': 0})  # 31: Left foot index
        landmarks.append({'x': 380, 'y': 478, 'z': 0})  # 32: Right foot index

        visibility = {
            'NOSE': 0.99, 'LEFT_EYE': 0.95, 'RIGHT_EYE': 0.95,
            'LEFT_SHOULDER': 0.98, 'RIGHT_SHOULDER': 0.98,
            'LEFT_ELBOW': 0.9, 'RIGHT_ELBOW': 0.9,
            'LEFT_WRIST': 0.85, 'RIGHT_WRIST': 0.85,
            'LEFT_HIP': 0.95, 'RIGHT_HIP': 0.95,
            'LEFT_KNEE': 0.9, 'RIGHT_KNEE': 0.9,
            'LEFT_ANKLE': 0.85, 'RIGHT_ANKLE': 0.85,
        }
        # Fill remaining visibility scores
        for i in range(33):
            key = f'LANDMARK_{i}'
            if key not in visibility:
                visibility[key] = 0.8

        pose = MockPoseLandmarks(landmarks=landmarks, visibility_scores=visibility)

        # Create mock image
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Initialize extractor
        extractor = Circumference3DExtractor(
            smpl_model_path=None,
            use_gpu=False,
            fallback_to_2d=False  # Use our 3D method
        )

        # Extract measurements
        measurements = extractor.extract_measurements(
            pose,
            mock_image,
            gender='neutral',
            known_height_cm=175.0
        )

        # Verify measurements are in reasonable ranges
        assert 60 < measurements.chest_circumference < 150, \
            f"Chest {measurements.chest_circumference} out of range"
        assert 50 < measurements.waist_circumference < 140, \
            f"Waist {measurements.waist_circumference} out of range"
        assert 70 < measurements.hip_circumference < 160, \
            f"Hip {measurements.hip_circumference} out of range"

        print(f"✓ Test passed!")
        print(f"  Chest circumference: {measurements.chest_circumference:.1f} cm")
        print(f"  Waist circumference: {measurements.waist_circumference:.1f} cm")
        print(f"  Hip circumference: {measurements.hip_circumference:.1f} cm")
        print(f"  Method: {measurements.confidence_scores.get('method', 'unknown')}")


def run_tests():
    """Run all tests"""
    print("=" * 60)
    print("Running 3D Body Reconstruction Tests")
    print("=" * 60)

    # Run pytest
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == '__main__':
    run_tests()
