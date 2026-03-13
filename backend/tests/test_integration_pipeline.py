"""
Integration tests for the ML measurement pipeline.

These tests run the actual ML models on test images to validate:
- Output structure matches expected schema
- Measurements are in reasonable ranges
- Error cases are handled properly

Mark: @pytest.mark.integration (requires ML models loaded, slower)
"""

import pytest
import numpy as np
import cv2
import os

# Skip all tests if ML dependencies are not available
pytestmark = pytest.mark.integration


def _create_synthetic_person_image(width=400, height=800):
    """
    Create a synthetic full-body-like image for testing.

    This is NOT a real person — it's a rough body-shaped silhouette that
    MediaPipe can sometimes detect. For real accuracy testing, use actual
    photos via the ground truth framework.
    """
    img = np.ones((height, width, 3), dtype=np.uint8) * 220  # light gray background

    # Draw a rough body shape (dark silhouette)
    body_color = (60, 60, 60)
    # Head
    cv2.circle(img, (200, 80), 40, body_color, -1)
    # Neck
    cv2.rectangle(img, (185, 120), (215, 150), body_color, -1)
    # Torso
    cv2.rectangle(img, (140, 150), (260, 400), body_color, -1)
    # Left arm
    cv2.rectangle(img, (100, 160), (140, 380), body_color, -1)
    # Right arm
    cv2.rectangle(img, (260, 160), (300, 380), body_color, -1)
    # Left leg
    cv2.rectangle(img, (150, 400), (195, 700), body_color, -1)
    # Right leg
    cv2.rectangle(img, (205, 400), (250, 700), body_color, -1)
    # Feet
    cv2.rectangle(img, (140, 700), (200, 730), body_color, -1)
    cv2.rectangle(img, (200, 700), (260, 730), body_color, -1)

    return img


def _create_headshot_image(width=400, height=400):
    """Create a headshot-only image (should fail full-body validation)."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 200
    cv2.circle(img, (200, 150), 80, (60, 60, 60), -1)  # Head
    cv2.rectangle(img, (150, 250), (250, 400), (60, 60, 60), -1)  # Shoulders
    return img


@pytest.fixture
def synthetic_person_image():
    return _create_synthetic_person_image()


@pytest.fixture
def headshot_image():
    return _create_headshot_image()


@pytest.fixture
def image_bytes(synthetic_person_image):
    """Encode image to JPEG bytes for API upload."""
    _, buf = cv2.imencode(".jpg", synthetic_person_image)
    return buf.tobytes()


@pytest.fixture
def headshot_bytes(headshot_image):
    _, buf = cv2.imencode(".jpg", headshot_image)
    return buf.tobytes()


class TestPoseDetector:
    """Test PoseDetector in isolation."""

    def test_detector_loads(self):
        from app.ml import PoseDetector
        detector = PoseDetector()
        assert detector is not None

    def test_detect_returns_none_for_blank_image(self):
        from app.ml import PoseDetector
        detector = PoseDetector()
        blank = np.zeros((400, 400, 3), dtype=np.uint8)
        result = detector.detect_from_array(blank)
        # Blank image should return None (no person)
        assert result is None


class TestMeasurementExtractor:
    """Test EnhancedMeasurementExtractor output structure."""

    def test_extractor_loads(self):
        from app.ml import EnhancedMeasurementExtractor
        extractor = EnhancedMeasurementExtractor(reference_height_cm=170.0)
        assert extractor is not None


class TestFullBodyValidator:
    """Test body validation logic."""

    def test_validator_loads(self):
        from app.ml import FullBodyValidator
        validator = FullBodyValidator()
        assert validator is not None


class TestEndToEndPipeline:
    """
    End-to-end pipeline test.

    Note: Synthetic images may not be detected by MediaPipe.
    These tests verify the pipeline handles both success and graceful failure.
    """

    def test_single_person_endpoint_invalid_file_type(self, client, auth_headers):
        """Test that non-image files are rejected."""
        response = client.post(
            "/api/v1/measurements/process",
            headers=auth_headers,
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400

    def test_multi_person_endpoint_invalid_file_type(self, client, auth_headers):
        """Test that non-image files are rejected for multi-person endpoint."""
        response = client.post(
            "/api/v1/measurements/process-multi",
            headers=auth_headers,
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400

    def test_single_person_with_image(self, client, auth_headers, image_bytes):
        """Test single person endpoint with a synthetic image."""
        response = client.post(
            "/api/v1/measurements/process",
            headers=auth_headers,
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )
        # Either processes successfully (200) or fails to detect (422)
        assert response.status_code in (200, 422)

        if response.status_code == 200:
            data = response.json()
            # Validate response structure
            assert "shoulder_width" in data
            assert "chest_width" in data
            assert "waist_width" in data
            assert "hip_width" in data
            assert "inseam" in data
            assert "arm_length" in data
            assert "confidence_scores" in data
            assert "recommended_size" in data

            # Validate reasonable measurement ranges (cm)
            assert 20 <= data["shoulder_width"] <= 70
            assert 50 <= data["chest_width"] <= 160
            assert 40 <= data["waist_width"] <= 150
            assert 50 <= data["hip_width"] <= 160
            assert 50 <= data["inseam"] <= 110
            assert 40 <= data["arm_length"] <= 100

    def test_multi_person_with_image(self, client, auth_headers, image_bytes):
        """Test multi-person endpoint with a synthetic image."""
        response = client.post(
            "/api/v1/measurements/process-multi",
            headers=auth_headers,
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )
        # Either processes (200) or no detection (422)
        assert response.status_code in (200, 422)

        if response.status_code == 200:
            data = response.json()
            assert "total_people_detected" in data
            assert "valid_people_count" in data
            assert "measurements" in data
            assert isinstance(data["measurements"], list)
            assert "processing_time_ms" in data

    def test_corrupt_image_rejected(self, client, auth_headers):
        """Test that corrupt image data returns an error."""
        response = client.post(
            "/api/v1/measurements/process",
            headers=auth_headers,
            files={"file": ("test.jpg", b"corrupt data", "image/jpeg")},
        )
        assert response.status_code in (400, 500)

    def test_no_api_key_rejected(self, client, image_bytes):
        """Test that requests without API key are rejected."""
        response = client.post(
            "/api/v1/measurements/process",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )
        assert response.status_code in (401, 403, 422)

    def test_invalid_fit_preference(self, client, auth_headers, image_bytes):
        """Test that invalid fit preference is rejected."""
        response = client.post(
            "/api/v1/measurements/process?fit_preference=invalid",
            headers=auth_headers,
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )
        assert response.status_code == 400


class TestGroundTruthEndpoints:
    """Test the ground truth validation API."""

    def test_create_ground_truth(self, client, auth_headers, image_bytes):
        """Test submitting a ground truth sample."""
        response = client.post(
            "/api/v1/validation/ground-truth"
            "?actual_chest_circumference=95.0"
            "&actual_waist_circumference=80.0"
            "&actual_height=175.0"
            "&gender=male"
            "&measured_by=test",
            headers=auth_headers,
            files={"file": ("person.jpg", image_bytes, "image/jpeg")},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["actual_chest_circumference"] == 95.0
        assert data["gender"] == "male"
        assert "id" in data

    def test_list_ground_truth(self, client, auth_headers):
        """Test listing ground truth samples."""
        response = client.get(
            "/api/v1/validation/ground-truth",
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_create_ground_truth_no_measurements(self, client, auth_headers, image_bytes):
        """Test that submitting without any measurements fails."""
        response = client.post(
            "/api/v1/validation/ground-truth",
            headers=auth_headers,
            files={"file": ("person.jpg", image_bytes, "image/jpeg")},
        )
        assert response.status_code == 400
