"""
MediaPipe Pose Detection System
Detects 33 body landmarks from a single image
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class PoseLandmarks:
    """Stores detected pose landmarks"""
    landmarks: list
    image_width: int
    image_height: int
    visibility_scores: Dict[str, float]


class PoseDetector:
    """
    Detects body pose using MediaPipe Pose
    Returns 33 3D landmarks with confidence scores
    """

    def __init__(self, static_image_mode=True, min_detection_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=2,  # Highest accuracy
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
        )

        # Key landmark indices
        self.LANDMARKS = {
            "LEFT_SHOULDER": 11,
            "RIGHT_SHOULDER": 12,
            "LEFT_HIP": 23,
            "RIGHT_HIP": 24,
            "LEFT_KNEE": 25,
            "RIGHT_KNEE": 26,
            "LEFT_ANKLE": 27,
            "RIGHT_ANKLE": 28,
            "LEFT_WRIST": 15,
            "RIGHT_WRIST": 16,
            "LEFT_ELBOW": 13,
            "RIGHT_ELBOW": 14,
            "NOSE": 0,
        }

    def detect(self, image_path: str) -> Optional[PoseLandmarks]:
        """
        Detect pose landmarks from an image file

        Args:
            image_path: Path to the image file

        Returns:
            PoseLandmarks object or None if detection fails
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        return self.detect_from_array(image)

    def detect_from_array(self, image: np.ndarray) -> Optional[PoseLandmarks]:
        """
        Detect pose landmarks from a numpy array (image)

        Args:
            image: BGR image as numpy array

        Returns:
            PoseLandmarks object or None if detection fails
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get image dimensions
        height, width, _ = image.shape

        # Process image
        results = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            return None

        # Extract landmarks
        landmarks = []
        visibility_scores = {}

        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            landmarks.append({
                "x": landmark.x * width,  # Convert normalized to pixel coordinates
                "y": landmark.y * height,
                "z": landmark.z,  # Depth (relative to hip)
                "visibility": landmark.visibility,
            })

            # Store visibility for key landmarks
            for name, lm_idx in self.LANDMARKS.items():
                if idx == lm_idx:
                    visibility_scores[name] = landmark.visibility

        return PoseLandmarks(
            landmarks=landmarks,
            image_width=width,
            image_height=height,
            visibility_scores=visibility_scores,
        )

    def get_landmark(self, pose_landmarks: PoseLandmarks, name: str) -> Dict:
        """Get specific landmark by name"""
        idx = self.LANDMARKS[name]
        return pose_landmarks.landmarks[idx]

    def calculate_distance(
        self, pose_landmarks: PoseLandmarks, point1_name: str, point2_name: str
    ) -> float:
        """
        Calculate Euclidean distance between two landmarks in pixels

        Args:
            pose_landmarks: Detected landmarks
            point1_name: Name of first landmark
            point2_name: Name of second landmark

        Returns:
            Distance in pixels
        """
        p1 = self.get_landmark(pose_landmarks, point1_name)
        p2 = self.get_landmark(pose_landmarks, point2_name)

        dx = p1["x"] - p2["x"]
        dy = p1["y"] - p2["y"]

        return np.sqrt(dx**2 + dy**2)

    def draw_landmarks(self, image: np.ndarray, pose_landmarks: PoseLandmarks) -> np.ndarray:
        """
        Draw pose landmarks on image for visualization

        Args:
            image: Input image
            pose_landmarks: Detected landmarks

        Returns:
            Image with drawn landmarks
        """
        # Convert landmarks back to MediaPipe format for drawing
        mp_landmarks = self.mp_pose.PoseLandmark

        # Create landmark list for drawing
        from mediapipe.framework.formats import landmark_pb2
        landmark_list = landmark_pb2.NormalizedLandmarkList()

        for lm in pose_landmarks.landmarks:
            landmark = landmark_list.landmark.add()
            landmark.x = lm["x"] / pose_landmarks.image_width
            landmark.y = lm["y"] / pose_landmarks.image_height
            landmark.z = lm["z"]
            landmark.visibility = lm["visibility"]

        # Draw on image
        annotated_image = image.copy()
        self.mp_drawing.draw_landmarks(
            annotated_image,
            landmark_list,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
        )

        return annotated_image

    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'pose'):
            self.pose.close()
