"""
Multi-Person Detection using YOLOv8
Detects all people in an image and returns bounding boxes
"""

from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class PersonBoundingBox:
    """Stores a detected person's bounding box"""
    x1: int  # Top-left x
    y1: int  # Top-left y
    x2: int  # Bottom-right x
    y2: int  # Bottom-right y
    confidence: float  # Detection confidence (0-1)
    person_id: int  # Index in image (0, 1, 2, ...)


class PersonDetector:
    """
    Detects multiple people in an image using YOLOv8
    Returns bounding boxes for cropping and pose detection
    """

    def __init__(self, model_size: str = "yolov8m.pt", confidence_threshold: float = 0.5):
        """
        Args:
            model_size: YOLOv8 model size (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
            confidence_threshold: Minimum confidence for person detection
        """
        self.model = YOLO(model_size)
        self.confidence_threshold = confidence_threshold
        self.person_class_id = 0  # COCO dataset: 0 = person

    def detect_people(self, image: np.ndarray) -> List[PersonBoundingBox]:
        """
        Detect all people in the image

        Args:
            image: BGR image as numpy array

        Returns:
            List of PersonBoundingBox objects, sorted by confidence (highest first)
        """
        # Run YOLOv8 inference
        results = self.model(image, verbose=False)

        people = []

        # Extract person detections
        for result in results:
            boxes = result.boxes

            for idx, box in enumerate(boxes):
                # Filter for person class only
                if int(box.cls) == self.person_class_id:
                    confidence = float(box.conf)

                    # Filter by confidence threshold
                    if confidence >= self.confidence_threshold:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                        people.append(PersonBoundingBox(
                            x1=int(x1),
                            y1=int(y1),
                            x2=int(x2),
                            y2=int(y2),
                            confidence=confidence,
                            person_id=len(people)
                        ))

        # Sort by confidence (highest first)
        people.sort(key=lambda p: p.confidence, reverse=True)

        # Re-assign person IDs after sorting
        for idx, person in enumerate(people):
            person.person_id = idx

        return people

    def crop_person(
        self,
        image: np.ndarray,
        bbox: PersonBoundingBox,
        padding_percent: float = 0.1
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Crop person from image with padding

        Args:
            image: Original image
            bbox: Person bounding box
            padding_percent: Padding around bbox (0.1 = 10% on each side)

        Returns:
            Tuple of (cropped_image, crop_metadata)
            crop_metadata contains offsets for coordinate translation
        """
        height, width = image.shape[:2]

        # Calculate padding
        bbox_width = bbox.x2 - bbox.x1
        bbox_height = bbox.y2 - bbox.y1

        pad_x = int(bbox_width * padding_percent)
        pad_y = int(bbox_height * padding_percent)

        # Apply padding with boundary checks
        x1_padded = max(0, bbox.x1 - pad_x)
        y1_padded = max(0, bbox.y1 - pad_y)
        x2_padded = min(width, bbox.x2 + pad_x)
        y2_padded = min(height, bbox.y2 + pad_y)

        # Crop image
        cropped = image[y1_padded:y2_padded, x1_padded:x2_padded]

        # Store metadata for coordinate translation
        metadata = {
            "offset_x": x1_padded,
            "offset_y": y1_padded,
            "original_width": width,
            "original_height": height,
            "crop_width": x2_padded - x1_padded,
            "crop_height": y2_padded - y1_padded
        }

        return cropped, metadata
