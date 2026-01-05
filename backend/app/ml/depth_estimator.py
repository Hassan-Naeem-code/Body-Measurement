"""
Depth Estimation using MiDaS for 3D body reconstruction
Provides depth map from a single 2D image
"""

import torch
import cv2
import numpy as np
from typing import Optional


class DepthEstimator:
    """
    Monocular depth estimation using MiDaS
    Converts 2D image to depth map for 3D reconstruction
    """

    def __init__(self, model_type: str = "DPT_Small"):
        """
        Initialize MiDaS depth estimator

        Args:
            model_type: MiDaS model variant
                - "DPT_Small": Fastest, good for real-time (256x256)
                - "DPT_Hybrid": Balanced speed/quality
                - "DPT_Large": Best quality, slower
        """
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load MiDaS model
        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.model.to(self.device)
        self.model.eval()

        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth map from RGB image

        Args:
            image: BGR image from OpenCV (H, W, 3)

        Returns:
            depth_map: Normalized depth map (H, W) - closer = higher values
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        input_batch = self.transform(image_rgb).to(self.device)

        # Inference
        with torch.no_grad():
            prediction = self.model(input_batch)

            # Resize to original image size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        # Normalize depth map to [0, 1] - closer objects have higher values
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

        return depth_map

    def create_3d_point_cloud(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Create 3D point cloud from RGB image and depth map

        Args:
            image: BGR image (H, W, 3)
            depth_map: Depth map (H, W)
            mask: Optional binary mask to filter points (H, W)

        Returns:
            points_3d: 3D points (N, 3) in camera coordinates [X, Y, Z]
        """
        height, width = image.shape[:2]

        # Create pixel grid
        u = np.arange(width)
        v = np.arange(height)
        uu, vv = np.meshgrid(u, v)

        # Camera intrinsics (approximate for standard camera)
        focal_length = max(width, height)  # Approximate
        cx = width / 2
        cy = height / 2

        # Back-project to 3D
        # X = (u - cx) * depth / focal_length
        # Y = (v - cy) * depth / focal_length
        # Z = depth

        X = (uu - cx) * depth_map / focal_length
        Y = (vv - cy) * depth_map / focal_length
        Z = depth_map

        # Stack to get 3D points
        points_3d = np.stack([X, Y, Z], axis=-1)  # (H, W, 3)

        # Apply mask if provided
        if mask is not None:
            points_3d = points_3d[mask > 0]  # (N, 3)
        else:
            points_3d = points_3d.reshape(-1, 3)  # (H*W, 3)

        return points_3d

    def estimate_depth_at_landmarks(
        self,
        depth_map: np.ndarray,
        landmarks: dict,
        image_width: int,
        image_height: int
    ) -> dict:
        """
        Get depth values at specific pose landmarks

        Args:
            depth_map: Depth map (H, W)
            landmarks: Dict of landmark name -> {x, y, visibility}
            image_width: Original image width
            image_height: Original image height

        Returns:
            landmark_depths: Dict of landmark name -> depth value
        """
        landmark_depths = {}

        for name, lm in landmarks.items():
            # Convert normalized coordinates to pixel coordinates
            x_px = int(lm["x"] * image_width)
            y_px = int(lm["y"] * image_height)

            # Clamp to image bounds
            x_px = max(0, min(x_px, depth_map.shape[1] - 1))
            y_px = max(0, min(y_px, depth_map.shape[0] - 1))

            # Get depth at landmark
            depth = depth_map[y_px, x_px]

            landmark_depths[name] = float(depth)

        return landmark_depths
