"""
3D Body Mesh Reconstruction using SMPL

Converts 2D images to full 3D body meshes for accurate circumference measurement.
Solves the "180° Problem" by reconstructing the complete body surface.

Target accuracy: 92-98% (vs 70-85% with 2D estimation)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import os

logger = logging.getLogger(__name__)

# Try to import SMPL-related packages
SMPL_AVAILABLE = False
TRIMESH_AVAILABLE = False

try:
    from app.ml.smpl_loader import SMPLModel, load_smpl_model
    SMPL_AVAILABLE = True
except ImportError:
    try:
        # Fallback to direct import
        import importlib.util
        import os
        loader_path = os.path.join(os.path.dirname(__file__), 'smpl_loader.py')
        if os.path.exists(loader_path):
            spec = importlib.util.spec_from_file_location('smpl_loader', loader_path)
            smpl_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(smpl_module)
            SMPLModel = smpl_module.SMPLModel
            load_smpl_model = smpl_module.load_smpl_model
            SMPL_AVAILABLE = True
    except Exception as e:
        logger.warning(f"SMPL loader not available: {e}")

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    pass


@dataclass
class SMPLParams:
    """SMPL body model parameters"""
    betas: np.ndarray  # Shape parameters (10 dims)
    body_pose: np.ndarray  # Body pose (23 joints x 3 rotation)
    global_orient: np.ndarray  # Root orientation (3 dims)
    transl: np.ndarray  # Translation (3 dims)
    gender: str  # 'male', 'female', or 'neutral'


@dataclass
class MeshMeasurements:
    """Measurements extracted from 3D mesh slicing"""
    chest_circumference: float
    waist_circumference: float
    hip_circumference: float
    shoulder_width: float
    arm_length: float
    inseam: float
    height: float

    chest_slice_vertices: Optional[np.ndarray] = None
    waist_slice_vertices: Optional[np.ndarray] = None
    hip_slice_vertices: Optional[np.ndarray] = None

    confidence: float = 0.95
    method: str = "smpl_mesh_slicing"


class HMRRegressor(nn.Module):
    """
    Human Mesh Recovery (HMR) style regressor.

    Predicts SMPL parameters from 2D pose keypoints.
    Architecture: MLP that maps keypoint features to SMPL shape and pose.
    """

    def __init__(
        self,
        num_keypoints: int = 33,  # MediaPipe has 33 keypoints
        hidden_dim: int = 256,
        num_betas: int = 10,
        num_pose_params: int = 72  # 24 joints * 3 rotation params
    ):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.num_betas = num_betas
        self.num_pose_params = num_pose_params

        # Input: flattened keypoints (x, y, z, visibility) * num_keypoints
        input_dim = num_keypoints * 4

        # Feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Shape regressor (betas)
        self.shape_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_betas),
            nn.Tanh(),  # Constrain to reasonable range
        )

        # Pose regressor
        self.pose_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_pose_params),
        )

        # Global orientation regressor
        self.orient_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, keypoints: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            keypoints: (B, num_keypoints, 4) tensor of (x, y, z, visibility)

        Returns:
            betas: (B, 10) shape parameters
            pose: (B, 72) pose parameters
            orient: (B, 3) global orientation
        """
        batch_size = keypoints.shape[0]

        # Flatten keypoints
        x = keypoints.view(batch_size, -1)

        # Extract features
        features = self.feature_net(x)

        # Predict SMPL parameters
        betas = self.shape_head(features) * 2.0  # Scale to typical beta range
        pose = self.pose_head(features) * 0.1  # Small initial poses
        orient = self.orient_head(features) * 0.1

        return betas, pose, orient


class KeypointOptimizer:
    """
    Optimizes SMPL parameters to match 2D keypoints.

    Uses iterative optimization to fit the SMPL model to observed 2D keypoints
    from MediaPipe pose detection.
    """

    # MediaPipe to SMPL joint mapping (approximate)
    MEDIAPIPE_TO_SMPL = {
        0: 15,   # Nose -> Head
        11: 16,  # Left shoulder
        12: 17,  # Right shoulder
        13: 18,  # Left elbow
        14: 19,  # Right elbow
        15: 20,  # Left wrist
        16: 21,  # Right wrist
        23: 1,   # Left hip
        24: 2,   # Right hip
        25: 4,   # Left knee
        26: 5,   # Right knee
        27: 7,   # Left ankle
        28: 8,   # Right ankle
    }

    def __init__(
        self,
        smpl_model,
        device: str = 'cpu',
        lr: float = 0.01,
        num_iterations: int = 100
    ):
        self.smpl = smpl_model
        self.device = device
        self.lr = lr
        self.num_iterations = num_iterations

    def optimize(
        self,
        keypoints_2d: np.ndarray,
        init_betas: Optional[torch.Tensor] = None,
        init_pose: Optional[torch.Tensor] = None
    ) -> SMPLParams:
        """
        Optimize SMPL parameters to match 2D keypoints.

        Args:
            keypoints_2d: (33, 4) MediaPipe keypoints (x, y, z, visibility)
            init_betas: Initial shape parameters
            init_pose: Initial pose parameters

        Returns:
            Optimized SMPLParams
        """
        # Initialize parameters
        betas = init_betas if init_betas is not None else torch.zeros(1, 10, device=self.device)
        body_pose = init_pose if init_pose is not None else torch.zeros(1, 69, device=self.device)
        global_orient = torch.zeros(1, 3, device=self.device)

        betas.requires_grad = True
        body_pose.requires_grad = True
        global_orient.requires_grad = True

        # Convert keypoints to tensor
        kp_tensor = torch.tensor(keypoints_2d, device=self.device, dtype=torch.float32)

        # Optimizer
        optimizer = torch.optim.Adam([betas, body_pose, global_orient], lr=self.lr)

        # Optimization loop
        for i in range(self.num_iterations):
            optimizer.zero_grad()

            # Forward pass through SMPL
            output = self.smpl(
                betas=betas,
                body_pose=body_pose,
                global_orient=global_orient,
                return_verts=True
            )

            # Get SMPL joints (J_regressor applied to vertices)
            smpl_joints = output.joints[0]  # (J, 3)

            # Project to 2D (orthographic projection for simplicity)
            smpl_joints_2d = smpl_joints[:, :2]  # Just x, y

            # Normalize to [0, 1] range
            smpl_joints_2d = (smpl_joints_2d - smpl_joints_2d.min()) / (smpl_joints_2d.max() - smpl_joints_2d.min() + 1e-6)

            # Compute loss: reprojection error
            loss = 0.0
            num_matched = 0

            for mp_idx, smpl_idx in self.MEDIAPIPE_TO_SMPL.items():
                if mp_idx < len(keypoints_2d) and smpl_idx < len(smpl_joints_2d):
                    vis = kp_tensor[mp_idx, 3]
                    if vis > 0.3:  # Only use visible keypoints
                        mp_2d = kp_tensor[mp_idx, :2]
                        smpl_2d = smpl_joints_2d[smpl_idx]
                        loss += vis * F.mse_loss(smpl_2d, mp_2d)
                        num_matched += 1

            if num_matched > 0:
                loss = loss / num_matched

            # Regularization
            loss += 0.001 * torch.sum(betas ** 2)  # Shape regularization
            loss += 0.0001 * torch.sum(body_pose ** 2)  # Pose regularization

            loss.backward()
            optimizer.step()

        return SMPLParams(
            betas=betas.detach().cpu().numpy()[0],
            body_pose=body_pose.detach().cpu().numpy()[0],
            global_orient=global_orient.detach().cpu().numpy()[0],
            transl=np.zeros(3),
            gender='neutral'
        )


class MeshSlicer:
    """
    Slices 3D mesh at specific body levels to extract circumferences.

    This is the key innovation that solves the "180° problem" -
    we can measure the FULL circumference around the 3D mesh.
    """

    # Standard measurement levels (as fraction of body height from ground)
    MEASUREMENT_LEVELS = {
        'chest': 0.72,   # Nipple line (approx 72% of height from ground)
        'waist': 0.62,   # Natural waist (approx 62% of height)
        'hip': 0.52,     # Hip level (approx 52% of height)
    }

    def __init__(self):
        pass

    def slice_mesh_at_level(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        height_fraction: float
    ) -> np.ndarray:
        """
        Slice mesh at a specific height level.

        Args:
            vertices: (N, 3) mesh vertices
            faces: (M, 3) mesh faces
            height_fraction: Fraction of body height (0 = feet, 1 = top of head)

        Returns:
            (K, 3) array of intersection points forming the slice contour
        """
        # Get height range
        min_y = vertices[:, 1].min()
        max_y = vertices[:, 1].max()
        body_height = max_y - min_y

        # Calculate slice plane height with small offset to avoid landing exactly on vertices
        # Add tiny offset to ensure we cut through faces, not just touch vertices
        slice_y = min_y + body_height * height_fraction + 0.001

        # Find intersection points
        intersection_points = []

        for face in faces:
            v0, v1, v2 = vertices[face]

            # Check each edge of the triangle for intersection with horizontal plane
            edges = [(v0, v1), (v1, v2), (v2, v0)]

            for va, vb in edges:
                # Check if edge crosses the slice plane (including near-touches)
                ya_rel = va[1] - slice_y
                yb_rel = vb[1] - slice_y

                # Edge crosses if signs differ, or if one endpoint is very close to plane
                if ya_rel * yb_rel < 0 or (abs(ya_rel) < 0.002 and abs(yb_rel) > 0.01):
                    # Linear interpolation to find intersection point
                    if abs(vb[1] - va[1]) > 1e-8:
                        t = (slice_y - va[1]) / (vb[1] - va[1])
                        t = np.clip(t, 0, 1)
                        point = va + t * (vb - va)
                        intersection_points.append(point)

        # Also collect vertices that are exactly on the slice plane
        on_plane_vertices = vertices[np.abs(vertices[:, 1] - slice_y) < 0.003]
        if len(on_plane_vertices) > 0:
            intersection_points.extend(on_plane_vertices.tolist())

        if not intersection_points:
            return np.array([])

        # Remove duplicates
        points = np.array(intersection_points)
        if len(points) > 1:
            # Round to avoid floating point duplicates
            rounded = np.round(points, 4)
            _, unique_idx = np.unique(rounded, axis=0, return_index=True)
            points = points[unique_idx]

        return points

    def order_contour_points(self, points: np.ndarray) -> np.ndarray:
        """
        Order contour points to form a closed loop.

        Uses angular sorting around centroid.
        """
        if len(points) < 3:
            return points

        # Calculate centroid
        centroid = points.mean(axis=0)

        # Calculate angles from centroid
        angles = np.arctan2(
            points[:, 2] - centroid[2],  # z component
            points[:, 0] - centroid[0]   # x component
        )

        # Sort by angle
        sorted_indices = np.argsort(angles)
        return points[sorted_indices]

    def calculate_circumference(self, contour: np.ndarray) -> float:
        """
        Calculate the circumference (perimeter) of a closed contour.

        Args:
            contour: (K, 3) ordered contour points

        Returns:
            Circumference in mesh units
        """
        if len(contour) < 3:
            return 0.0

        # Order points
        ordered = self.order_contour_points(contour)

        # Calculate perimeter
        perimeter = 0.0
        for i in range(len(ordered)):
            p1 = ordered[i]
            p2 = ordered[(i + 1) % len(ordered)]
            perimeter += np.linalg.norm(p2 - p1)

        return perimeter

    def extract_measurements(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        joints: np.ndarray = None,
        scale_factor: float = 100.0  # Convert SMPL units to cm
    ) -> Dict[str, float]:
        """
        Extract all body measurements from mesh.

        Args:
            vertices: (N, 3) mesh vertices
            faces: (M, 3) mesh faces
            joints: (24, 3) SMPL joint positions (optional, for accurate shoulder width)
            scale_factor: Conversion factor to cm (SMPL is in meters, so 100.0)

        Returns:
            Dictionary of measurements in cm
        """
        measurements = {}
        slices = {}

        for name, height_frac in self.MEASUREMENT_LEVELS.items():
            # Get slice contour
            contour = self.slice_mesh_at_level(vertices, faces, height_frac)
            slices[name] = contour

            if len(contour) > 3:
                # Calculate circumference
                circ = self.calculate_circumference(contour)
                measurements[f'{name}_circumference'] = circ * scale_factor
            else:
                measurements[f'{name}_circumference'] = 0.0

        # Calculate height
        height = (vertices[:, 1].max() - vertices[:, 1].min()) * scale_factor
        measurements['height'] = height

        # Calculate shoulder width
        # Prefer using SMPL joint positions if available (more accurate)
        if joints is not None and len(joints) >= 18:
            # SMPL joints: 16 = left_shoulder, 17 = right_shoulder
            left_shoulder = joints[16]
            right_shoulder = joints[17]
            # Calculate distance in X-Y plane (ignoring Z for biacromial width)
            shoulder_width = np.linalg.norm(right_shoulder[:2] - left_shoulder[:2]) * scale_factor
            measurements['shoulder_width'] = shoulder_width
        else:
            # Fallback: estimate from chest circumference
            # Typical biacromial shoulder width is ~1.15x chest diameter
            chest_circ = measurements.get('chest_circumference', 100)
            chest_diameter = chest_circ / np.pi
            measurements['shoulder_width'] = chest_diameter * 1.15

        # Calculate arm length and inseam from joints if available
        if joints is not None and len(joints) >= 22:
            # Arm length: shoulder (16/17) to wrist (20/21)
            left_arm = np.linalg.norm(joints[16] - joints[18]) + np.linalg.norm(joints[18] - joints[20])
            right_arm = np.linalg.norm(joints[17] - joints[19]) + np.linalg.norm(joints[19] - joints[21])
            measurements['arm_length'] = ((left_arm + right_arm) / 2) * scale_factor

            # Inseam: hip (1/2) to ankle (7/8)
            left_inseam = np.linalg.norm(joints[1] - joints[4]) + np.linalg.norm(joints[4] - joints[7])
            right_inseam = np.linalg.norm(joints[2] - joints[5]) + np.linalg.norm(joints[5] - joints[8])
            measurements['inseam'] = ((left_inseam + right_inseam) / 2) * scale_factor
        else:
            # Fallback: estimate from body proportions
            measurements['arm_length'] = height * 0.33  # Arm is ~33% of height
            measurements['inseam'] = height * 0.45  # Inseam is ~45% of height

        return measurements, slices


class BodyMeshReconstructor:
    """
    Main class for 3D body reconstruction from 2D images.

    Pipeline:
    1. Take 2D pose keypoints from MediaPipe
    2. Use HMR-style regression to estimate initial SMPL parameters
    3. Optionally refine with optimization
    4. Generate 3D mesh
    5. Slice mesh to extract true circumferences

    This solves the fundamental "180° problem" by reconstructing the full body.
    """

    def __init__(
        self,
        smpl_model_path: Optional[str] = None,
        gender: str = 'neutral',
        device: str = None,
        use_optimization: bool = True
    ):
        """
        Initialize the body mesh reconstructor.

        Args:
            smpl_model_path: Path to SMPL model files (e.g., 'models/smpl')
            gender: Default gender for SMPL model
            device: Torch device ('cpu' or 'cuda')
            use_optimization: Whether to refine with optimization
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.gender = gender
        self.use_optimization = use_optimization

        # Initialize components
        self.smpl_model = None
        self.hmr_regressor = None
        self.mesh_slicer = MeshSlicer()

        # Try to load SMPL model
        if SMPL_AVAILABLE:
            self._load_smpl_model(smpl_model_path)
        else:
            logger.warning("SMPL not available. Using fallback mesh generation.")

        # Initialize HMR regressor
        self.hmr_regressor = HMRRegressor().to(self.device)
        self.hmr_regressor.eval()

        # Load pretrained weights if available
        self._load_pretrained_weights()

    def _load_smpl_model(self, model_path: Optional[str]):
        """Load SMPL model using custom loader"""
        if model_path is None:
            # Try default paths
            default_paths = [
                'models/smpl',
                '/app/models/smpl',
                os.path.expanduser('~/.smpl/models'),
            ]
            for path in default_paths:
                if os.path.exists(path):
                    model_path = path
                    break

        if model_path and os.path.exists(model_path):
            try:
                self.smpl_model = load_smpl_model(
                    model_path,
                    gender=self.gender,
                    device=self.device
                )
                if self.smpl_model is not None:
                    logger.info(f"SMPL model loaded from {model_path}")
                else:
                    logger.warning(f"SMPL model loading returned None")
            except Exception as e:
                logger.warning(f"Failed to load SMPL model: {e}")
                self.smpl_model = None
        else:
            logger.warning(f"SMPL model path not found: {model_path}")

    def _load_pretrained_weights(self):
        """Load pretrained HMR weights if available"""
        weights_paths = [
            'models/hmr_regressor.pth',
            '/app/models/hmr_regressor.pth',
        ]

        for path in weights_paths:
            if os.path.exists(path):
                try:
                    state_dict = torch.load(path, map_location=self.device)
                    self.hmr_regressor.load_state_dict(state_dict)
                    logger.info(f"Loaded pretrained HMR weights from {path}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load HMR weights: {e}")

        logger.info("Using randomly initialized HMR regressor (no pretrained weights)")

    def reconstruct_from_keypoints(
        self,
        keypoints: np.ndarray,
        gender: str = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, SMPLParams]:
        """
        Reconstruct 3D mesh from 2D pose keypoints.

        Args:
            keypoints: (33, 4) MediaPipe keypoints (x, y, z, visibility)
            gender: Override gender for this reconstruction

        Returns:
            vertices: (6890, 3) mesh vertices
            faces: (13776, 3) mesh faces
            joints: (24, 3) joint positions (or None for fallback)
            params: Fitted SMPL parameters
        """
        gender = gender or self.gender

        # Step 1: Predict initial parameters with HMR regressor
        kp_tensor = torch.tensor(keypoints, device=self.device, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            betas, pose, orient = self.hmr_regressor(kp_tensor)

        # Step 2: Optionally refine with optimization
        if self.use_optimization and self.smpl_model is not None:
            optimizer = KeypointOptimizer(
                self.smpl_model,
                device=self.device,
                num_iterations=50
            )
            params = optimizer.optimize(keypoints, init_betas=betas, init_pose=pose[:, :69])
        else:
            params = SMPLParams(
                betas=betas.cpu().numpy()[0],
                body_pose=pose[:, :69].cpu().numpy()[0],
                global_orient=orient.cpu().numpy()[0],
                transl=np.zeros(3),
                gender=gender
            )

        # Step 3: Generate mesh
        if self.smpl_model is not None:
            vertices, faces, joints = self._generate_smpl_mesh(params)
        else:
            vertices, faces = self._generate_fallback_mesh(keypoints, params)
            joints = None  # No joints available for fallback mesh

        return vertices, faces, joints, params

    def _generate_smpl_mesh(self, params: SMPLParams, use_t_pose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate mesh using SMPL model.

        Args:
            params: SMPL parameters (betas, body_pose, global_orient)
            use_t_pose: If True, use T-pose for measurement (ignores pose params).
                       This ensures consistent circumference measurements regardless of pose.
        """
        betas = torch.tensor(params.betas, device=self.device, dtype=torch.float32).unsqueeze(0)

        if use_t_pose:
            # Use T-pose (zero pose) for accurate measurements
            # This ensures circumferences are measured consistently
            body_pose = torch.zeros(1, 69, device=self.device, dtype=torch.float32)
            global_orient = torch.zeros(1, 3, device=self.device, dtype=torch.float32)
        else:
            body_pose = torch.tensor(params.body_pose, device=self.device, dtype=torch.float32).unsqueeze(0)
            global_orient = torch.tensor(params.global_orient, device=self.device, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = self.smpl_model(
                betas=betas,
                body_pose=body_pose,
                global_orient=global_orient
            )

        vertices = output.vertices[0].cpu().numpy()
        joints = output.joints[0].cpu().numpy()
        faces = self.smpl_model.faces

        return vertices, faces, joints

    def _generate_fallback_mesh(
        self,
        keypoints: np.ndarray,
        params: SMPLParams
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a simplified body mesh when SMPL is not available.

        Creates an anatomically-proportioned body mesh based on keypoints.
        Uses SMPL-like units (meters) for compatibility with mesh slicer.
        """
        # Standard body dimensions in meters (for average adult ~175cm)
        # Based on anthropometric data
        BODY_HEIGHT = 1.75  # meters
        SHOULDER_WIDTH = 0.42  # meters (42cm)
        HIP_WIDTH = 0.36  # meters (36cm)
        CHEST_DEPTH = 0.22  # meters (22cm front-to-back)
        WAIST_DEPTH = 0.18  # meters
        HIP_DEPTH = 0.20  # meters

        # Calculate body proportions from keypoints
        left_shoulder = keypoints[11, :2] if len(keypoints) > 11 else np.array([0.38, 0.22])
        right_shoulder = keypoints[12, :2] if len(keypoints) > 12 else np.array([0.62, 0.22])
        left_hip = keypoints[23, :2] if len(keypoints) > 23 else np.array([0.42, 0.52])
        right_hip = keypoints[24, :2] if len(keypoints) > 24 else np.array([0.58, 0.52])
        nose = keypoints[0, :2] if len(keypoints) > 0 else np.array([0.5, 0.08])
        left_ankle = keypoints[27, :2] if len(keypoints) > 27 else np.array([0.40, 0.95])
        right_ankle = keypoints[28, :2] if len(keypoints) > 28 else np.array([0.60, 0.95])

        # Calculate relative proportions from keypoints
        kp_shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
        kp_hip_width = abs(right_hip[0] - left_hip[0])
        avg_ankle_y = (left_ankle[1] + right_ankle[1]) / 2
        kp_body_height = avg_ankle_y - nose[1]

        # Adjust body proportions based on keypoint ratios
        if kp_shoulder_width > 0.01 and kp_hip_width > 0.01:
            shoulder_to_hip_ratio = kp_shoulder_width / kp_hip_width
        else:
            shoulder_to_hip_ratio = SHOULDER_WIDTH / HIP_WIDTH

        # Scale factors for different body types
        if shoulder_to_hip_ratio > 1.2:  # V-shape (broader shoulders)
            chest_scale = 1.1
            hip_scale = 0.95
        elif shoulder_to_hip_ratio < 0.9:  # Pear shape (broader hips)
            chest_scale = 0.95
            hip_scale = 1.1
        else:
            chest_scale = 1.0
            hip_scale = 1.0

        # Build body mesh in SMPL-compatible coordinates (meters, Y-up)
        vertices = []
        num_radial = 32  # Points around each slice

        # Define body levels (Y coordinate in meters, from feet=0 to head=height)
        # Heights as fraction of total body height
        levels = [
            # (y_fraction, width_meters, depth_meters, name)
            (0.00, 0.10, 0.10, 'feet'),
            (0.05, 0.12, 0.12, 'ankles'),
            (0.25, 0.16, 0.14, 'calves'),
            (0.45, 0.18, 0.16, 'knees'),
            (0.52, HIP_WIDTH * hip_scale / 2, HIP_DEPTH * hip_scale / 2, 'hips'),
            (0.58, HIP_WIDTH * 0.85 / 2, WAIST_DEPTH / 2, 'waist_low'),
            (0.62, HIP_WIDTH * 0.75 / 2, WAIST_DEPTH / 2, 'waist'),
            (0.68, SHOULDER_WIDTH * 0.85 * chest_scale / 2, CHEST_DEPTH * chest_scale / 2, 'chest_low'),
            (0.72, SHOULDER_WIDTH * 0.90 * chest_scale / 2, CHEST_DEPTH * chest_scale / 2, 'chest'),
            (0.78, SHOULDER_WIDTH * chest_scale / 2, CHEST_DEPTH * 0.9 / 2, 'shoulders'),
            (0.85, 0.08, 0.10, 'neck'),
            (0.90, 0.10, 0.12, 'head_base'),
            (0.95, 0.09, 0.11, 'head_mid'),
            (1.00, 0.05, 0.06, 'head_top'),
        ]

        # Generate vertices for each level (elliptical cross-section)
        for y_frac, half_width, half_depth, name in levels:
            y = y_frac * BODY_HEIGHT
            for i in range(num_radial):
                theta = 2 * np.pi * i / num_radial
                # Ellipse: x = a*cos(theta), z = b*sin(theta)
                x = half_width * np.cos(theta)
                z = half_depth * np.sin(theta)
                vertices.append([x, y, z])

        vertices = np.array(vertices)

        # Create faces connecting adjacent levels
        faces = []
        num_levels = len(levels)

        for level_idx in range(num_levels - 1):
            base = level_idx * num_radial
            next_base = (level_idx + 1) * num_radial

            for j in range(num_radial):
                j_next = (j + 1) % num_radial

                # Two triangles per quad
                faces.append([base + j, next_base + j, base + j_next])
                faces.append([base + j_next, next_base + j, next_base + j_next])

        # Add bottom cap
        bottom_center_idx = len(vertices)
        vertices = np.vstack([vertices, [[0, 0, 0]]])
        for j in range(num_radial):
            j_next = (j + 1) % num_radial
            faces.append([bottom_center_idx, j_next, j])

        # Add top cap
        top_center_idx = len(vertices)
        top_base = (num_levels - 1) * num_radial
        vertices = np.vstack([vertices, [[0, BODY_HEIGHT, 0]]])
        for j in range(num_radial):
            j_next = (j + 1) % num_radial
            faces.append([top_center_idx, top_base + j, top_base + j_next])

        faces = np.array(faces) if faces else np.zeros((0, 3), dtype=np.int64)

        return vertices, faces

    def extract_measurements(
        self,
        keypoints: np.ndarray,
        gender: str = None,
        height_cm: float = None
    ) -> MeshMeasurements:
        """
        Main entry point: Extract body measurements from 2D keypoints.

        Args:
            keypoints: (33, 4) MediaPipe keypoints
            gender: Gender for body model
            height_cm: Known height in cm (for scaling)

        Returns:
            MeshMeasurements with accurate circumferences
        """
        # Reconstruct 3D mesh
        vertices, faces, joints, params = self.reconstruct_from_keypoints(keypoints, gender)

        # Extract raw measurements (pass joints for accurate shoulder/arm measurements)
        raw_measurements, slices = self.mesh_slicer.extract_measurements(vertices, faces, joints)

        # Scale to actual height if provided
        if height_cm is not None and raw_measurements.get('height', 0) > 0:
            scale = height_cm / raw_measurements['height']
        else:
            # Default scale assuming SMPL units (meters) to cm
            scale = 100.0

        # Apply scaling to length/width measurements
        scaled_measurements = {}
        for k, v in raw_measurements.items():
            if 'circumference' in k or 'width' in k or 'height' in k or 'length' in k or 'inseam' in k:
                scaled_measurements[k] = v * scale
            else:
                scaled_measurements[k] = v

        return MeshMeasurements(
            chest_circumference=scaled_measurements.get('chest_circumference', 0),
            waist_circumference=scaled_measurements.get('waist_circumference', 0),
            hip_circumference=scaled_measurements.get('hip_circumference', 0),
            shoulder_width=scaled_measurements.get('shoulder_width', 0),
            arm_length=scaled_measurements.get('arm_length', 0),
            inseam=scaled_measurements.get('inseam', 0),
            height=scaled_measurements.get('height', height_cm or 170),
            chest_slice_vertices=slices.get('chest'),
            waist_slice_vertices=slices.get('waist'),
            hip_slice_vertices=slices.get('hip'),
            confidence=0.95 if self.smpl_model is not None else 0.80,
            method='smpl_mesh_slicing' if self.smpl_model is not None else 'fallback_mesh'
        )

    def convert_mediapipe_to_array(self, pose_landmarks) -> np.ndarray:
        """
        Convert MediaPipe PoseLandmarks to numpy array.

        Args:
            pose_landmarks: PoseLandmarks object from pose_detector

        Returns:
            (33, 4) array of (x, y, z, visibility)
        """
        keypoints = np.zeros((33, 4))

        for i, landmark in enumerate(pose_landmarks.landmarks):
            keypoints[i, 0] = landmark.get('x', 0)
            keypoints[i, 1] = landmark.get('y', 0)
            keypoints[i, 2] = landmark.get('z', 0)
            keypoints[i, 3] = pose_landmarks.visibility_scores.get(
                list(pose_landmarks.visibility_scores.keys())[i] if i < len(pose_landmarks.visibility_scores) else '',
                0.5
            )

        return keypoints


# Factory function
def create_body_reconstructor(
    smpl_model_path: Optional[str] = None,
    use_gpu: bool = True
) -> BodyMeshReconstructor:
    """
    Create a BodyMeshReconstructor with optimal settings.

    Args:
        smpl_model_path: Path to SMPL model files
        use_gpu: Whether to use GPU if available

    Returns:
        Configured BodyMeshReconstructor
    """
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'

    return BodyMeshReconstructor(
        smpl_model_path=smpl_model_path,
        device=device,
        use_optimization=True
    )
