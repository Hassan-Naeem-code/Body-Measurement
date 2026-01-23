"""
Custom SMPL Model Loader

Loads SMPL model from npz files without requiring chumpy.
This bypasses the smplx library's pkl dependency issues.
"""

import numpy as np
import torch
import torch.nn as nn
import os
import logging
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SMPLOutput:
    """Output from SMPL forward pass"""
    vertices: torch.Tensor  # (B, 6890, 3)
    joints: torch.Tensor    # (B, 24, 3)
    full_pose: torch.Tensor # (B, 72)


class SMPLModel(nn.Module):
    """
    SMPL Body Model - Direct implementation

    Loads from npz files and performs linear blend skinning.
    """

    NUM_VERTICES = 6890
    NUM_JOINTS = 24
    NUM_BETAS = 10

    def __init__(
        self,
        model_path: str,
        gender: str = 'neutral',
        device: str = 'cpu'
    ):
        """
        Initialize SMPL model.

        Args:
            model_path: Path to directory containing SMPL npz files
            gender: 'neutral', 'male', or 'female'
            device: torch device
        """
        super().__init__()

        self.device = device
        self.gender = gender

        # Load model data
        data = self._load_model_data(model_path, gender)

        # Register buffers (non-trainable parameters)
        self.register_buffer('v_template', torch.tensor(data['v_template'], dtype=torch.float32))

        # shapedirs may have more than NUM_BETAS components - use only first 10
        shapedirs = data['shapedirs']
        if shapedirs.shape[-1] > self.NUM_BETAS:
            shapedirs = shapedirs[:, :, :self.NUM_BETAS]
        self.register_buffer('shapedirs', torch.tensor(shapedirs, dtype=torch.float32))

        self.register_buffer('posedirs', torch.tensor(data['posedirs'], dtype=torch.float32))
        self.register_buffer('J_regressor', torch.tensor(data['J_regressor'], dtype=torch.float32))
        self.register_buffer('weights', torch.tensor(data['weights'], dtype=torch.float32))
        self.register_buffer('kintree_table', torch.tensor(data['kintree_table'], dtype=torch.long))

        # Store faces
        self.faces = data['f'].astype(np.int64)

        # Parent joint indices for kinematic tree
        self.parents = self.kintree_table[0].tolist()
        self.parents[0] = -1  # Root has no parent

        logger.info(f"SMPL {gender} model loaded: {self.NUM_VERTICES} vertices, {len(self.faces)} faces")

    def _load_model_data(self, model_path: str, gender: str) -> Dict:
        """Load model data from npz file"""
        # Try different path patterns
        paths_to_try = [
            os.path.join(model_path, 'smpl', f'SMPL_{gender.upper()}.npz'),
            os.path.join(model_path, f'SMPL_{gender.upper()}.npz'),
            os.path.join(model_path, 'smpl', f'smpl_{gender.lower()}.npz'),
        ]

        for path in paths_to_try:
            if os.path.exists(path):
                logger.info(f"Loading SMPL from {path}")
                data = dict(np.load(path, allow_pickle=False))
                return data

        raise FileNotFoundError(f"SMPL model not found. Tried: {paths_to_try}")

    def forward(
        self,
        betas: torch.Tensor = None,
        body_pose: torch.Tensor = None,
        global_orient: torch.Tensor = None,
        transl: torch.Tensor = None,
        return_verts: bool = True
    ) -> SMPLOutput:
        """
        Forward pass of SMPL model.

        Args:
            betas: Shape parameters (B, 10)
            body_pose: Body pose parameters (B, 69) - 23 joints * 3
            global_orient: Global orientation (B, 3)
            transl: Translation (B, 3)
            return_verts: Whether to return vertices

        Returns:
            SMPLOutput with vertices and joints
        """
        batch_size = 1

        if betas is not None:
            batch_size = betas.shape[0]
            betas = betas.to(self.device)
        else:
            betas = torch.zeros(batch_size, self.NUM_BETAS, device=self.device)

        if body_pose is not None:
            body_pose = body_pose.to(self.device)
        else:
            body_pose = torch.zeros(batch_size, (self.NUM_JOINTS - 1) * 3, device=self.device)

        if global_orient is not None:
            global_orient = global_orient.to(self.device)
        else:
            global_orient = torch.zeros(batch_size, 3, device=self.device)

        if transl is not None:
            transl = transl.to(self.device)
        else:
            transl = torch.zeros(batch_size, 3, device=self.device)

        # Full pose: global orientation + body pose
        full_pose = torch.cat([global_orient, body_pose], dim=1)  # (B, 72)

        # 1. Add shape blend shapes to template
        v_shaped = self.v_template + torch.einsum('bl,vkl->bvk', betas, self.shapedirs)

        # 2. Get joint locations from shaped vertices
        J = torch.einsum('jv,bvk->bjk', self.J_regressor, v_shaped)

        # 3. Add pose blend shapes
        # Convert axis-angle to rotation matrices
        rot_mats = self._batch_rodrigues(full_pose.view(-1, 3)).view(batch_size, -1, 3, 3)

        # Pose blend shapes: (R - I) for joints 1-23 (excluding root), flattened
        # Each rotation matrix R is 3x3, subtract identity, flatten to 9 values
        # 23 joints * 9 = 207 pose features
        ident = torch.eye(3, device=self.device).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
        pose_feature = (rot_mats[:, 1:, :, :] - ident).view(batch_size, -1)  # (B, 23*9=207)
        v_posed = v_shaped + torch.einsum('bl,vkl->bvk', pose_feature, self.posedirs)

        # 4. Linear Blend Skinning
        # For T-pose (all zeros), skip complex LBS and use posed vertices directly
        pose_norm = torch.norm(full_pose)

        if pose_norm < 0.01:
            # T-pose: vertices are already in correct position
            vertices = v_posed + transl.unsqueeze(1)
        else:
            # Apply full LBS for non-zero poses
            transforms = self._get_transform_chain(rot_mats, J)
            T = torch.einsum('vj,bjkl->bvkl', self.weights, transforms)
            v_homo = torch.cat([v_posed, torch.ones(batch_size, self.NUM_VERTICES, 1, device=self.device)], dim=2)
            v_posed_homo = torch.einsum('bvkl,bvl->bvk', T, v_homo)
            vertices = v_posed_homo[:, :, :3] + transl.unsqueeze(1)

        # Get joint positions
        joints = torch.einsum('jv,bvk->bjk', self.J_regressor, vertices)

        return SMPLOutput(
            vertices=vertices,
            joints=joints,
            full_pose=full_pose
        )

    def _batch_rodrigues(self, rot_vecs: torch.Tensor) -> torch.Tensor:
        """
        Convert axis-angle to rotation matrix using Rodrigues formula.

        Args:
            rot_vecs: (N, 3) axis-angle vectors

        Returns:
            (N, 3, 3) rotation matrices
        """
        batch_size = rot_vecs.shape[0]

        angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
        rot_dir = rot_vecs / angle

        cos = torch.cos(angle).unsqueeze(2)
        sin = torch.sin(angle).unsqueeze(2)

        # Rodrigues formula
        rx, ry, rz = rot_dir[:, 0], rot_dir[:, 1], rot_dir[:, 2]
        zeros = torch.zeros_like(rx)

        K = torch.stack([
            torch.stack([zeros, -rz, ry], dim=1),
            torch.stack([rz, zeros, -rx], dim=1),
            torch.stack([-ry, rx, zeros], dim=1)
        ], dim=1)

        ident = torch.eye(3, device=rot_vecs.device).unsqueeze(0)
        rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)

        return rot_mat

    def _get_transform_chain(
        self,
        rot_mats: torch.Tensor,
        joints: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute transformation matrices for the kinematic chain.

        Args:
            rot_mats: (B, J, 3, 3) rotation matrices
            joints: (B, J, 3) joint positions

        Returns:
            (B, J, 4, 4) transformation matrices
        """
        batch_size = rot_mats.shape[0]
        num_joints = rot_mats.shape[1]

        # Create 4x4 transformation matrices
        transforms = torch.zeros(batch_size, num_joints, 4, 4, device=rot_mats.device)
        transforms[:, :, :3, :3] = rot_mats
        transforms[:, :, :3, 3] = joints
        transforms[:, :, 3, 3] = 1

        # Apply kinematic chain
        result = [transforms[:, 0]]

        for i in range(1, num_joints):
            parent_idx = self.parents[i]
            if parent_idx >= 0:
                # Transform relative to parent
                rel_transform = transforms[:, i].clone()
                rel_transform[:, :3, 3] = joints[:, i] - joints[:, parent_idx]
                result.append(torch.bmm(result[parent_idx], rel_transform))
            else:
                result.append(transforms[:, i])

        return torch.stack(result, dim=1)


def load_smpl_model(
    model_path: str,
    gender: str = 'neutral',
    device: str = 'cpu'
) -> Optional[SMPLModel]:
    """
    Load SMPL model from npz files.

    Args:
        model_path: Path to SMPL model directory
        gender: 'neutral', 'male', or 'female'
        device: torch device

    Returns:
        SMPLModel instance or None if loading fails
    """
    try:
        model = SMPLModel(model_path, gender, device)
        return model.to(device)
    except Exception as e:
        logger.error(f"Failed to load SMPL model: {e}")
        return None
