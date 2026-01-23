"""
3D Mesh API Routes

Provides endpoints for generating and retrieving 3D body meshes
for visualization in web-based Three.js viewers.
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np
import torch
import json
import logging
import io

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mesh3d", tags=["3D Mesh"])


class MeshRequest(BaseModel):
    """Request body for generating a 3D mesh"""
    height_cm: float = Field(default=175.0, ge=100, le=250, description="Height in centimeters")
    gender: str = Field(default="neutral", description="Gender: 'male', 'female', or 'neutral'")
    # Shape parameters (optional) - affects body proportions
    weight_factor: float = Field(default=0.0, ge=-3, le=3, description="Weight factor: -3 (thin) to +3 (heavy)")

    # Optional: keypoints from pose detection (for personalized mesh)
    keypoints: Optional[List[List[float]]] = Field(default=None, description="33x4 keypoints array from pose detection")


class MeshResponse(BaseModel):
    """Response containing 3D mesh data"""
    vertices: List[List[float]]  # Nx3 array of vertex positions
    faces: List[List[int]]       # Mx3 array of triangle indices
    vertex_colors: Optional[List[List[float]]] = None  # Nx3 RGB colors (0-1)
    uv_coordinates: Optional[List[List[float]]] = None  # Nx2 UV texture coordinates
    measurements: dict           # Body measurements


class TexturedMeshRequest(BaseModel):
    """Request for generating textured mesh with photo"""
    height_cm: float = Field(default=175.0, ge=100, le=250)
    gender: str = Field(default="neutral")
    weight_factor: float = Field(default=0.0, ge=-3, le=3)
    image_base64: Optional[str] = Field(default=None, description="Base64 encoded image for texture")


def get_mesh_reconstructor():
    """Get or create the body mesh reconstructor (cached)"""
    import sys
    import os

    # Add app to path if needed
    backend_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    try:
        from app.ml.body_mesh_reconstructor import BodyMeshReconstructor

        reconstructor = BodyMeshReconstructor(
            smpl_model_path='models/smpl',
            device='cpu',
            use_optimization=False
        )
        return reconstructor
    except Exception as e:
        logger.error(f"Failed to create mesh reconstructor: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize 3D model: {str(e)}")


@router.post("/generate", response_model=MeshResponse)
async def generate_mesh(request: MeshRequest):
    """
    Generate a 3D body mesh based on parameters.

    Returns vertices, faces, and measurements that can be rendered in Three.js.
    """
    try:
        reconstructor = get_mesh_reconstructor()

        # Create keypoints (use provided or generate default standing pose)
        if request.keypoints:
            keypoints = np.array(request.keypoints)
        else:
            # Default standing pose keypoints
            keypoints = np.zeros((33, 4))
            keypoints[0] = [0.5, 0.08, 0, 0.99]   # Nose
            keypoints[11] = [0.35, 0.22, 0, 0.98]  # Left shoulder
            keypoints[12] = [0.65, 0.22, 0, 0.98]  # Right shoulder
            keypoints[13] = [0.28, 0.35, 0, 0.95]  # Left elbow
            keypoints[14] = [0.72, 0.35, 0, 0.95]  # Right elbow
            keypoints[15] = [0.25, 0.48, 0, 0.90]  # Left wrist
            keypoints[16] = [0.75, 0.48, 0, 0.90]  # Right wrist
            keypoints[23] = [0.42, 0.52, 0, 0.98]  # Left hip
            keypoints[24] = [0.58, 0.52, 0, 0.98]  # Right hip
            keypoints[25] = [0.40, 0.70, 0, 0.95]  # Left knee
            keypoints[26] = [0.60, 0.70, 0, 0.95]  # Right knee
            keypoints[27] = [0.38, 0.92, 0, 0.90]  # Left ankle
            keypoints[28] = [0.62, 0.92, 0, 0.90]  # Right ankle
            # Fill remaining with defaults
            for i in range(33):
                if keypoints[i, 3] == 0:
                    keypoints[i] = [0.5, 0.5, 0, 0.5]

        # Get mesh and measurements
        vertices, faces, joints, params = reconstructor.reconstruct_from_keypoints(
            keypoints,
            gender=request.gender
        )

        # Apply weight factor to shape if SMPL model is available
        if reconstructor.smpl_model is not None and request.weight_factor != 0:
            # Regenerate with weight factor
            betas = torch.zeros(1, 10)
            betas[0, 0] = request.weight_factor  # First beta controls overall size

            with torch.no_grad():
                output = reconstructor.smpl_model(
                    betas=betas,
                    body_pose=torch.zeros(1, 69),
                    global_orient=torch.zeros(1, 3)
                )
            vertices = output.vertices[0].cpu().numpy()
            joints = output.joints[0].cpu().numpy()

        # Scale vertices to requested height
        current_height = vertices[:, 1].max() - vertices[:, 1].min()
        target_height_m = request.height_cm / 100.0
        scale = target_height_m / current_height
        vertices = vertices * scale

        # Center the mesh
        center = vertices.mean(axis=0)
        vertices = vertices - center

        # Extract measurements
        measurements, _ = reconstructor.mesh_slicer.extract_measurements(
            vertices / scale,  # Use unscaled for measurement extraction
            reconstructor.smpl_model.faces if reconstructor.smpl_model else faces,
            joints if joints is not None else None
        )

        # Scale measurements
        for key in measurements:
            if 'circumference' in key or 'width' in key or 'height' in key or 'length' in key or 'inseam' in key:
                measurements[key] = measurements[key] * scale

        # Generate vertex colors based on measurement regions
        vertex_colors = generate_measurement_colors(vertices, request.height_cm)

        return MeshResponse(
            vertices=vertices.tolist(),
            faces=faces.tolist(),
            vertex_colors=vertex_colors,
            measurements={
                "height_cm": request.height_cm,
                "chest_cm": round(measurements.get('chest_circumference', 0), 1),
                "waist_cm": round(measurements.get('waist_circumference', 0), 1),
                "hip_cm": round(measurements.get('hip_circumference', 0), 1),
                "shoulder_cm": round(measurements.get('shoulder_width', 0), 1),
                "arm_length_cm": round(measurements.get('arm_length', 0), 1),
                "inseam_cm": round(measurements.get('inseam', 0), 1),
                "gender": request.gender
            }
        )

    except Exception as e:
        logger.error(f"Error generating mesh: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/generate")
async def generate_mesh_get(
    height_cm: float = Query(default=175.0, ge=100, le=250),
    gender: str = Query(default="neutral"),
    weight_factor: float = Query(default=0.0, ge=-3, le=3)
):
    """GET endpoint for generating mesh (for easy testing)"""
    request = MeshRequest(
        height_cm=height_cm,
        gender=gender,
        weight_factor=weight_factor
    )
    return await generate_mesh(request)


@router.get("/export/obj")
async def export_obj(
    height_cm: float = Query(default=175.0),
    gender: str = Query(default="neutral"),
    weight_factor: float = Query(default=0.0)
):
    """Export mesh as OBJ file for download"""
    request = MeshRequest(height_cm=height_cm, gender=gender, weight_factor=weight_factor)
    mesh_data = await generate_mesh(request)

    # Generate OBJ content
    obj_lines = ["# FitWhisperer 3D Body Mesh", f"# Height: {height_cm}cm, Gender: {gender}", ""]

    # Vertices
    for v in mesh_data.vertices:
        obj_lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")

    obj_lines.append("")

    # Faces (OBJ uses 1-indexed)
    for f in mesh_data.faces:
        obj_lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}")

    obj_content = "\n".join(obj_lines)

    return Response(
        content=obj_content,
        media_type="model/obj",
        headers={"Content-Disposition": f"attachment; filename=body_mesh_{gender}_{int(height_cm)}cm.obj"}
    )


def generate_measurement_colors(vertices: np.ndarray, height_cm: float) -> List[List[float]]:
    """
    Generate vertex colors to highlight measurement regions.

    Colors:
    - Chest region: Red
    - Waist region: Green
    - Hip region: Blue
    - Rest: Skin tone
    """
    height_m = height_cm / 100.0
    min_y = vertices[:, 1].min()
    max_y = vertices[:, 1].max()
    body_height = max_y - min_y

    # Measurement levels (as fraction of height)
    chest_level = 0.72
    waist_level = 0.62
    hip_level = 0.52

    colors = []
    skin_color = [0.96, 0.80, 0.69]  # Light skin tone

    for v in vertices:
        y_frac = (v[1] - min_y) / body_height

        # Check if vertex is near measurement levels
        if abs(y_frac - chest_level) < 0.03:
            colors.append([1.0, 0.3, 0.3])  # Red for chest
        elif abs(y_frac - waist_level) < 0.03:
            colors.append([0.3, 1.0, 0.3])  # Green for waist
        elif abs(y_frac - hip_level) < 0.03:
            colors.append([0.3, 0.3, 1.0])  # Blue for hip
        else:
            colors.append(skin_color)

    return colors


def generate_uv_coordinates(vertices: np.ndarray) -> List[List[float]]:
    """
    Generate UV coordinates for texture mapping.

    Uses simple front projection for the body:
    - U (horizontal): based on X position (left to right)
    - V (vertical): based on Y position (bottom to top)

    This projects the image directly onto the front of the body like a billboard.
    """
    min_x = vertices[:, 0].min()
    max_x = vertices[:, 0].max()
    min_y = vertices[:, 1].min()
    max_y = vertices[:, 1].max()

    width = max_x - min_x
    height = max_y - min_y

    uv_coords = []

    for v in vertices:
        # U coordinate: horizontal position (0 = left, 1 = right)
        # Invert X so left side of person appears on left of texture
        u_coord = (v[0] - min_x) / width

        # V coordinate: vertical position (0 = bottom, 1 = top)
        v_coord = (v[1] - min_y) / height

        # Add some padding to avoid edge artifacts
        u_coord = 0.05 + u_coord * 0.9
        v_coord = 0.02 + v_coord * 0.96

        uv_coords.append([u_coord, v_coord])

    return uv_coords


@router.post("/generate-textured", response_model=MeshResponse)
async def generate_textured_mesh(request: TexturedMeshRequest):
    """
    Generate a 3D body mesh with UV coordinates for texture mapping.

    The mesh includes UV coordinates that allow projecting the uploaded
    photo onto the 3D body model.
    """
    try:
        reconstructor = get_mesh_reconstructor()

        # Create default standing pose keypoints
        keypoints = np.zeros((33, 4))
        keypoints[0] = [0.5, 0.08, 0, 0.99]   # Nose
        keypoints[11] = [0.35, 0.22, 0, 0.98]  # Left shoulder
        keypoints[12] = [0.65, 0.22, 0, 0.98]  # Right shoulder
        keypoints[13] = [0.28, 0.35, 0, 0.95]  # Left elbow
        keypoints[14] = [0.72, 0.35, 0, 0.95]  # Right elbow
        keypoints[15] = [0.25, 0.48, 0, 0.90]  # Left wrist
        keypoints[16] = [0.75, 0.48, 0, 0.90]  # Right wrist
        keypoints[23] = [0.42, 0.52, 0, 0.98]  # Left hip
        keypoints[24] = [0.58, 0.52, 0, 0.98]  # Right hip
        keypoints[25] = [0.40, 0.70, 0, 0.95]  # Left knee
        keypoints[26] = [0.60, 0.70, 0, 0.95]  # Right knee
        keypoints[27] = [0.38, 0.92, 0, 0.90]  # Left ankle
        keypoints[28] = [0.62, 0.92, 0, 0.90]  # Right ankle
        for i in range(33):
            if keypoints[i, 3] == 0:
                keypoints[i] = [0.5, 0.5, 0, 0.5]

        # Get mesh and measurements
        vertices, faces, joints, params = reconstructor.reconstruct_from_keypoints(
            keypoints,
            gender=request.gender
        )

        # Apply weight factor
        if reconstructor.smpl_model is not None and request.weight_factor != 0:
            betas = torch.zeros(1, 10)
            betas[0, 0] = request.weight_factor

            with torch.no_grad():
                output = reconstructor.smpl_model(
                    betas=betas,
                    body_pose=torch.zeros(1, 69),
                    global_orient=torch.zeros(1, 3)
                )
            vertices = output.vertices[0].cpu().numpy()
            joints = output.joints[0].cpu().numpy()

        # Scale vertices to requested height
        current_height = vertices[:, 1].max() - vertices[:, 1].min()
        target_height_m = request.height_cm / 100.0
        scale = target_height_m / current_height
        vertices = vertices * scale

        # Center the mesh
        center = vertices.mean(axis=0)
        vertices = vertices - center

        # Extract measurements
        measurements, _ = reconstructor.mesh_slicer.extract_measurements(
            vertices / scale,
            reconstructor.smpl_model.faces if reconstructor.smpl_model else faces,
            joints if joints is not None else None
        )

        # Scale measurements
        for key in measurements:
            if 'circumference' in key or 'width' in key or 'height' in key or 'length' in key or 'inseam' in key:
                measurements[key] = measurements[key] * scale

        # Generate UV coordinates for texture mapping
        uv_coordinates = generate_uv_coordinates(vertices)

        return MeshResponse(
            vertices=vertices.tolist(),
            faces=faces.tolist(),
            vertex_colors=None,  # No vertex colors when using texture
            uv_coordinates=uv_coordinates,
            measurements={
                "height_cm": request.height_cm,
                "chest_cm": round(measurements.get('chest_circumference', 0), 1),
                "waist_cm": round(measurements.get('waist_circumference', 0), 1),
                "hip_cm": round(measurements.get('hip_circumference', 0), 1),
                "shoulder_cm": round(measurements.get('shoulder_width', 0), 1),
                "arm_length_cm": round(measurements.get('arm_length', 0), 1),
                "inseam_cm": round(measurements.get('inseam', 0), 1),
                "gender": request.gender
            }
        )

    except Exception as e:
        logger.error(f"Error generating textured mesh: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Health check
@router.get("/health")
async def health_check():
    """Check if 3D mesh service is available"""
    try:
        reconstructor = get_mesh_reconstructor()
        has_smpl = reconstructor.smpl_model is not None
        return {
            "status": "healthy",
            "smpl_model": "loaded" if has_smpl else "fallback",
            "confidence": "95%" if has_smpl else "80%"
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}
