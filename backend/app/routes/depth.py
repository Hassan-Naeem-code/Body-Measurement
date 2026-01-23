"""
Depth Estimation API Routes

Provides endpoints for generating depth maps from images
using MiDaS (free, open-source depth estimation model).
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np
import cv2
import torch
import base64
import io
import logging
from PIL import Image

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/depth", tags=["Depth Estimation"])

# Global model cache
_midas_model = None
_midas_transform = None


class DepthRequest(BaseModel):
    """Request body for depth estimation"""
    image_base64: str = Field(..., description="Base64 encoded image")
    resolution: int = Field(default=384, ge=128, le=512, description="Output resolution")


class DepthResponse(BaseModel):
    """Response containing depth map data"""
    depth_map: List[List[float]]  # HxW normalized depth values (0-1)
    width: int
    height: int
    min_depth: float
    max_depth: float


def get_midas_model():
    """Get or create the MiDaS depth estimation model (cached)"""
    global _midas_model, _midas_transform

    if _midas_model is not None:
        return _midas_model, _midas_transform

    try:
        logger.info("Loading MiDaS depth estimation model...")

        # Use MiDaS small model for faster inference
        # Options: "MiDaS_small", "DPT_Hybrid", "DPT_Large"
        model_type = "MiDaS_small"  # Fast and good enough for 2.5D effect

        # Load model from torch hub
        _midas_model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        _midas_model.eval()

        # Use CPU (can switch to CUDA if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _midas_model.to(device)

        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        if model_type == "MiDaS_small":
            _midas_transform = midas_transforms.small_transform
        else:
            _midas_transform = midas_transforms.dpt_transform

        logger.info(f"MiDaS model loaded successfully on {device}")
        return _midas_model, _midas_transform

    except Exception as e:
        logger.error(f"Failed to load MiDaS model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load depth model: {str(e)}")


def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 image string to numpy array"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]

        image_bytes = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert('RGB')
        return np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")


@router.post("/estimate", response_model=DepthResponse)
async def estimate_depth(request: DepthRequest):
    """
    Estimate depth from a single image using MiDaS.

    Returns a depth map that can be used for 2.5D visualization.
    Depth values are normalized to 0-1 range.
    """
    try:
        # Decode image
        image = decode_base64_image(request.image_base64)
        original_height, original_width = image.shape[:2]

        # Get model
        model, transform = get_midas_model()
        device = next(model.parameters()).device

        # Transform image for model
        input_batch = transform(image).to(device)

        # Run inference
        with torch.no_grad():
            prediction = model(input_batch)

            # Resize to requested resolution while maintaining aspect ratio
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(request.resolution, request.resolution),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # Convert to numpy and normalize
        depth_map = prediction.cpu().numpy()

        # Normalize depth to 0-1 range
        min_depth = float(depth_map.min())
        max_depth = float(depth_map.max())
        if max_depth - min_depth > 0:
            depth_map = (depth_map - min_depth) / (max_depth - min_depth)
        else:
            depth_map = np.zeros_like(depth_map)

        return DepthResponse(
            depth_map=depth_map.tolist(),
            width=depth_map.shape[1],
            height=depth_map.shape[0],
            min_depth=min_depth,
            max_depth=max_depth
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error estimating depth: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/estimate-file")
async def estimate_depth_from_file(
    file: UploadFile = File(...),
    resolution: int = 384
):
    """
    Estimate depth from an uploaded image file.
    Alternative to base64 endpoint for direct file uploads.
    """
    try:
        # Read and decode image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image = image.convert('RGB')
        image_array = np.array(image)

        # Get model
        model, transform = get_midas_model()
        device = next(model.parameters()).device

        # Transform image for model
        input_batch = transform(image_array).to(device)

        # Run inference
        with torch.no_grad():
            prediction = model(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(resolution, resolution),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # Convert to numpy and normalize
        depth_map = prediction.cpu().numpy()

        min_depth = float(depth_map.min())
        max_depth = float(depth_map.max())
        if max_depth - min_depth > 0:
            depth_map = (depth_map - min_depth) / (max_depth - min_depth)
        else:
            depth_map = np.zeros_like(depth_map)

        return DepthResponse(
            depth_map=depth_map.tolist(),
            width=depth_map.shape[1],
            height=depth_map.shape[0],
            min_depth=min_depth,
            max_depth=max_depth
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error estimating depth: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Check if depth estimation service is available"""
    try:
        model, _ = get_midas_model()
        device = next(model.parameters()).device
        return {
            "status": "healthy",
            "model": "MiDaS_small",
            "device": str(device)
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}
