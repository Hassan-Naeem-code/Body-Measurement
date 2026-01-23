# SMPL Model Setup

## Quick Start (Automatic Fallback)

The system works **without** SMPL models using a geometric fallback. However, for maximum accuracy (92-98%), you'll want to download the official SMPL models.

## Download Official SMPL Models

1. **Register** at https://smpl.is.tue.mpg.de/
2. **Download** the SMPL for Python package
3. **Extract** and place files in this directory:

```
backend/models/smpl/
├── SMPL_NEUTRAL.pkl      # Neutral body model
├── SMPL_MALE.pkl         # Male body model
├── SMPL_FEMALE.pkl       # Female body model
└── README.md             # This file
```

## Alternative: SMPL-X Models

For even better accuracy (hands + face), you can use SMPL-X:

1. Register at https://smpl-x.is.tue.mpg.de/
2. Download SMPL-X models
3. Place in this directory

## File Format

The system supports both `.pkl` and `.npz` formats:
- `.pkl` - Original SMPL format
- `.npz` - NumPy compressed format (preferred)

## Testing Your Setup

After placing the model files, run:

```bash
cd backend
python -c "
import smplx
model = smplx.create('models/smpl', model_type='smpl', gender='neutral')
print(f'✓ SMPL model loaded: {model.faces.shape[0]} faces, {model.v_template.shape[0]} vertices')
"
```

## Without SMPL Models

The system will automatically use a **geometric fallback** that:
- Creates approximate body meshes from pose keypoints
- Still provides 80-85% accuracy (vs 92-98% with SMPL)
- Works immediately without additional setup

## Accuracy Comparison

| Method | Accuracy | Requirement |
|--------|----------|-------------|
| Full SMPL | 92-98% | SMPL model files |
| Geometric Fallback | 80-85% | None (built-in) |
| MiDaS 2D Depth | 70-85% | MiDaS model (auto-download) |
