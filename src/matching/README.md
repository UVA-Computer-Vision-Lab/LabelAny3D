# Matching Module

This is a concise readme to facilitate the usage of this pose estimation module. 

Object pose estimation via 2D-3D correspondence matching.

## Pipeline

```
Reference Image (crop)     3D Mesh (GLB)
        |                       |
        |                  Multi-view Render
        |                       |
        v                       v
   [MASt3R] <--- 2D-2D Match ---> Rendered Views
        |                           |
        |                      Depth + Camera
        |                           |
        v                           v
   2D Points  <------>  3D Points (world coords)
                  |
                  v
            PnP RANSAC
                  |
                  v
          Camera Pose (R, T)
```

## Files

| File | Description |
|------|-------------|
| `process_image_space.py` | Main entry point, orchestrates the pipeline |
| `renderer.py` | PyTorch3D-based GLB mesh renderer |
| `matcher.py` | MASt3R-based 2D-2D feature matching |
| `pose_estimator.py` | PnP pose estimation + camera conversion |

## Usage

```python
from matching.process_image_space import load_model, process_object

# Load MASt3R model (once)
device = torch.device("cuda:0")
model = load_model(device)

# Process single object
R, T, rendered_img, depth = process_object(
    object_name="0_chair",      # object identifier
    project_root="path/to/scene",  # scene directory
    model=model
)
```

## Input Structure

```
project_root/
  input.png                          # original image
  cam_params.json                    # camera intrinsics {"K": [...], "H": ..., "W": ...}
  crops/
    {object_name}_rgba.png           # cropped object image
    {object_name}_crop_params.npy    # [x_offset, y_offset, scale]
  object_space/
    {object_name}.glb                # reconstructed 3D mesh
    {object_name}/
      estimated_elevation.npy        # estimated camera elevation
```

## Output

- `R`: Rotation matrix (3x3)
- `T`: Translation vector (3,)
- Intermediate renderings saved to `object_space/{object_name}/renderings/`

## Dependencies

- PyTorch3D (rendering)
- MASt3R (feature matching)
- OpenCV (PnP solver)
