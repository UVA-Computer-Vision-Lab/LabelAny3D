# COCO Dataset Processing Pipeline

This document describes how to run the LabelAny3D pipeline on COCO dataset to generate 3D bounding box annotations.


## Download Data

From the repository root directory:

```bash
# Download COCO images + COCONUT annotations
# Note: COCONUT annotations for COCO training set requires more than one hour to process. 
bash src/download_coco.sh

# Or download separately:
bash src/download_coco.sh --coco      # Only COCO images
bash src/download_coco.sh --coconut   # Only COCONUT annotations 
```

This creates:
```
dataset/coco/
├── images/
│   ├── train2017/
│   └── val2017/
└── annotations/
    ├── instances_train2017.json
    ├── instances_val2017.json
    ├── coconut_train.json
    └── coconut_val.json
```

## Run the Pipeline

**All commands should be run from the `src/` directory:**

```bash
cd src
```

### Step 1: Depth Estimation

Combines MoGe (scale-invariant) + DepthPro (metric) with RANSAC alignment.

```bash
python batch_scripts/depth.py --start_index=0 --end_index=1000 --split=val
```

### Step 2: Image Enhancement

Super-resolution using InvSR.

```bash
python batch_scripts/enhance.py --start_index=0 --end_index=1000 --split=val
```

### Step 3: Object Cropping

Extract individual object crops using COCONUT segmentation masks.

```bash
python batch_scripts/get_crops_enhanced.py --start_index=0 --end_index=1000 --split=val
```

### Step 4: Amodal Completion

Complete occluded regions of object crops.

```bash
python batch_scripts/completion.py --start_index=0 --end_index=1000 --split=val
```

### Step 5: Elevation Estimation

Estimate viewing angle for each object.

```bash
python batch_scripts/elevation.py --start_index=0 --end_index=1000 --split=val
```

### Step 6: 3D Reconstruction

Per-object 3D reconstruction using TRELLIS.

```bash
# Set compiler paths (required for TRELLIS)
export CC=$(which gcc)
export CXX=$(which g++)

python batch_scripts/reconstruction.py --start_index=0 --end_index=1000 --split=val
```

### Step 7: Scene Layout Alignment

Align reconstructed objects into the scene using depth-guided placement.

```bash
python batch_scripts/whole.py --start_index=0 --end_index=1000 --split=val
```

### Step 8: Combine Results

Combine all scene results into Omni3D format JSON.

```bash
python tools/combine_results.py --split=val
```

**Output:** `../experimental_results/COCO/COCO3D_val.json`

## Arguments

All batch scripts support:

| Argument | Description | Default |
|----------|-------------|---------|
| `--start_index` | Start image index | 0 |
| `--end_index` | End image index | -1 (all) |
| `--split` | Dataset split | `val` |

## Output Structure

```
experimental_results/COCO/
├── val/
│   ├── 000000000139/              # Scene folder (image ID)
│   │   ├── input.png              # Original image
│   │   ├── cam_params.json        # Camera intrinsics
│   │   ├── depth_map.npy          # Aligned depth map
│   │   ├── depth_scene.ply        # Scene point cloud
│   │   ├── depth_scene_no_edge.ply # Point cloud (no edge artifacts)
│   │   ├── bboxes.json            # 2D bboxes from COCONUT
│   │   ├── 3dbbox.json            # 3D bounding boxes
│   │   ├── enhanced/              # Super-resolved images
│   │   ├── crops/                 # Object crops
│   │   │   ├── 0_chair_reproj.png # original object crop
│   │   │   ├── 0_chair_rgba.png   # amodal-completed crop
│   │   │   └── 0_chair_crop_params.npy
│   │   ├── object_space/          # Per-object intermediate results
│   │   ├── reconstruction/        # 3D meshes
│   │   │   ├── 0_chair.glb
│   │   │   └── full_scene.glb
│   │   └── scene_bbox.mp4         # Rendered video with bboxes
│   └── ...
└── COCO3D_val.json                # Combined annotations (Omni3D format)
```

## Visualization

Render 3D scenes with bounding box overlays using Blender.

### Setup (one-time)

```bash
# Install trimesh into Blender's Python
blender --background --python-expr "import subprocess, sys; subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'trimesh'])"
```

### Render Scenes

```bash
blender --background --python bpy_render/bpy_load_blender_pointmap_plot.py -- \
    --root ../experimental_results/COCO/val [--start_idx 0] [--end_idx 10] [--verbose]
```

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--root` | Root directory containing scene folders | (required) |
| `--start_idx` | Start directory index | 0 |
| `--end_idx` | End directory index | last |
| `--verbose` | Show rendering progress | false |

**Output:** `scene_bbox.mp4` in each scene folder - camera trajectory video with 3D bbox overlay.

## SLURM Batch Processing

For HPC clusters, you can process in parallel using SLURM array jobs:

```bash
#!/bin/bash
#SBATCH --array=0-9
#SBATCH --gres=gpu:1

START=$((SLURM_ARRAY_TASK_ID * 100))
END=$((START + 100))

python batch_scripts/depth.py --start_index=$START --end_index=$END --split=val
```
