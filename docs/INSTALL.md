# Installation

Tested on Rocky Linux 8.10, NVIDIA Ampere A40, CUDA 12.1, Python 3.10.

## 1. Clone Repository

```bash
git clone https://github.com/UVA-Computer-Vision-Lab/LabelAny3D.git
cd LabelAny3D
export EXT_DIR=$(pwd)/external
```

## 2. Create Environment

```bash
conda create -n la3d python=3.10
conda activate la3d
```

## 3. Install PyTorch and Dependencies

```bash
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 4. Install Additional Packages

```bash
pip install git+https://github.com/facebookresearch/pytorch3d.git@055ab3a --no-build-isolation
pip install git+https://github.com/yaojin17/detectron2.git --no-build-isolation
pip install pycocotools==2.0 --no-build-isolation
```

## 5. Install External Dependencies

### MoGe (Depth Estimation)

```bash
cd $EXT_DIR/MoGe && pip install -r requirements.txt
```

### DepthPro (Metric Depth)

```bash
cd $EXT_DIR/ml-depth-pro && pip install -e .
```

### TRELLIS (3D Reconstruction)

Requires GCC 11+.

```bash
export CC=$(which gcc)
export CXX=$(which g++)
cd $EXT_DIR/TRELLIS && . ./setup.sh --basic --xformers --diffoctreerast --spconv --mipgaussian --nvdiffrast
```

### flash-attn

Download pre-built wheel for your Python/PyTorch/CUDA version from [flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases).

Example for Python 3.10, PyTorch 2.2, CUDA 12:

```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
rm flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### kaolin

Build from source (required for older glibc systems):

```bash
git clone --recursive https://github.com/NVIDIAGameWorks/kaolin.git /tmp/kaolin
cd /tmp/kaolin && git checkout v0.17.0
pip install . --no-build-isolation
```

### InvSR (Image Enhancement)

```bash
cd $EXT_DIR/InvSR && pip install -e . --no-deps
```

## 6. Download Checkpoints

```bash
cd $EXT_DIR/checkpoints
./download.sh
```

This downloads:
- **DepthPro** - metric depth estimation (~1.8GB)
- **InvSR** - image super-resolution (~130MB)
- **Amodal Completion** - complete occluded regions (~3.3GB)

Other models (TRELLIS, MoGe) are auto-downloaded from HuggingFace at runtime.

## 7. Blender (Optional, for Visualization)

Install Blender 3.6+ and add trimesh:

```bash
blender --background --python-expr "import subprocess, sys; subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'trimesh'])"
```

## Verify Installation

```bash
cd src
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```
