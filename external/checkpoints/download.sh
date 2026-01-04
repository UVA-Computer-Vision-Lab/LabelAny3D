#!/bin/bash
# Download model weights for LA3D pipeline
# Run from this directory: cd external/checkpoints && bash download.sh
#
# Usage:
#   bash download.sh              # Download all weights

set -e

echo "=== Downloading model weights to $(pwd) ==="

# ============================================================================
# Required for batch_scripts/ (COCO pipeline)
# ============================================================================

# DepthPro - metric depth estimation
echo "Downloading DepthPro..."
wget -nc https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt || true

# InvSR - image super-resolution/enhancement
echo "Downloading InvSR..."
wget -nc https://huggingface.co/OAOA/InvSR/resolve/main/noise_predictor_sd_turbo_v5.pth || true

# Amodal completion - complete occluded object regions
# Requires both config.json and diffusion_pytorch_model.safetensors for UNet2DConditionModel.from_pretrained()
echo "Downloading Amodal Completion..."
mkdir -p amodal_completion
wget -nc https://huggingface.co/andreead-a/amodal-completion/resolve/main/unet/config.json -P amodal_completion || true
wget -nc https://huggingface.co/andreead-a/amodal-completion/resolve/main/unet/diffusion_pytorch_model.safetensors -P amodal_completion || true


# ============================================================================
echo ""
echo "=== Download complete ==="
echo ""
echo "Downloaded files:"
ls -lh *.pth *.pt 2>/dev/null || true
ls -lh amodal_completion/ 2>/dev/null || true
echo ""
echo "Note: TRELLIS, MoGe, MASt3R are auto-downloaded from HuggingFace at runtime."
