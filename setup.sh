pip install git+https://github.com/facebookresearch/pytorch3d.git@055ab3a
pip install git+https://github.com/yaojin17/detectron2.git  # slightly modified detectron2 for OVMono3D
pip install cython opencv-python scipy pandas einops open_clip_torch open3d

pip install git+https://github.com/apple/ml-depth-pro.git@b2cd0d5
pip install git+https://github.com/facebookresearch/segment-anything.git@dca509f
pip install git+https://github.com/IDEA-Research/GroundingDINO.git@856dde2

mkdir -p checkpoints
wget -P ./checkpoints/ https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
wget  -P checkpoints https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
huggingface-cli download uva-cv-lab/ovmono3d_lift ovmono3d_lift.pth --local-dir checkpoints
