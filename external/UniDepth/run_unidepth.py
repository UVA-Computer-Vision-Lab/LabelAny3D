import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
torch.backends.cudnn.enabled = False

from PIL import Image
from tqdm import tqdm
import json
import argparse
from unidepth.models import UniDepthV1


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset Configuration")
    parser.add_argument('--dataset', type=str, default='SUNRGBD', help='Name of the dataset')
    return parser.parse_args()

version="v1"
backbone="ViTL14"

model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)



def process_single_image_depth(scene_dir, K):
    try:
        rgb = torch.from_numpy(np.array(Image.open(f'{scene_dir}/input.png'))).permute(2, 0, 1)
    except Exception as e:
        print(e)
        return []

    intrinsics = np.array(K).reshape(3,3)
    intrinsics = torch.from_numpy(intrinsics).float()

    predictions = model.infer(rgb, intrinsics)
    depth = predictions["depth"]
    intrinsics = predictions["intrinsics"]

    return depth.cpu().numpy().squeeze(0).squeeze(0)

