import cv2
import torch
from moge.model import MoGeModel
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import utils3d_moge
from moge.utils.io import save_glb, save_ply
from moge.utils.vis import colorize_depth, colorize_normal
from PIL import Image
import numpy as np

device = torch.device("cuda")

# Load the model from huggingface hub (or load from local).
model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)                             


def infer_geometry_on_image(image_path, out_dir):
    # Read the input image and convert to tensor (3, H, W) and normalize to [0, 1]
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)                       
    input_image = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)    

    # Infer 
    output = model.infer(input_image)
    points, depth, mask, intrinsics = output['points'].cpu().numpy(), output['depth'].cpu().numpy(), output['mask'].cpu().numpy(), output['intrinsics'].cpu().numpy()

    # transform the normalized intrinsics to the pixel coordinates
    W, H = image.shape[1], image.shape[0]
    intrinsics = intrinsics * np.array([[W, 1, W], [1, H, H], [1, 1, 1]])

    # save_moge_data( image, points, depth, mask, out_dir)

    return  points, depth, mask, intrinsics 

def save_moge_data( image, points, depth, mask, save_path ):
        height, width = image.shape[:2]
        normals, normals_mask = utils3d_moge.numpy.points_to_normals(points, mask=mask)

        faces, vertices, vertex_colors, vertex_uvs = utils3d_moge.numpy.image_mesh(
            points,
            image.astype(np.float32) / 255,
            utils3d_moge.numpy.image_uv(width=width, height=height),
            mask=mask & ~(utils3d_moge.numpy.depth_edge(depth, rtol=0.03, mask=mask) & utils3d_moge.numpy.normals_edge(normals, tol=5, mask=normals_mask)), # remove edge
            tri=True
        )

        # save a ply file with faces and edges removed, which could be visualized to reprojection by bpy_load_blender_pointmap.py
        save_ply(save_path / 'depth_scene_no_edge.ply', vertices, faces, vertex_colors)
        
        # save a ply file with only vertices, which is utilized in the following reconstruction
        # save_ply(save_path / 'depth_scene.ply', points.reshape(-1, 3), faces=None, vertex_colors=image.reshape(-1, 3))
