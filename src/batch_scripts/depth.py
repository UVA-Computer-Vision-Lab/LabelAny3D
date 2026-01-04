"""
Depth estimation pipeline: MoGe + DepthPro + RANSAC alignment.

Uses MoGe for scale-invariant depth, DepthPro for metric depth,
and RANSAC to align MoGe depth to DepthPro scale.

Usage:
    python batch_scripts/depth.py --start_index 0 --end_index 100 --split val
"""
import argparse
from omegaconf import OmegaConf
import sys
import os
from tqdm import tqdm
import torch
sys.path = [
    './',
    '../external/MoGe',
] + sys.path
from dataset_model import get_scene
from pathlib import Path
import numpy as np
import json
import trimesh

import depth_pro
import utils3d_moge
from moge.utils.io import save_ply
from infer_moge import infer_geometry_on_image

from sklearn.linear_model import RANSACRegressor, LinearRegression
from util import depth_to_points
from batch_scripts.coconut_loader import CoconutLoader, get_dataset_paths


def save_moge_data(image, points, depth, mask, save_path):
    """Save MoGe output as PLY mesh with edges removed."""
    height, width = image.shape[:2]
    normals, normals_mask = utils3d_moge.numpy.points_to_normals(points, mask=mask)

    faces, vertices, vertex_colors, vertex_uvs = utils3d_moge.numpy.image_mesh(
        points,
        image.astype(np.float32) / 255,
        utils3d_moge.numpy.image_uv(width=width, height=height),
        mask=mask & ~(utils3d_moge.numpy.depth_edge(depth, rtol=0.03, mask=mask) &
                      utils3d_moge.numpy.normals_edge(normals, tol=5, mask=normals_mask)),
        tri=True
    )
    save_ply(save_path / 'depth_scene_no_edge.ply', vertices, faces, vertex_colors)


def align_depth(relative_depth, metric_depth, mask=None, min_samples=0.2, max_valid_depth=400.0):
    """
    Align scale-invariant depth to metric depth using RANSAC linear regression.

    Args:
        relative_depth: Input scale-invariant depth map (e.g., from MoGe).
        metric_depth: Reference metric depth map (e.g., from DepthPro).
        mask: Optional mask to specify valid fitting regions.
        min_samples: Minimum proportion of samples for RANSAC.
        max_valid_depth: Maximum metric depth to be considered valid.

    Returns:
        Aligned metric depth map.
    """
    regressor = RANSACRegressor(estimator=LinearRegression(fit_intercept=False), min_samples=min_samples)

    valid = (~np.isinf(relative_depth)) & (metric_depth < max_valid_depth)
    if mask is not None:
        valid &= mask

    if valid.sum() == 0:
        print("Warning: No valid points for alignment. Returning metric depth.")
        return metric_depth

    try:
        regressor.fit(relative_depth[valid].reshape(-1, 1), metric_depth[valid].reshape(-1, 1))
    except Exception as e:
        print(f"Error fitting RANSACRegressor: {e}, using metric depth directly")
        return metric_depth

    depth = np.full_like(relative_depth, 10000.0)

    if mask is not None:
        masked_pred = regressor.predict(relative_depth[mask].reshape(-1, 1)).flatten()
        depth[mask] = masked_pred
    else:
        valid_mask = ~np.isinf(relative_depth)
        masked_pred = regressor.predict(relative_depth[valid_mask].reshape(-1, 1)).flatten()
        depth[valid_mask] = masked_pred

    return depth


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to the yaml config file", default='configs/image.yaml', type=str)
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index')
    parser.add_argument('--start_index', type=int, default=0, help='Object index to start processing')
    parser.add_argument('--end_index', type=int, default=1, help='Object index to end processing')
    parser.add_argument("--split", help="split", default="val", type=str)
    parser.add_argument("--save_dir", help="save directory", default="../experimental_results/COCO/", type=str)

    args, extras = parser.parse_known_args()
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    # Load COCONUT data
    dataset_root, annotations_dir = get_dataset_paths(args.split)
    loader = CoconutLoader(split=args.split, annotations_dir=annotations_dir)

    assert torch.cuda.is_available()
    device = f"cuda:{args.gpu_idx}"

    # Load DepthPro model once
    print("Loading DepthPro model...")
    depthpro_model, depthpro_transform = depth_pro.create_model_and_transforms(device=device, precision=torch.float16)
    depthpro_model.eval()
    print("DepthPro model loaded.")

    for i in tqdm(range(args.start_index, args.end_index)):
        image_info = loader.get_image_by_index(i)
        img_name = image_info["file_name"]
        image_path = os.path.join(dataset_root, img_name)
        output_dir = os.path.join(args.save_dir, args.split, img_name.split(".")[0].replace("/", "_").replace("-", "_"))

        opt.scene.attributes.img_path = image_path
        scene = get_scene(opt.scene.type, opt.scene.attributes)

        out_dir = Path(output_dir)
        print(f"Saving to {out_dir}")
        out_dir.mkdir(exist_ok=True, parents=True)
        (out_dir / "crops").mkdir(exist_ok=True)
        (out_dir / "object_space").mkdir(exist_ok=True)
        (out_dir / "reconstruction").mkdir(exist_ok=True)

        # Save input image
        if not os.path.exists(out_dir / 'input.png'):
            scene.image_pil.save(out_dir / 'input.png')

        # Skip if already processed
        if os.path.exists(out_dir / 'depth_map.npy') and os.path.exists(out_dir / 'cam_params.json'):
            continue

        # Step 1: MoGe - Scale-invariant depth estimation
        _, moge_depth_map, moge_mask, K_img = infer_geometry_on_image(f'{out_dir}/input.png', out_dir)

        # Step 2: DepthPro - Metric depth estimation
        img = depthpro_transform(scene.image_pil)
        prediction = depthpro_model.infer(img, f_px=K_img[0, 0])
        pro_depth_map = prediction["depth"].cpu().numpy()

        # Step 3: Align depth and save results
        depth_map = align_depth(moge_depth_map, pro_depth_map, mask=moge_mask)
        pts3d = depth_to_points(depth_map[None], K_img)
        save_moge_data(scene.image_np, pts3d, depth_map, moge_mask, out_dir)
        np.save(out_dir / 'depth_map.npy', depth_map)
        trimesh.PointCloud(pts3d.reshape(-1, 3), scene.image_np.reshape(-1, 3)).export(out_dir / 'depth_scene.ply')

        pose = np.eye(4)
        cam_params = {
            'K': K_img.tolist(),
            'c2w': pose.tolist(),
            'W': scene.image_pil.width,
            'H': scene.image_pil.height,
        }
        with open(out_dir / 'cam_params.json', 'w') as fp:
            json.dump(cam_params, fp)
