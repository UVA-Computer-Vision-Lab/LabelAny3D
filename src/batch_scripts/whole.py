import argparse
from omegaconf import OmegaConf
import sys
import os
import json
from tqdm import tqdm
import torch
import trimesh
import cv2
sys.path = ['./',] + sys.path
from dataset_model import get_scene
from pathlib import Path
import numpy as np
from PIL import Image
from util import restore_mask_from_crop, align_to_depth_match, draw_cube
from util_3dbox import save_3d_with_ground_alignment_bbox
from matching.process_image_space import load_model
from batch_scripts.coconut_loader import CoconutLoader, get_dataset_paths


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

    assert (torch.cuda.is_available())
    device = f"cuda:{args.gpu_idx}"

    mast3r_model = load_model(device)
    for i in tqdm(range(args.start_index, args.end_index)):
        image_info = loader.get_image_by_index(i)
        img_name = image_info["file_name"]
        image_path = os.path.join(dataset_root, img_name)
        output_dir = os.path.join(args.save_dir, args.split, img_name.split(".")[0].replace("/", "_").replace("-", "_"))

        opt.scene.attributes.img_path = image_path
        opt.run.amodal_completion = 'our'
        scene = get_scene(opt.scene.type, opt.scene.attributes)

        out_dir = Path(output_dir)
        print(f"Saving to {out_dir}")
        out_dir.mkdir(exist_ok=True, parents=True)
        (out_dir / "crops").mkdir(exist_ok=True)
        (out_dir / "object_space").mkdir(exist_ok=True)
        (out_dir / "reconstruction").mkdir(exist_ok=True)

        image_size = scene.image_pil.size

        if os.path.exists(out_dir / '3dbbox.json'):
            continue
        with open(out_dir / 'cam_params.json', 'r') as fp:
            cam_params = json.load(fp)
        K_img = np.array(cam_params['K'])
        pose = np.array(cam_params['c2w'])
        depth_map = np.load(out_dir / 'depth_map.npy')

        scene_mesh = trimesh.Scene([None])
        crop_root = out_dir / "crops"
        crop_paths = list(crop_root.glob("*_reproj.png"))
        for i in range(len(crop_paths) - 1, -1, -1):
            crop_path = crop_paths[i]
            obj_id = crop_path.stem.replace("_reproj", "")
            label = obj_id.split("_", 1)[-1]

            # Check if full crop exists
            crop = Image.open(crop_path)
            crop_params_path = out_dir / "crops" / f"{obj_id}_crop_params.npy"
            if not crop_params_path.exists():
                continue
            crop_params = np.load(crop_params_path)
            resized_mask = np.array(crop)[:, :, 3] > 127
            mask = restore_mask_from_crop(resized_mask, crop_params[0], crop_params[1], crop_params[2], scene.image_np.shape[:2])
            full_crop_path = out_dir / "crops" / f"{obj_id}_rgba.png"
            if not full_crop_path.exists():
                full_crop_path = out_dir / "crops" / f"{obj_id}_reproj.png"
            full_crop = Image.open(full_crop_path)

            # Check if elevation exists
            elevation_path = out_dir / "object_space" / f"{obj_id}" / "estimated_elevation.npy"
            obj_elevation = np.load(elevation_path)
            object_space_path = out_dir / "object_space" / f"{obj_id}.glb"
            if not os.path.exists(object_space_path):
                print(f"Object space file {object_space_path} does not exist")
                continue
            obj_mesh = trimesh.load(object_space_path)
            if isinstance(obj_mesh, trimesh.Scene):
                # Dump all meshes from the scene
                meshes = obj_mesh.dump()
                obj_mesh = meshes[0]

            project_root = out_dir
            try:
                transform = align_to_depth_match(mask, depth_map, obj_id, project_root, mast3r_model)
            except Exception as e:
                print(f"Error aligning {obj_id}: {e}")
                continue
            obj_mesh.apply_transform(transform)
            obj_mesh.apply_transform(pose)

            convention_transform = np.array(
                [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            )

            obj_mesh.apply_transform(convention_transform)

            obj_mesh.export(out_dir / 'reconstruction' / f"{obj_id}.glb")
            scene_mesh.add_geometry([obj_mesh])
            print(f"Saved, {obj_id}.glb")
            canonical_upright = (convention_transform @ transform)[:, 1]
            np.save(out_dir / 'reconstruction' / f'{obj_id}_canonical_upright.npy', canonical_upright)
        if len(scene_mesh.geometry) > 0:
            scene_mesh.export(out_dir / 'reconstruction' / 'full_scene.glb')

            print("Going to save ground aligned bbox")
            save_3d_with_ground_alignment_bbox(out_dir)
            draw_cube(out_dir, is_ground=True)

            # Rename to remove _ground suffix
            if os.path.exists(out_dir / '3dbbox_ground.json'):
                os.rename(out_dir / '3dbbox_ground.json', out_dir / '3dbbox.json')
