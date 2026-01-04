import os
import sys
import cv2
import numpy as np
import torch
import json

sys.path.append('../external/mast3r')
from mast3r.model import AsymmetricMASt3R
from dust3r.utils.image import load_images
# print(sys.path)
sys.path.append('./')
from matching.renderer import GLBRenderer
from matching.matcher import ImageMatcher
from matching.pose_estimator import PoseEstimator

def setup_device():
    """Setup CUDA device if available"""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return device

def load_model(device):
    """Load MAST3R model"""
    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    model.eval()
    return model


def process_object(object_name, project_root, model):
    """Main processing pipeline for a single object"""
    # Setup
    device = setup_device()
    
    # Initialize components
    renderer = GLBRenderer(device)
    matcher = ImageMatcher(model, device)
    pose_estimator = PoseEstimator(device)
    
    # Load and render mesh
    mesh = renderer.load_mesh(f'{project_root}/object_space/{object_name}.glb')
    elevation = -np.load(f'{project_root}/object_space/{object_name}/estimated_elevation.npy')
    elevations = [elevation] * 8
    azimuths = list(range(0, 360, 45))
    
    render_dir = f'{project_root}/object_space/{object_name}/renderings'
    rgbs, depths, Rs, Ts = renderer.render_multiple_views(
        mesh, render_dir, elevations, azimuths
    )
    
    # Load reference image
    unprocessed_img0_path = f'{project_root}/crops/{object_name}_rgba.png'
    if not os.path.exists(unprocessed_img0_path):
        unprocessed_img0_path = f'{project_root}/crops/{object_name}_reproj.png'
    unprocessed_img0 = cv2.imread(unprocessed_img0_path, cv2.IMREAD_UNCHANGED)
    unprocessed_img0 = cv2.cvtColor(unprocessed_img0, cv2.COLOR_BGRA2RGBA)
    ref_img = load_images([unprocessed_img0_path], 512, verbose=False)[0]

    # Process each view
    all_points_world = None
    all_points_target = None
    
    for i in range(len(rgbs)):
        rgb_18 = cv2.imread(f'{render_dir}/rgb_{i}.png', cv2.IMREAD_UNCHANGED)
        depth = depths[i]
        R = Rs[i].cpu().numpy()
        T = Ts[i].cpu().numpy()
        
        render_img = load_images([f'{render_dir}/rgb_{i}.png'], 512, verbose=False)[0]
        points_world, unprocessed_matches0 = matcher.get_correspondences(
            ref_img, render_img, unprocessed_img0, rgb_18, depth, T, R
        )
        
        if all_points_world is None:
            all_points_world = points_world
            all_points_target = unprocessed_matches0
        else:
            all_points_world = np.concatenate((all_points_world, points_world), axis=0)
            all_points_target = np.concatenate((all_points_target, unprocessed_matches0), axis=0)
    
    # Estimate initial pose
    camera_matrix = np.array([
        [560.44, 0, 256],
        [0, 560.44, 256],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((5,1))
    
    success, rvec, tvec, inliers, error, _ = pose_estimator.estimate_pose_pnp(
        all_points_world, all_points_target, camera_matrix, dist_coeffs
    )
    
    if success:
        print("Initial pose estimation successful!")
        print("Rotation vector:", rvec)
        print("Translation vector:", tvec)
        print("Number of inliers:", len(inliers))
        print("Mean reprojection error:", error)
        
        # Create camera from estimated pose
        cameras = pose_estimator.create_camera_from_pose(
            rvec, tvec, camera_matrix, [512, 512]
        )
        
        # Render with estimated pose
        
        image_render, depth = renderer.render_mesh(mesh, cameras, None, None)
        cv2.imwrite(f'{render_dir}/rgb_iter1.png', image_render[..., [2,1,0]] * 255)
        
        # Refine pose estimation
        rgb_18 = cv2.imread(f'{render_dir}/rgb_iter1.png', cv2.IMREAD_UNCHANGED)
        R = cameras.R.cpu().numpy()
        T = cameras.T.cpu().numpy()
        
        render_img = load_images([f'{render_dir}/rgb_iter1.png'], 512, verbose=False)[0]
        points_world, unprocessed_matches0 = matcher.get_correspondences(
            ref_img, render_img, unprocessed_img0, rgb_18, depth, T, R
        )
        image_camera_intrinsic = json.load(open(f'{project_root}/cam_params.json'))
        crop_params = np.load(f'{project_root}/crops/{object_name}_crop_params.npy')
        image_camera_matrix = np.array(image_camera_intrinsic["K"]).astype('float32')
        image_H = image_camera_intrinsic["H"]
        image_W = image_camera_intrinsic["W"]
        unprocessed_matches0_image = unprocessed_matches0/crop_params[2] + np.array([[crop_params[0], crop_params[1]]])

        success, rvec, tvec, inliers, error, _ = pose_estimator.estimate_pose_pnp(
            points_world, unprocessed_matches0_image, image_camera_matrix, dist_coeffs
        )
        
        if success:
            print("Refined pose estimation successful!")
            print("Rotation vector:", rvec)
            print("Translation vector:", tvec)
            print("Number of inliers:", len(inliers))
            print("Mean reprojection error:", error)
            
            # Render final result
            cameras = pose_estimator.create_camera_from_pose(
                rvec, tvec, image_camera_matrix, [image_H, image_W]
            )
            image_render, depth = renderer.render_mesh(mesh, cameras, None, None, [image_H, image_W])
            
            # Visualize results
            # Read original image
            # img0 = cv2.imread(f'{project_root}/crops/{object_name}_restored.png', cv2.IMREAD_UNCHANGED)
            img0 = cv2.imread(f'{project_root}/input.png', cv2.IMREAD_UNCHANGED)
            
            # Convert rendered image to uint8
            # img1 = (image_render[..., [2,1,0,3]] * 255).astype(np.uint8)
            img1 = (image_render[..., [2,1,0]] * 255).astype(np.uint8)
            
            # Concatenate horizontally and save
            img_concat = np.concatenate((img0, img1), axis=1)
            cv2.imwrite(f'{render_dir}/rgb_iter2.png', img_concat)
            return cameras.R.cpu().numpy(), cameras.T.cpu().numpy(), image_render, depth
    else:
        print("Pose estimation failed")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process object in image space')
    parser.add_argument('--object_name', type=str, required=True, help='Object name (e.g., 0_toilet)')
    parser.add_argument('--project_root', type=str, required=True, help='Path to project root directory')
    args = parser.parse_args()

    device = setup_device()
    model = load_model(device)
    process_object(args.object_name, args.project_root, model) 