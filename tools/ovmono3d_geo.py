import json
import pickle
import os
import sys
import torch
import numpy as np
import pdb
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pytorch3d.transforms import (
    rotation_6d_to_matrix, 
    matrix_to_rotation_6d,
)
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from segment_anything import SamPredictor, sam_model_registry
import glob
from pytorch3d import _C
import depth_pro
import tqdm
from sklearn.utils import shuffle

sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

from cubercnn.data import xywh_to_xyxy
import cubercnn.util as util

def project_3d_to_2d(X, Y, Z, K):

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    x = (fx * X) / Z + cx
    y = (fy * Y) / Z + cy

    return x, y


def get_dims(bbox3d):
    x = np.sqrt(np.sum((bbox3d[0] - bbox3d[1]) * (bbox3d[0] - bbox3d[1])))
    y = np.sqrt(np.sum((bbox3d[0] - bbox3d[3]) * (bbox3d[0] - bbox3d[3])))
    z = np.sqrt(np.sum((bbox3d[0] - bbox3d[4]) * (bbox3d[0] - bbox3d[4])))
    return np.array([z, y, x])

def get_pose(bbox3d_a, bbox3d_b):
    # assume a and b share the same bbox center and have same dimension
    center = np.mean(bbox3d_a, axis=0)
    dim_a = get_dims(bbox3d_a)
    dim_b = get_dims(bbox3d_b)
    bbox3d_a -= center 
    bbox3d_b -= center 
    U, _, Vt = np.linalg.svd(bbox3d_a.T @ bbox3d_b, full_matrices=True)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def auto_downsample(points, max_points):
    """
    If the number of points exceeds max_points, randomly sample down to max_points.
    Otherwise, return the original point cloud.
    
    Parameters:
        points (numpy.ndarray): Input point cloud with shape (N, D), where N is the number of points, and D is the dimension.
        max_points (int): The maximum number of points to retain.
        
    Returns:
        sampled_points (numpy.ndarray): The downsampled point cloud.
    """
    num_points = len(points)
    if num_points > max_points:
        # Randomly sample points
        sampled_points = shuffle(points, random_state=42)[:max_points]
        print(f"Points downsampled from {num_points} to {max_points}.")
    else:
        sampled_points = points
        print(f"Points remain unchanged: {num_points}.")
    return sampled_points

# (3) for each annotation, load image, run seg anything, unproject, clustering, 3D bbox, save to new annotations
def build_lineset(bbox3d, color=[1,0,0], flip=True):
    if flip:
        flip_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        bbox3d_flip = bbox3d.dot(flip_matrix)
    else:
        bbox3d_flip = bbox3d.copy()
    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
             [4, 5], [5, 6], [6, 7], [4, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    # Use the same color for all lines
    colors = [color for _ in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(bbox3d_flip)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def gen_8corners(x_min, y_min, z_min, cx, cy, cz):
    corners_flag = [[0,0,0], [1,0,0], [1,1,0], [0,1,0],
               [0,0,1], [1,0,1], [1,1,1], [0,1,1]]
    corners = []
    for flag in corners_flag:
        c = np.array([x_min, y_min, z_min]) + np.array(flag) * np.array([cx, cy, cz])
        corners.append(c)
    return np.array(corners)

def heading2rotmat(heading_angle):
    rotmat = np.zeros((3,3))
    rotmat[1, 1] = 1
    cosval = np.cos(heading_angle)
    sinval = np.sin(heading_angle)
    rotmat[0, 0] = cosval
    rotmat[0, 2] = -sinval
    rotmat[2, 0] = sinval
    rotmat[2, 2] = cosval
    return rotmat


def build_pseudo_bbox3d_from_mask2d_outlier(mask2d, depth, K):
    frustum = []
    depth = np.array(depth) # HxW

    ys, xs = np.where(mask2d > 0.5)
    # (1) generate mask 
    for y, x in zip(ys, xs):
        # (2) unproject 2d points (visualize in 3D) 
        z = depth[y, x]
        x_3d = z * (x - K[0, 2]) / K[0, 0]
        y_3d = z * (y - K[1, 2]) / K[1, 1]
        frustum.append([x_3d, -y_3d, -z]) # flip
    frustum = np.array(frustum)

    # (3) fit 3D bounding boxes (visualize in 3D)
    xyz_offset = np.mean(frustum, axis=0)
    xyz = frustum - xyz_offset
    pca = PCA(2)
    pca.fit(xyz[:, [0, 2]]) # xz plane
    yaw_vec = pca.components_[0, :]
    yaw = np.arctan2(yaw_vec[1], yaw_vec[0])
    xyz_tmp = xyz.copy()
    pose = heading2rotmat(-yaw)
    xyz_tmp = (pose @ xyz_tmp[:,:3].T).T
    xyz_tmp += xyz_offset

    # remove outliers
    eps=0.01
    min_samples=100
    trial_time = 0
    # print(len(xyz_tmp))
    max_points = 40000
    xyz_tmp = auto_downsample(xyz_tmp, max_points)
    while True:
        trial_time += 1
        if trial_time > 4:
            xyz_clean = xyz_tmp.copy()
            break
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(xyz_tmp)
        xyz_clean = []
        count_points = 0
        for cluster in np.unique(db.labels_):
            if cluster < 0:
                continue
            cluster_ind = np.where(db.labels_ == cluster)[0]
            if cluster_ind.shape[0] / xyz_tmp.shape[0] < 0.1 or cluster_ind.shape[0] <=100:
                continue
            xyz_clean.append(xyz_tmp[cluster_ind, :])
            count_points += len(cluster_ind)
        if count_points > 0.5 * len(xyz_tmp):
            xyz_clean = np.concatenate(xyz_clean, axis=0)
            print("%d --> %d" % (len(xyz_tmp), len(xyz_clean)))
            break
        else:
            eps = 2 * eps
            print("try once more: eps = %f" % eps)
    # xyz_clean = xyz_tmp

    x_min = xyz_tmp[:,0].min()
    x_max = xyz_tmp[:,0].max()
    y_max = xyz_tmp[:,1].min()
    y_min = xyz_tmp[:,1].max()
    z_max = xyz_tmp[:,2].min()
    z_min = xyz_tmp[:,2].max()
    dx_orig = x_max-x_min
    dy_orig = y_max-y_min
    dz_orig = z_max-z_min

    x_min = xyz_clean[:,0].min()
    x_max = xyz_clean[:,0].max()
    y_max = xyz_clean[:,1].min()
    y_min = xyz_clean[:,1].max()
    z_max = xyz_clean[:,2].min()
    z_min = xyz_clean[:,2].max()
    dx = x_max-x_min
    dy = y_max-y_min
    dz = z_max-z_min
    # 8 corners
    bbox3d_pseudo = gen_8corners(x_min, y_min, z_min, dx, dy, dz)
    bbox3d_pseudo -= xyz_offset
    bbox = heading2rotmat(yaw) @ bbox3d_pseudo.T
    bbox = bbox.T + xyz_offset
    lineset = build_lineset(bbox, color=[0,0,1], flip=False)
    return bbox, lineset, (dx, dy, dz), yaw


def run_seg_anything(model, im, bbox2D):
    model.set_image(im, image_format="BGR")
    bbox = np.array(bbox2D) # XYXY
    masks, _, _ = model.predict(box=bbox)
    return masks


def run_one_2dbox_to_3d(depth_o3d, mask2d, rgb_o3d, K):

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=rgb_o3d,
        depth=depth_o3d,
        depth_scale=1.0,  
        depth_trunc=1000.0,  
        convert_rgb_to_intensity=False
    )
    # try:
    if True:
        print("start build pseudo bbox3d")
        bbox3d_pseudo, _, _, yaw = build_pseudo_bbox3d_from_mask2d_outlier(
            mask2d, rgbd_image.depth, K
        ) 
        print("end build pseudo bbox3d")

    flip_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    bbox3d_pseudo = bbox3d_pseudo.dot(flip_matrix)

    # center, dimension, then get the pose 
    # such that conver from (center, dimension, pose) to 8 corners 
    # aligning with the pseudo label
    cube_dims = torch.from_numpy(get_dims(bbox3d_pseudo)).unsqueeze(0)
    cube_3d = torch.from_numpy(np.mean(bbox3d_pseudo, axis=0)).unsqueeze(0)
    cube_pose = torch.eye(3).unsqueeze(0)
    bbox3d_infer = util.get_cuboid_verts_faces(
        torch.cat((cube_3d, cube_dims), dim=1), 
        cube_pose,
    )[0]
    bbox3d_infer = bbox3d_infer.squeeze().numpy()

    cube_pose_new = get_pose(bbox3d_pseudo, bbox3d_infer)
    bbox3d_infer2 = util.get_cuboid_verts_faces(
        torch.cat((cube_3d, cube_dims), dim=1), 
        cube_pose_new,
    )[0]
    bbox3d_infer2 = bbox3d_infer2.squeeze().numpy()
    return cube_3d.tolist(), cube_dims.tolist(), cube_pose_new.tolist(), bbox3d_infer2.tolist()


dataset_list = {
                'KITTI_test_novel': './datasets/Omni3D/gdino_kitti_novel_oracle_2d.json',
                'ARKitScenes_test_novel': './datasets/Omni3D/gdino_arkitscenes_novel_oracle_2d.json',
                'SUNRGBD_test_novel': './datasets/Omni3D/gdino_sunrgbd_novel_oracle_2d.json',}

# Load model and preprocessing transform
depthpro_model, depthpro_transform = depth_pro.create_model_and_transforms(device=torch.device("cuda"),precision=torch.float16)
depthpro_model.eval()

ckpt = "./checkpoints/sam_vit_h_4b8939.pth"
sam = sam_model_registry["default"](checkpoint=ckpt).to(device="cuda")
seg_predictor = SamPredictor(sam)

threshold = 0.30

for dataset_name, dataset_pth in dataset_list.items():
    with open(dataset_pth, 'r') as f:
        dataset = json.load(f)
    root = "./datasets/"
    with open(os.path.join(root, "Omni3D", f"{dataset_name}.json"), "r") as file:
        gt_anns = json.load(file)
    imgid2path = {}
    for img in gt_anns["images"]:
        imgid2path[img['id']] = img['file_path']
    new_dataset = []
    for img in tqdm.tqdm(dataset):
        im_path = os.path.join(root, imgid2path[img['image_id']])

        # Load and preprocess an image.
        image, _, f_px = depth_pro.load_rgb(im_path)
        image = depthpro_transform(image)

        # Run inference.
        prediction = depthpro_model.infer(image, f_px=f_px)
        depth = prediction["depth"]  # Depth in [m].

        depth_numpy = depth.cpu().numpy().astype(np.float32)

        depth_o3d = o3d.geometry.Image(depth_numpy)
        new_instances = []
        rgb = cv2.imread(im_path)
        rgb_o3d = o3d.io.read_image(im_path)
        K = np.array(img['K'])
        for ins in img["instances"]:
            if ins['score'] < threshold:
                continue
            bbox2D = xywh_to_xyxy(ins["bbox"])
            mask2D = run_seg_anything(seg_predictor, rgb, bbox2D)
            mask2d = mask2D[2, :, :] # largest mask
            cube_3d, cube_dims, cube_pose_new, bbox3d_infer2 = run_one_2dbox_to_3d(depth_o3d, mask2d, rgb_o3d, K)

            new_instance = {key: value for key, value in ins.items() if key in ['category_id', 'bbox', 'score', 'category_name']}
            new_instance["image_id"] = img['image_id']
            new_instance["bbox3D"] = bbox3d_infer2
            new_instance["depth"] = cube_3d[0][-1]

            new_instance["center_cam"] = cube_3d[0]
            new_instance["dimensions"] = cube_dims[0]
            new_instance["pose"] = cube_pose_new
            x, y = project_3d_to_2d(cube_3d[0][0], cube_3d[0][1], cube_3d[0][2], K)
            new_instance["center_2D"] = [x, y]
            new_instances.append(new_instance)
            
        new_img = {key: value for key, value in img.items()}
        new_img["instances"] = new_instances
        new_dataset.append(new_img)
    # Create output directory if it doesn't exist
    output_dir = "./output/ovmono3d_geo"
    os.makedirs(output_dir, exist_ok=True)
    
    torch.save(new_dataset, f"{output_dir}/{dataset_name}.pth")
