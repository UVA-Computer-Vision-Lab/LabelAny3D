import cv2
import trimesh
import torch
import numpy as np
from PIL import Image
from sklearn.linear_model import RANSACRegressor, LinearRegression
import rembg
import os
import json
from pycocotools import mask as mask_utils


def initialize_acompletion(device):
    from diffusers import StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel
    Pix2PixCfg = {
        'initial_model': "runwayml/stable-diffusion-v1-5",
        'finetuned_unet': "../external/checkpoints/amodal_completion",
    }
    unet = UNet2DConditionModel.from_pretrained(
        Pix2PixCfg['finetuned_unet'],
    )
    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        Pix2PixCfg['initial_model'], unet=unet
    ).to(device)

    def disabled_safety_checker(images, clip_input):
        if len(images.shape) == 4:
            num_images = images.shape[0]
            return images, [False] * num_images
        else:
            return images, False

    pipeline.safety_checker = disabled_safety_checker

    return pipeline


def initialize_zero123(device):
    from zero123 import Zero123Pipeline
    pipe = Zero123Pipeline.from_pretrained(
        "ashawkey/zero123-xl-diffusers",
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to(device)
    pipe.image_encoder.eval()
    pipe.vae.eval()
    pipe.unet.eval()
    pipe.clip_camera_projection.eval()
    return pipe


def depth_to_points(depth, K=None, R=None, t=None):
    """
    Reference: https://github.com/isl-org/ZoeDepth/blob/edb6daf45458569e24f50250ef1ed08c015f17a7/zoedepth/utils/geometry.py
    """
    Kinv = np.linalg.inv(K)
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)

    height, width = depth.shape[1:3]

    x = np.arange(width)
    y = np.arange(height)
    coord = np.stack(np.meshgrid(x, y), -1)
    coord = np.concatenate((coord, np.ones_like(coord)[:, :, [0]]), -1)  # z=1
    coord = coord.astype(np.float32)
    coord = coord[None]  # bs, h, w, 3

    D = depth[:, :, :, None, None]
    pts3D_1 = D * Kinv[None, None, None, ...] @ coord[:, :, :, :, None]
    # from reference to target viewpoint
    pts3D_2 = R[None, None, None, ...] @ pts3D_1 + t[None, None, None, :, None]
    return pts3D_2[:, :, :, :3, 0][0]


def estimate_elevation(img, cache_dir, zero123, dtype=torch.float16):
    from elevation_estimate.utils.elev_est_api import elev_est_api
    cache_dir.mkdir(exist_ok=True)
    img = img.astype(np.float32) / 255.0
    img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])
    img = Image.fromarray(np.uint8((img * 255).clip(0, 255)))
    delta_x_2 = [-10, 10, 0, 0]
    delta_y_2 = [0, 0, -10, 10]
    tensor_def = {
        "device": zero123.device,
        "dtype": dtype
    }
    out = zero123(
        [img] * len(delta_x_2),
        torch.tensor(delta_x_2, **tensor_def),      # elevation
        torch.tensor(delta_y_2, **tensor_def),      # azimuth
        torch.zeros(len(delta_x_2), **tensor_def)   # distance
    )
    file_paths = []
    for i in range(len(out[0])):
        opath = str((cache_dir / f"{i}.png").absolute())
        file_paths.append(opath)
        out[0][i].save(opath)
    elev = elev_est_api(file_paths)
    if elev is not None:
        elev -= 90
    else:
        elev = 0
        print("!!!Failed to estimate elevation!!!")
    np.save(cache_dir / 'estimated_elevation.npy', elev)


def align_depth(relative_depth, metric_depth, mask=None, min_samples=0.2):
    regressor = RANSACRegressor(estimator=LinearRegression(fit_intercept=True), min_samples=min_samples)
    if mask is not None:
        try:
            regressor.fit(relative_depth[mask].reshape(-1, 1), metric_depth[mask].reshape(-1, 1))
        except Exception as e:
            print(f"Error fitting RANSACRegressor: {e}, using metric depth directly")
            return metric_depth
    else:
        regressor.fit(relative_depth.reshape(-1, 1), metric_depth.reshape(-1, 1))
    
    # Initialize output depth array with large values
    depth = np.full_like(relative_depth, 10000.0)
    
    # Only predict for masked regions to avoid inf values
    if mask is not None:
        # Get prediction only for masked values
        masked_pred = regressor.predict(relative_depth[mask].reshape(-1, 1)).flatten()
        # Assign predictions back to masked locations
        depth[mask] = masked_pred
    else:
        # Get prediction for non-inf values
        valid_mask = ~np.isinf(relative_depth)
        masked_pred = regressor.predict(relative_depth[valid_mask].reshape(-1, 1)).flatten()
        depth[valid_mask] = masked_pred
        
    return depth



def crop_object(image: np.ndarray, mask: np.ndarray, crop_size=256):
    """Crop object from image using mask, with padding to maintain aspect ratio."""
    x, y, w, h = cv2.boundingRect(np.uint8(mask))
    max_size = max(w, h)
    ratio = 0.7
    side_len = int(max_size / ratio)

    # Create padded crop centered on object
    padded_image = np.zeros((side_len, side_len, 3), dtype=image.dtype)
    padded_mask = np.zeros((side_len, side_len), dtype=mask.dtype)
    center = side_len // 2
    padded_image[center - h // 2:center - h // 2 + h, center - w // 2:center - w // 2 + w] = image[y:y + h, x:x + w]
    padded_mask[center - h // 2:center - h // 2 + h, center - w // 2:center - w // 2 + w] = mask[y:y + h, x:x + w]
    resized_image = cv2.resize(padded_image, (crop_size, crop_size), cv2.INTER_LANCZOS4)
    resized_mask = cv2.resize(np.uint8(padded_mask), (crop_size, crop_size), cv2.INTER_LANCZOS4) == 1
    offset_x = x + (w - side_len) / 2
    offset_y = y + (h - side_len) / 2
    scale_factor = crop_size / side_len
    crop = Image.fromarray(np.concatenate([resized_image, np.uint8(resized_mask[:, :, None]) * 255], axis=-1))

    return crop, (offset_x, offset_y, scale_factor)


def segment_completed(completed_crop, original_crop):
    orig_mask = np.array(original_crop)[..., -1] / 255 > 0.5
    new_mask = np.array(rembg.remove(completed_crop, rembg.new_session("isnet-general-use"), post_process_mask=True))
    new_mask[:, :, :3][orig_mask] = np.array(completed_crop)[orig_mask]
    new_mask[:, :, -1][orig_mask] = 255
    return Image.fromarray(new_mask)


def restore_mask_from_crop(resized_mask: np.ndarray, offset_x: float, offset_y: float,
                           scale_factor: float, original_mask_shape: tuple) -> np.ndarray:
    """
    Restore the original-size mask from a cropped and resized mask.

    Args:
        resized_mask (np.ndarray): (H, W), boolean or 0/1 mask from the crop
        offset_x (float): x-offset of the crop center relative to the original image
        offset_y (float): y-offset of the crop center relative to the original image
        scale_factor (float): scale from the cropped region to the resized mask
        original_mask_shape (tuple): shape of the original mask, e.g., (H, W)

    Returns:
        np.ndarray: boolean mask restored to the original image size
    """
    # Resize back to the pre-padded size
    original_crop_size = int(resized_mask.shape[0] / scale_factor)
    unpadded_mask = cv2.resize(resized_mask.astype(np.uint8),
                               (original_crop_size, original_crop_size),
                               interpolation=cv2.INTER_NEAREST)

    # Create an empty mask of the original image size
    restored_mask = np.zeros(original_mask_shape, dtype=np.uint8)

    # Compute the paste coordinates in the original image
    x1 = int(round(offset_x))
    y1 = int(round(offset_y))
    x2 = x1 + unpadded_mask.shape[1]
    y2 = y1 + unpadded_mask.shape[0]

    # Clip the coordinates to image boundaries
    x1_clip, x2_clip = max(x1, 0), min(x2, original_mask_shape[1])
    y1_clip, y2_clip = max(y1, 0), min(y2, original_mask_shape[0])

    # Compute the corresponding region in the unpadded mask
    mx1 = x1_clip - x1
    my1 = y1_clip - y1
    mx2 = mx1 + (x2_clip - x1_clip)
    my2 = my1 + (y2_clip - y1_clip)

    # Paste the unpadded mask back into the full-size mask
    restored_mask[y1_clip:y2_clip, x1_clip:x2_clip] = unpadded_mask[my1:my2, mx1:mx2]

    return restored_mask.astype(bool)


def complete_crop(crop, label, model, run_opt):
    """Complete a cropped object using amodal completion."""
    from model_wrappers import complete_object
    if run_opt == 'our':
        completed = complete_object(crop, label, model)
        return segment_completed(completed, crop)
    else:
        return crop


def project_to_2d(point_3d, camera_matrix):
    point_2d_homogeneous = np.dot(camera_matrix, point_3d)
    return point_2d_homogeneous[:2] / point_2d_homogeneous[2]


def draw_cube(scene_dir, is_ground=False):
    # Load camera parameters
    with open(os.path.join(scene_dir, "cam_params.json"), 'r') as json_file:
        cam_param = json.load(json_file)
    K = np.array(cam_param["K"])

    # Load appropriate bbox file based on is_ground flag
    bbox_file = "3dbbox_ground.json" if is_ground else "3dbbox.json"
    with open(os.path.join(scene_dir, bbox_file), 'r') as json_file:
        cube_list = json.load(json_file)

    # Load and convert image
    image = Image.open(os.path.join(scene_dir, "input.png"))
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for cube in cube_list:
        # Project 3D points to 2D
        verts = cube["bbox3D_cam"]
        points_2d = [project_to_2d(np.array(point), K) for point in verts]
        points_2d = np.array(points_2d)

        # if points_2d.max() <= 1.0:  # This indicates we have normalized coordinates
        #     print("2d points were normalized")
            # Scale to image dimensions
        # image_height, image_width = image.shape[:2]
        # points_2d[:, 0] *= image_width
        # points_2d[:, 1] *= image_height
        min_y = float('inf')
        topmost_point = None
        topmost_index = -1
        for i, point in enumerate(points_2d):
            if point[1] < min_y:
                min_y = point[1]
                topmost_point = point
                topmost_index = i

        # Draw points
        for i, point in enumerate(points_2d):
            point_int = tuple(np.round(point).astype(int))
            cv2.circle(image, point_int, radius=3, color=(0, 255, 0), thickness=-1)  # green circles
            # cv2.putText(image, f'v{i}', point_int, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # white texts
        # Define the edges of a cube
        edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7)]
        for start, end in edges:
            start_point = tuple(np.round(points_2d[start]).astype(int))
            end_point = tuple(np.round(points_2d[end]).astype(int))
            cv2.line(image, start_point, end_point, (255, 0, 0), 2)

        # Draw category name
        if topmost_point is not None:
            text_position = (int(topmost_point[0]), int(topmost_point[1]) - 10)  
            cv2.putText(image, f'{cube["category_name"]}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Save output image
    output_file = 'vis_3dbox.png' if is_ground else 'vis_3dbox_no_ground.png'
    cv2.imwrite(os.path.join(scene_dir, output_file), image)

def analyze_mask(mask, image_size, scale_threshold=100, boundary_threshold=10):
    """
    Analyzes a binary mask for scale and boundary truncation.
    """
    image_width = image_size[0]
    image_height = image_size[1]
    # Ensure the mask is binary
    if not np.array_equal(mask, mask.astype(bool)):
        raise ValueError("Image Mask must be binary (contain only 0s and 1s).")

    # Calculate the scale (area of the mask)
    scale = np.sum(mask)

    # Create boundary masks
    top_boundary = np.zeros_like(mask)
    top_boundary[:boundary_threshold, :] = 1

    bottom_boundary = np.zeros_like(mask)
    bottom_boundary[-boundary_threshold:, :] = 1

    left_boundary = np.zeros_like(mask)
    left_boundary[:, :boundary_threshold] = 1

    right_boundary = np.zeros_like(mask)
    right_boundary[:, -boundary_threshold:] = 1

    # Find intersection of the mask with each boundary
    top_intersection = np.sum(mask * top_boundary)
    bottom_intersection = np.sum(mask * bottom_boundary)
    left_intersection = np.sum(mask * left_boundary)
    right_intersection = np.sum(mask * right_boundary)

    # Total boundary truncation
    total_truncation = top_intersection + bottom_intersection + left_intersection + right_intersection

    return total_truncation >= 10, scale >= scale_threshold

def get_maximum_height(binary_mask):
    # Find rows containing the object
    rows = np.any(binary_mask, axis=1)  # Check for any non-zero pixels in each row
    non_zero_row_indices = np.where(rows)[0]
    if non_zero_row_indices.size == 0:
        return 0  # No object found
    max_height = non_zero_row_indices[-1] - non_zero_row_indices[0] + 1
    return max_height

def read_bounding_boxes_segmentations(annotations_path_or_list, image_size):
    """
    Reads bounding box and segmentation mask data and returns a list of bounding boxes.

    Args:
        annotations_path_or_list: Either a path to a JSON file or a list of annotation dicts.
        image_size: (width, height) tuple.
    """
    if isinstance(annotations_path_or_list, (str, os.PathLike)):
        with open(annotations_path_or_list, 'r') as file:
            annotations = json.load(file)
    else:
        annotations = annotations_path_or_list

    # Extract bounding boxes from the annotations
    bboxes = []
    segmentation_mask = []
    category_ids = []
    for index, annotation in enumerate(annotations):
        if annotation["iscrowd"]:
            print("Skip crowd annotation")
            continue
        if "segmentation" in annotation:
            seg = annotation["segmentation"]
            # Check if segmentation is RLE format (dict with 'counts') or polygon format (list)
            if isinstance(seg, dict) and 'counts' in seg:
                # RLE format from COCONUT - make a copy to avoid modifying original
                rle = {'size': seg['size'], 'counts': seg['counts']}
                if isinstance(rle['counts'], str):
                    rle['counts'] = rle['counts'].encode('utf-8')
                mask = mask_utils.decode(rle).astype(bool)
                rows = np.any(mask, axis=1)
                height = np.sum(rows)
            else:
                # Polygon format
                mask, height = create_boolean_mask_from_polygon(image_size, seg)

            is_truncated, is_scaleable = analyze_mask(mask, image_size)
            if (height/image_size[1] > 0.0625 and not is_truncated and is_scaleable): # object must be 6.25% of the orginal image height
                segmentation_mask.append(mask)
                category_ids.append(annotation["category_id"])
                bboxes.append(annotation["bbox"])  # Add the bbox to the list
            else:
                print("Too small segmentation")
        
    return bboxes, np.array(segmentation_mask), np.arange(len(segmentation_mask)), replace_categories_with_supercategories(category_ids)



def create_boolean_mask_from_polygon(image_shape, segmentation):
    """
    Create a boolean mask for the image where pixels within the polygon are True, and the rest are False.
    """
    # Create an empty black mask

    reversed_size = (image_shape[1], image_shape[0])

    mask = np.zeros(reversed_size[:2], dtype=np.uint8)
    
    if isinstance(segmentation, list):  # Polygon format
        # Create a mask for each object
        for polygon in segmentation:
            points = np.array(polygon).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [points], color=1)

    elif isinstance(segmentation, dict):  # RLE format
        rle = segmentation
        height = image_shape[0]  # Number of rows in the image
        width = image_shape[1] 
        if isinstance(rle['counts'], list):  # Uncompressed RLE
            rle = mask_utils.frPyObjects([rle], height, width)
        object_mask = mask_utils.decode(rle)
        object_mask = np.squeeze(object_mask)  # Removes singleton dimensions
        mask = np.maximum(mask, object_mask)

    # Convert the mask to a boolean array
    boolean_mask = mask.astype(bool)
    
    return boolean_mask, get_maximum_height(boolean_mask)


# COCO/COCONUT category ID to name mapping
COCO_CATEGORIES = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
    22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
    28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
    35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
    40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
    44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife',
    50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich',
    55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza',
    60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant',
    65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop',
    74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave',
    79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
    85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier',
    90: 'toothbrush',
    # Stuff categories (isthing=0)
    92: 'banner', 93: 'blanket', 95: 'bridge', 100: 'cardboard', 107: 'counter',
    109: 'curtain', 112: 'door-stuff', 118: 'floor-wood', 119: 'flower',
    122: 'fruit', 125: 'gravel', 128: 'house', 130: 'light', 133: 'mirror-stuff',
    138: 'net', 141: 'pillow', 144: 'platform', 145: 'playingfield', 147: 'railroad',
    148: 'river', 149: 'road', 151: 'roof', 154: 'sand', 155: 'sea', 156: 'shelf',
    159: 'snow', 161: 'stairs', 166: 'tent', 168: 'towel', 171: 'wall-brick',
    175: 'wall-stone', 176: 'wall-tile', 177: 'wall-wood', 178: 'water-other',
    180: 'window-blind', 181: 'window-other', 184: 'tree-merged', 185: 'fence-merged',
    186: 'ceiling-merged', 187: 'sky-other-merged', 188: 'cabinet-merged',
    189: 'table-merged', 190: 'floor-other-merged', 191: 'pavement-merged',
    192: 'mountain-merged', 193: 'grass-merged', 194: 'dirt-merged', 195: 'paper-merged',
    196: 'food-other-merged', 197: 'building-other-merged', 198: 'rock-merged',
    199: 'wall-other-merged', 200: 'rug-merged',
}


def replace_categories_with_supercategories(category_ids, json_file_path=None):
    """Map category IDs to category names using built-in COCO/COCONUT mapping."""
    updated_categories = []
    for category_id in category_ids:
        if category_id in COCO_CATEGORIES:
            updated_categories.append(COCO_CATEGORIES[category_id])
        else:
            updated_categories.append("unknown")
    return updated_categories

def align_to_depth_match(mask, depth_map, object_name, project_root, model):
    from matching.process_image_space import process_object
    # Get rendered depth and mask from process_object 
    # TODO: may need to consider the occlusion later
    R, T, image_render, depth = process_object(object_name, project_root, model)
    # Get mask from alpha channel of rendered image
    render_mask = image_render[..., -1] > 0
    
    # Get overlapping region between input mask and rendered mask
    overlap_mask = mask & render_mask
    
    if not overlap_mask.any():
        print("No overlap between masks found")
        return np.eye(4)
        
    # Get depth values in overlapping region
    depth_map_values = depth_map[overlap_mask]
    depth_render_values = depth[overlap_mask]
    
    # Calculate scale factor between the two depth maps
    # Using median ratio to be robust to outliers
    depth_ratios = depth_map_values / depth_render_values
    scale = np.median(depth_ratios)
    
    # Create transformation matrix with scale
    transform = np.eye(4)

    transform[:3, :3] = np.linalg.inv(R[:3, :3]) * scale
    transform[:3, -1] = T[:3] * scale
    
    return transform