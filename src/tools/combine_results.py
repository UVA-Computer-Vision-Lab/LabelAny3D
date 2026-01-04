"""
Combine per-scene 3D bbox results into a single Omni3D format JSON file.

Usage:
    python tools/combine_results.py --split val
    python tools/combine_results.py --split train --results_dir ../experimental_results/COCO
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment


# COCO categories with Omni3D-style IDs
COCO_CATEGORIES = [
    {'supercategory': 'person', 'id': 7, 'name': 'person'},
    {'supercategory': 'vehicle', 'id': 11, 'name': 'bicycle'},
    {'supercategory': 'vehicle', 'id': 1, 'name': 'car'},
    {'supercategory': 'vehicle', 'id': 10, 'name': 'motorcycle'},
    {'supercategory': 'vehicle', 'id': 98, 'name': 'airplane'},
    {'supercategory': 'vehicle', 'id': 12, 'name': 'bus'},
    {'supercategory': 'vehicle', 'id': 99, 'name': 'train'},
    {'supercategory': 'vehicle', 'id': 5, 'name': 'truck'},
    {'supercategory': 'vehicle', 'id': 100, 'name': 'boat'},
    {'supercategory': 'outdoor', 'id': 101, 'name': 'traffic light'},
    {'supercategory': 'outdoor', 'id': 102, 'name': 'fire hydrant'},
    {'supercategory': 'outdoor', 'id': 103, 'name': 'stop sign'},
    {'supercategory': 'outdoor', 'id': 104, 'name': 'parking meter'},
    {'supercategory': 'outdoor', 'id': 105, 'name': 'bench'},
    {'supercategory': 'animal', 'id': 106, 'name': 'bird'},
    {'supercategory': 'animal', 'id': 107, 'name': 'cat'},
    {'supercategory': 'animal', 'id': 108, 'name': 'dog'},
    {'supercategory': 'animal', 'id': 109, 'name': 'horse'},
    {'supercategory': 'animal', 'id': 110, 'name': 'sheep'},
    {'supercategory': 'animal', 'id': 111, 'name': 'cow'},
    {'supercategory': 'animal', 'id': 112, 'name': 'elephant'},
    {'supercategory': 'animal', 'id': 113, 'name': 'bear'},
    {'supercategory': 'animal', 'id': 114, 'name': 'zebra'},
    {'supercategory': 'animal', 'id': 115, 'name': 'giraffe'},
    {'supercategory': 'accessory', 'id': 116, 'name': 'backpack'},
    {'supercategory': 'accessory', 'id': 117, 'name': 'umbrella'},
    {'supercategory': 'accessory', 'id': 118, 'name': 'handbag'},
    {'supercategory': 'accessory', 'id': 119, 'name': 'tie'},
    {'supercategory': 'accessory', 'id': 120, 'name': 'suitcase'},
    {'supercategory': 'sports', 'id': 121, 'name': 'frisbee'},
    {'supercategory': 'sports', 'id': 122, 'name': 'skis'},
    {'supercategory': 'sports', 'id': 123, 'name': 'snowboard'},
    {'supercategory': 'sports', 'id': 124, 'name': 'sports ball'},
    {'supercategory': 'sports', 'id': 125, 'name': 'kite'},
    {'supercategory': 'sports', 'id': 126, 'name': 'baseball bat'},
    {'supercategory': 'sports', 'id': 127, 'name': 'baseball glove'},
    {'supercategory': 'sports', 'id': 128, 'name': 'skateboard'},
    {'supercategory': 'sports', 'id': 129, 'name': 'surfboard'},
    {'supercategory': 'sports', 'id': 130, 'name': 'tennis racket'},
    {'supercategory': 'kitchen', 'id': 15, 'name': 'bottle'},
    {'supercategory': 'kitchen', 'id': 131, 'name': 'wine glass'},
    {'supercategory': 'kitchen', 'id': 19, 'name': 'cup'},
    {'supercategory': 'kitchen', 'id': 132, 'name': 'fork'},
    {'supercategory': 'kitchen', 'id': 133, 'name': 'knife'},
    {'supercategory': 'kitchen', 'id': 134, 'name': 'spoon'},
    {'supercategory': 'kitchen', 'id': 56, 'name': 'bowl'},
    {'supercategory': 'food', 'id': 135, 'name': 'banana'},
    {'supercategory': 'food', 'id': 136, 'name': 'apple'},
    {'supercategory': 'food', 'id': 137, 'name': 'sandwich'},
    {'supercategory': 'food', 'id': 138, 'name': 'orange'},
    {'supercategory': 'food', 'id': 139, 'name': 'broccoli'},
    {'supercategory': 'food', 'id': 140, 'name': 'carrot'},
    {'supercategory': 'food', 'id': 141, 'name': 'hot dog'},
    {'supercategory': 'food', 'id': 142, 'name': 'pizza'},
    {'supercategory': 'food', 'id': 143, 'name': 'donut'},
    {'supercategory': 'food', 'id': 144, 'name': 'cake'},
    {'supercategory': 'furniture', 'id': 18, 'name': 'chair'},
    {'supercategory': 'furniture', 'id': 145, 'name': 'couch'},
    {'supercategory': 'furniture', 'id': 73, 'name': 'potted plant'},
    {'supercategory': 'furniture', 'id': 39, 'name': 'bed'},
    {'supercategory': 'furniture', 'id': 146, 'name': 'dining table'},
    {'supercategory': 'furniture', 'id': 32, 'name': 'toilet'},
    {'supercategory': 'electronic', 'id': 147, 'name': 'tv'},
    {'supercategory': 'electronic', 'id': 20, 'name': 'laptop'},
    {'supercategory': 'electronic', 'id': 81, 'name': 'mouse'},
    {'supercategory': 'electronic', 'id': 95, 'name': 'remote'},
    {'supercategory': 'electronic', 'id': 77, 'name': 'keyboard'},
    {'supercategory': 'electronic', 'id': 148, 'name': 'cell phone'},
    {'supercategory': 'appliance', 'id': 54, 'name': 'microwave'},
    {'supercategory': 'appliance', 'id': 57, 'name': 'oven'},
    {'supercategory': 'appliance', 'id': 72, 'name': 'toaster'},
    {'supercategory': 'appliance', 'id': 28, 'name': 'sink'},
    {'supercategory': 'appliance', 'id': 49, 'name': 'refrigerator'},
    {'supercategory': 'indoor', 'id': 149, 'name': 'book'},
    {'supercategory': 'indoor', 'id': 87, 'name': 'clock'},
    {'supercategory': 'indoor', 'id': 58, 'name': 'vase'},
    {'supercategory': 'indoor', 'id': 150, 'name': 'scissors'},
    {'supercategory': 'indoor', 'id': 151, 'name': 'teddy bear'},
    {'supercategory': 'indoor', 'id': 152, 'name': 'hair drier'},
    {'supercategory': 'indoor', 'id': 153, 'name': 'toothbrush'},
]

# Build name to id mapping
CATEGORY_NAME_TO_ID = {cat["name"]: cat["id"] for cat in COCO_CATEGORIES}


def project_to_2d(point_3d, K):
    """Project 3D point to 2D using camera intrinsics."""
    point_2d_homogeneous = np.dot(K, point_3d)
    return point_2d_homogeneous[:2] / point_2d_homogeneous[2]


def iou2D(box1, box2):
    """Calculate IoU between two boxes in xyxy format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection / (area1 + area2 - intersection + 1e-6)
    return iou


def hungarian_matching(boxes0, boxes1):
    """Perform Hungarian matching based on IoU for two sets of bounding boxes."""
    n0, n1 = len(boxes0), len(boxes1)

    # Create cost matrix (negative IoU to convert max problem to min problem)
    cost_matrix = np.zeros((n0, n1))
    for i, box0 in enumerate(boxes0):
        for j, box1 in enumerate(boxes1):
            cost_matrix[i, j] = -iou2D(box0, box1)

    # Perform Hungarian matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Create matching pairs: (index0, index1, IoU)
    matches = []
    for i, j in zip(row_ind, col_ind):
        matches.append((i, j, -cost_matrix[i, j]))

    return matches


def combine_coco_results(results_dir, split, output_path, bbox_filename="3dbbox.json"):
    """
    Combine per-scene 3D bbox results into Omni3D format JSON.

    Args:
        results_dir: Path to experimental_results/COCO
        split: 'train' or 'val'
        output_path: Output JSON file path
        bbox_filename: Name of the 3D bbox JSON file (default: 3dbbox.json)
    """
    scene_dir = os.path.join(results_dir, split)

    if not os.path.exists(scene_dir):
        raise FileNotFoundError(f"Results directory not found: {scene_dir}")

    # Get all scene folders
    scene_ids = sorted([d for d in os.listdir(scene_dir) if os.path.isdir(os.path.join(scene_dir, d))])
    print(f"Found {len(scene_ids)} scenes in {scene_dir}")

    # Dataset IDs (matching notebook convention)
    dataset_id = 22 if split == "val" else 23
    image_id_start = 1000000 if split == "val" else 2000000
    annotation_id_start = 100000000 if split == "val" else 200000000

    images = []
    annotations = []
    image_id = image_id_start
    annotation_id = annotation_id_start

    for scene_name in tqdm(scene_ids, desc="Processing scenes"):
        scene_path = os.path.join(scene_dir, scene_name)
        bbox_path = os.path.join(scene_path, bbox_filename)
        cam_path = os.path.join(scene_path, "cam_params.json")
        bbox2d_path = os.path.join(scene_path, "bboxes.json")

        # Skip if required files don't exist
        if not os.path.exists(bbox_path):
            print(f"Warning: Missing {bbox_filename} in {scene_name}, skipping")
            continue
        if not os.path.exists(cam_path):
            print(f"Warning: Missing cam_params.json in {scene_name}, skipping")
            continue

        # Load camera parameters
        with open(cam_path, 'r') as f:
            cam_params = json.load(f)
        K = np.array(cam_params["K"])
        H, W = cam_params["H"], cam_params["W"]

        # Create image entry
        image_dict = {
            "width": int(W),
            "height": int(H),
            "file_path": f"coco/images/{split}2017/{scene_name}.jpg",
            "K": K.tolist(),
            "src_90_rotate": 0,
            "src_flagged": False,
            "incomplete": False,
            "id": image_id,
            "dataset_id": dataset_id,
        }

        # Load 3D bbox predictions
        with open(bbox_path, 'r') as f:
            bbox_anno = json.load(f)

        if len(bbox_anno) == 0:
            print(f"Warning: Empty bbox in {scene_name}, skipping")
            continue

        # Load 2D bboxes if available (for Hungarian matching)
        bbox2d_anno = None
        if os.path.exists(bbox2d_path):
            with open(bbox2d_path, 'r') as f:
                bbox2d_anno = json.load(f)
        else:
            print(f"Warning: Missing bboxes.json in {scene_name}, using projected bbox as bbox2D_tight")

        images.append(image_dict)

        # Process annotations
        local_annotations = []
        for anno in bbox_anno:
            category_name = anno.get("category_name", "").replace("_", " ")
            category_id = CATEGORY_NAME_TO_ID.get(category_name, -1)

            if category_id == -1:
                print(f"Warning: Unknown category '{category_name}' in {scene_name}, skipping")
                continue

            # Project 3D bbox corners to 2D
            corners = np.array(anno["bbox3D_cam"])
            points_2d = [project_to_2d(np.array(point), K) for point in corners]

            min_x = min(p[0] for p in points_2d)
            min_y = min(p[1] for p in points_2d)
            max_x = max(p[0] for p in points_2d)
            max_y = max(p[1] for p in points_2d)

            bbox2D_proj = [min_x, min_y, max_x, max_y]
            bbox2D_trunc = [
                max(0, min_x),
                max(0, min_y),
                min(W, max_x),
                min(H, max_y),
            ]

            new_anno = {
                "behind_camera": False,
                "truncation": 0.0,
                "visibility": 1,
                "segmentation_pts": -1,
                "lidar_pts": -1,
                "valid3D": True,
                "category_name": category_name,
                "category_id": category_id,
                "image_id": image_id,
                "id": annotation_id,
                "dataset_id": dataset_id,
                "center_cam": anno.get("center_cam"),
                "dimensions": anno.get("dimensions"),
                "R_cam": anno.get("R_cam"),
                "bbox3D_cam": anno.get("bbox3D_cam"),
                "bbox2D_proj": bbox2D_proj,
                "bbox2D_trunc": bbox2D_trunc,
                "depth_error": -1,
            }
            local_annotations.append(new_anno)
            annotation_id += 1

        # Hungarian matching to get bbox2D_tight
        if bbox2d_anno is not None and len(local_annotations) > 0 and len(bbox2d_anno) > 0:
            truncated_boxes = np.array([anno["bbox2D_trunc"] for anno in local_annotations])
            matches = hungarian_matching(truncated_boxes, np.array(bbox2d_anno))
            for match in matches:
                local_annotations[match[0]]["bbox2D_tight"] = bbox2d_anno[match[1]]
        else:
            # Fallback: use projected bbox as tight bbox
            for anno in local_annotations:
                anno["bbox2D_tight"] = anno["bbox2D_trunc"]

        annotations.extend(local_annotations)
        image_id += 1

    # Build output JSON
    output = {
        "info": {
            "id": dataset_id,
            "source": "COCO",
            "name": f"COCO {'Validation' if split == 'val' else 'Train'}",
            "split": split.capitalize(),
            "version": "0.1",
            "url": "https://cocodataset.org/#home",
        },
        "categories": COCO_CATEGORIES,
        "images": images,
        "annotations": annotations,
    }

    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f)

    print(f"Saved {len(images)} images, {len(annotations)} annotations to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine COCO 3D bbox results into Omni3D format")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], help="Dataset split")
    parser.add_argument("--results_dir", type=str, default="../experimental_results/COCO", help="Results directory")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--bbox_file", type=str, default="3dbbox.json", help="3D bbox JSON filename")

    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.results_dir, f"COCO3D_{args.split}.json")

    combine_coco_results(args.results_dir, args.split, args.output, args.bbox_file)
