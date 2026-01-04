#!/usr/bin/env python3
"""
Download COCONUT annotations from Hugging Face and convert to instance segmentation format.
Based on: https://github.com/bytedance/coconut_cvpr2024

This outputs the same format as create_instance_by_panseg.py from the original COCONUT repo.

Usage:
    python download_coconut.py --output_dir ../dataset/coco/annotations
    python download_coconut.py --split coconut_s --output_dir ../dataset/coco/annotations
"""

import argparse
import json
import os
import numpy as np
from tqdm import tqdm
from itertools import groupby
from skimage import measure
from datasets import load_dataset
from pycocotools import mask as mask_utils


CATEGORIES = [
    {'supercategory': 'person', 'isthing': 1, 'id': 1, 'name': 'person'},
    {'supercategory': 'vehicle', 'isthing': 1, 'id': 2, 'name': 'bicycle'},
    {'supercategory': 'vehicle', 'isthing': 1, 'id': 3, 'name': 'car'},
    {'supercategory': 'vehicle', 'isthing': 1, 'id': 4, 'name': 'motorcycle'},
    {'supercategory': 'vehicle', 'isthing': 1, 'id': 5, 'name': 'airplane'},
    {'supercategory': 'vehicle', 'isthing': 1, 'id': 6, 'name': 'bus'},
    {'supercategory': 'vehicle', 'isthing': 1, 'id': 7, 'name': 'train'},
    {'supercategory': 'vehicle', 'isthing': 1, 'id': 8, 'name': 'truck'},
    {'supercategory': 'vehicle', 'isthing': 1, 'id': 9, 'name': 'boat'},
    {'supercategory': 'outdoor', 'isthing': 1, 'id': 10, 'name': 'traffic light'},
    {'supercategory': 'outdoor', 'isthing': 1, 'id': 11, 'name': 'fire hydrant'},
    {'supercategory': 'outdoor', 'isthing': 1, 'id': 13, 'name': 'stop sign'},
    {'supercategory': 'outdoor', 'isthing': 1, 'id': 14, 'name': 'parking meter'},
    {'supercategory': 'outdoor', 'isthing': 1, 'id': 15, 'name': 'bench'},
    {'supercategory': 'animal', 'isthing': 1, 'id': 16, 'name': 'bird'},
    {'supercategory': 'animal', 'isthing': 1, 'id': 17, 'name': 'cat'},
    {'supercategory': 'animal', 'isthing': 1, 'id': 18, 'name': 'dog'},
    {'supercategory': 'animal', 'isthing': 1, 'id': 19, 'name': 'horse'},
    {'supercategory': 'animal', 'isthing': 1, 'id': 20, 'name': 'sheep'},
    {'supercategory': 'animal', 'isthing': 1, 'id': 21, 'name': 'cow'},
    {'supercategory': 'animal', 'isthing': 1, 'id': 22, 'name': 'elephant'},
    {'supercategory': 'animal', 'isthing': 1, 'id': 23, 'name': 'bear'},
    {'supercategory': 'animal', 'isthing': 1, 'id': 24, 'name': 'zebra'},
    {'supercategory': 'animal', 'isthing': 1, 'id': 25, 'name': 'giraffe'},
    {'supercategory': 'accessory', 'isthing': 1, 'id': 27, 'name': 'backpack'},
    {'supercategory': 'accessory', 'isthing': 1, 'id': 28, 'name': 'umbrella'},
    {'supercategory': 'accessory', 'isthing': 1, 'id': 31, 'name': 'handbag'},
    {'supercategory': 'accessory', 'isthing': 1, 'id': 32, 'name': 'tie'},
    {'supercategory': 'accessory', 'isthing': 1, 'id': 33, 'name': 'suitcase'},
    {'supercategory': 'sports', 'isthing': 1, 'id': 34, 'name': 'frisbee'},
    {'supercategory': 'sports', 'isthing': 1, 'id': 35, 'name': 'skis'},
    {'supercategory': 'sports', 'isthing': 1, 'id': 36, 'name': 'snowboard'},
    {'supercategory': 'sports', 'isthing': 1, 'id': 37, 'name': 'sports ball'},
    {'supercategory': 'sports', 'isthing': 1, 'id': 38, 'name': 'kite'},
    {'supercategory': 'sports', 'isthing': 1, 'id': 39, 'name': 'baseball bat'},
    {'supercategory': 'sports', 'isthing': 1, 'id': 40, 'name': 'baseball glove'},
    {'supercategory': 'sports', 'isthing': 1, 'id': 41, 'name': 'skateboard'},
    {'supercategory': 'sports', 'isthing': 1, 'id': 42, 'name': 'surfboard'},
    {'supercategory': 'sports', 'isthing': 1, 'id': 43, 'name': 'tennis racket'},
    {'supercategory': 'kitchen', 'isthing': 1, 'id': 44, 'name': 'bottle'},
    {'supercategory': 'kitchen', 'isthing': 1, 'id': 46, 'name': 'wine glass'},
    {'supercategory': 'kitchen', 'isthing': 1, 'id': 47, 'name': 'cup'},
    {'supercategory': 'kitchen', 'isthing': 1, 'id': 48, 'name': 'fork'},
    {'supercategory': 'kitchen', 'isthing': 1, 'id': 49, 'name': 'knife'},
    {'supercategory': 'kitchen', 'isthing': 1, 'id': 50, 'name': 'spoon'},
    {'supercategory': 'kitchen', 'isthing': 1, 'id': 51, 'name': 'bowl'},
    {'supercategory': 'food', 'isthing': 1, 'id': 52, 'name': 'banana'},
    {'supercategory': 'food', 'isthing': 1, 'id': 53, 'name': 'apple'},
    {'supercategory': 'food', 'isthing': 1, 'id': 54, 'name': 'sandwich'},
    {'supercategory': 'food', 'isthing': 1, 'id': 55, 'name': 'orange'},
    {'supercategory': 'food', 'isthing': 1, 'id': 56, 'name': 'broccoli'},
    {'supercategory': 'food', 'isthing': 1, 'id': 57, 'name': 'carrot'},
    {'supercategory': 'food', 'isthing': 1, 'id': 58, 'name': 'hot dog'},
    {'supercategory': 'food', 'isthing': 1, 'id': 59, 'name': 'pizza'},
    {'supercategory': 'food', 'isthing': 1, 'id': 60, 'name': 'donut'},
    {'supercategory': 'food', 'isthing': 1, 'id': 61, 'name': 'cake'},
    {'supercategory': 'furniture', 'isthing': 1, 'id': 62, 'name': 'chair'},
    {'supercategory': 'furniture', 'isthing': 1, 'id': 63, 'name': 'couch'},
    {'supercategory': 'furniture', 'isthing': 1, 'id': 64, 'name': 'potted plant'},
    {'supercategory': 'furniture', 'isthing': 1, 'id': 65, 'name': 'bed'},
    {'supercategory': 'furniture', 'isthing': 1, 'id': 67, 'name': 'dining table'},
    {'supercategory': 'furniture', 'isthing': 1, 'id': 70, 'name': 'toilet'},
    {'supercategory': 'electronic', 'isthing': 1, 'id': 72, 'name': 'tv'},
    {'supercategory': 'electronic', 'isthing': 1, 'id': 73, 'name': 'laptop'},
    {'supercategory': 'electronic', 'isthing': 1, 'id': 74, 'name': 'mouse'},
    {'supercategory': 'electronic', 'isthing': 1, 'id': 75, 'name': 'remote'},
    {'supercategory': 'electronic', 'isthing': 1, 'id': 76, 'name': 'keyboard'},
    {'supercategory': 'electronic', 'isthing': 1, 'id': 77, 'name': 'cell phone'},
    {'supercategory': 'appliance', 'isthing': 1, 'id': 78, 'name': 'microwave'},
    {'supercategory': 'appliance', 'isthing': 1, 'id': 79, 'name': 'oven'},
    {'supercategory': 'appliance', 'isthing': 1, 'id': 80, 'name': 'toaster'},
    {'supercategory': 'appliance', 'isthing': 1, 'id': 81, 'name': 'sink'},
    {'supercategory': 'appliance', 'isthing': 1, 'id': 82, 'name': 'refrigerator'},
    {'supercategory': 'indoor', 'isthing': 1, 'id': 84, 'name': 'book'},
    {'supercategory': 'indoor', 'isthing': 1, 'id': 85, 'name': 'clock'},
    {'supercategory': 'indoor', 'isthing': 1, 'id': 86, 'name': 'vase'},
    {'supercategory': 'indoor', 'isthing': 1, 'id': 87, 'name': 'scissors'},
    {'supercategory': 'indoor', 'isthing': 1, 'id': 88, 'name': 'teddy bear'},
    {'supercategory': 'indoor', 'isthing': 1, 'id': 89, 'name': 'hair drier'},
    {'supercategory': 'indoor', 'isthing': 1, 'id': 90, 'name': 'toothbrush'},
    {'supercategory': 'textile', 'isthing': 0, 'id': 92, 'name': 'banner'},
    {'supercategory': 'textile', 'isthing': 0, 'id': 93, 'name': 'blanket'},
    {'supercategory': 'building', 'isthing': 0, 'id': 95, 'name': 'bridge'},
    {'supercategory': 'raw-material', 'isthing': 0, 'id': 100, 'name': 'cardboard'},
    {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 107, 'name': 'counter'},
    {'supercategory': 'textile', 'isthing': 0, 'id': 109, 'name': 'curtain'},
    {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 112, 'name': 'door-stuff'},
    {'supercategory': 'floor', 'isthing': 0, 'id': 118, 'name': 'floor-wood'},
    {'supercategory': 'plant', 'isthing': 0, 'id': 119, 'name': 'flower'},
    {'supercategory': 'food-stuff', 'isthing': 0, 'id': 122, 'name': 'fruit'},
    {'supercategory': 'ground', 'isthing': 0, 'id': 125, 'name': 'gravel'},
    {'supercategory': 'building', 'isthing': 0, 'id': 128, 'name': 'house'},
    {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 130, 'name': 'light'},
    {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 133, 'name': 'mirror-stuff'},
    {'supercategory': 'structural', 'isthing': 0, 'id': 138, 'name': 'net'},
    {'supercategory': 'textile', 'isthing': 0, 'id': 141, 'name': 'pillow'},
    {'supercategory': 'ground', 'isthing': 0, 'id': 144, 'name': 'platform'},
    {'supercategory': 'ground', 'isthing': 0, 'id': 145, 'name': 'playingfield'},
    {'supercategory': 'ground', 'isthing': 0, 'id': 147, 'name': 'railroad'},
    {'supercategory': 'water', 'isthing': 0, 'id': 148, 'name': 'river'},
    {'supercategory': 'ground', 'isthing': 0, 'id': 149, 'name': 'road'},
    {'supercategory': 'building', 'isthing': 0, 'id': 151, 'name': 'roof'},
    {'supercategory': 'ground', 'isthing': 0, 'id': 154, 'name': 'sand'},
    {'supercategory': 'water', 'isthing': 0, 'id': 155, 'name': 'sea'},
    {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 156, 'name': 'shelf'},
    {'supercategory': 'ground', 'isthing': 0, 'id': 159, 'name': 'snow'},
    {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 161, 'name': 'stairs'},
    {'supercategory': 'building', 'isthing': 0, 'id': 166, 'name': 'tent'},
    {'supercategory': 'textile', 'isthing': 0, 'id': 168, 'name': 'towel'},
    {'supercategory': 'wall', 'isthing': 0, 'id': 171, 'name': 'wall-brick'},
    {'supercategory': 'wall', 'isthing': 0, 'id': 175, 'name': 'wall-stone'},
    {'supercategory': 'wall', 'isthing': 0, 'id': 176, 'name': 'wall-tile'},
    {'supercategory': 'wall', 'isthing': 0, 'id': 177, 'name': 'wall-wood'},
    {'supercategory': 'water', 'isthing': 0, 'id': 178, 'name': 'water-other'},
    {'supercategory': 'window', 'isthing': 0, 'id': 180, 'name': 'window-blind'},
    {'supercategory': 'window', 'isthing': 0, 'id': 181, 'name': 'window-other'},
    {'supercategory': 'plant', 'isthing': 0, 'id': 184, 'name': 'tree-merged'},
    {'supercategory': 'structural', 'isthing': 0, 'id': 185, 'name': 'fence-merged'},
    {'supercategory': 'ceiling', 'isthing': 0, 'id': 186, 'name': 'ceiling-merged'},
    {'supercategory': 'sky', 'isthing': 0, 'id': 187, 'name': 'sky-other-merged'},
    {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 188, 'name': 'cabinet-merged'},
    {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 189, 'name': 'table-merged'},
    {'supercategory': 'floor', 'isthing': 0, 'id': 190, 'name': 'floor-other-merged'},
    {'supercategory': 'ground', 'isthing': 0, 'id': 191, 'name': 'pavement-merged'},
    {'supercategory': 'solid', 'isthing': 0, 'id': 192, 'name': 'mountain-merged'},
    {'supercategory': 'plant', 'isthing': 0, 'id': 193, 'name': 'grass-merged'},
    {'supercategory': 'ground', 'isthing': 0, 'id': 194, 'name': 'dirt-merged'},
    {'supercategory': 'raw-material', 'isthing': 0, 'id': 195, 'name': 'paper-merged'},
    {'supercategory': 'food-stuff', 'isthing': 0, 'id': 196, 'name': 'food-other-merged'},
    {'supercategory': 'building', 'isthing': 0, 'id': 197, 'name': 'building-other-merged'},
    {'supercategory': 'solid', 'isthing': 0, 'id': 198, 'name': 'rock-merged'},
    {'supercategory': 'wall', 'isthing': 0, 'id': 199, 'name': 'wall-other-merged'},
    {'supercategory': 'textile', 'isthing': 0, 'id': 200, 'name': 'rug-merged'},
]


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_to_rle(binary_mask):
    """Convert binary mask to uncompressed RLE format."""
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


def binary_mask_to_polygon(binary_mask, tolerance=2):
    """Convert binary mask to COCO polygon format."""
    polygons = []
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)

    if len(contours) == 1:
        contours = np.subtract(contours, 1)
    else:
        contours = [np.subtract(contour, 1) for contour in contours]

    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons


def download_coconut(split: str, output_dir: str, save_masks: bool = False):
    """Download COCONUT annotations and convert to instance segmentation format."""

    os.makedirs(output_dir, exist_ok=True)

    # Map split names to output filenames
    output_filenames = {
        "relabeled_coco_val": "coconut_val.json",
        "coconut_s": "coconut_train.json",
        "coconut_b": "coconut_train_b.json",
    }

    if split not in output_filenames:
        raise ValueError(f"Unknown split: {split}. Choose from {list(output_filenames.keys())}")

    output_filename = output_filenames[split]
    output_json_file = os.path.join(output_dir, output_filename)

    # Dataset name format: xdeng77/{split}
    dataset_name = f"xdeng77/{split}"
    print(f"Downloading {split} from {dataset_name}...")
    dataset = load_dataset(dataset_name)

    # Optionally create mask output folder
    if save_masks:
        output_mask_dir = os.path.join(output_dir, split)
        os.makedirs(output_mask_dir, exist_ok=True)

    output_annotations = []
    output_img_infos = []
    box_id = 0

    print(f"Processing {len(dataset['train'])} images...")
    for item in tqdm(dataset["train"]):
        anno_info = item["segments_info"]
        panoptic_mask = np.array(item['mask'])

        # Convert panoptic mask to segment IDs (RGB -> ID)
        if len(panoptic_mask.shape) == 3:
            panoptic_ids = panoptic_mask[:, :, 0].astype(np.int32) + \
                           panoptic_mask[:, :, 1].astype(np.int32) * 256 + \
                           panoptic_mask[:, :, 2].astype(np.int32) * 256 * 256
        else:
            panoptic_ids = panoptic_mask.astype(np.int32)

        image_id = anno_info['image_id']

        # Process each segment
        for seg_info in anno_info['segments_info']:
            # Skip stuff categories (only process things)
            if not seg_info.get('isthing', 0):
                continue

            box_id += 1
            seg_id = seg_info['id']
            binary_mask = (panoptic_ids == seg_id).astype(np.uint8)

            # Compute area
            area = int(np.count_nonzero(binary_mask))
            if area == 0:
                continue

            # Compute bbox [x, y, width, height]
            rows = np.any(binary_mask, axis=1)
            cols = np.any(binary_mask, axis=0)
            if rows.any() and cols.any():
                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]
                bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
            else:
                continue

            # Convert mask to segmentation format
            if seg_info.get('iscrowd', 0):
                segmentation = binary_mask_to_rle(binary_mask)
            else:
                segmentation = binary_mask_to_polygon(binary_mask)
                if len(segmentation) == 0:
                    continue

            # Create annotation entry (flat format, same as create_instance_by_panseg.py)
            annotation = {
                'image_id': image_id,
                'category_id': seg_info['category_id'],
                'id': box_id,
                'iscrowd': seg_info.get('iscrowd', 0),
                'area': area,
                'segmentation': segmentation,
                'bbox': bbox,
            }
            output_annotations.append(annotation)

        # Optionally save mask PNG
        if save_masks:
            img_id = anno_info['file_name'].split('.')[0]
            mask_path = os.path.join(output_mask_dir, f"{img_id}.png")
            item['mask'].save(mask_path)

        output_img_infos.append(item['image_info'])

    # Build output JSON (same format as create_instance_by_panseg.py)
    output_json = {
        'images': output_img_infos,
        'annotations': output_annotations,  # Flat list of annotations
        'categories': CATEGORIES,
        'licenses': [
            {'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/', 'id': 1, 'name': 'Attribution-NonCommercial-ShareAlike License'},
            {'url': 'http://creativecommons.org/licenses/by-nc/2.0/', 'id': 2, 'name': 'Attribution-NonCommercial License'},
            {'url': 'http://creativecommons.org/licenses/by-nc-nd/2.0/', 'id': 3, 'name': 'Attribution-NonCommercial-NoDerivs License'},
            {'url': 'http://creativecommons.org/licenses/by/2.0/', 'id': 4, 'name': 'Attribution License'},
            {'url': 'http://creativecommons.org/licenses/by-sa/2.0/', 'id': 5, 'name': 'Attribution-ShareAlike License'},
            {'url': 'http://creativecommons.org/licenses/by-nd/2.0/', 'id': 6, 'name': 'Attribution-NoDerivs License'},
            {'url': 'http://flickr.com/commons/usage/', 'id': 7, 'name': 'No known copyright restrictions'},
            {'url': 'http://www.usa.gov/copyright.shtml', 'id': 8, 'name': 'United States Government Work'},
        ],
        'info': {
            'description': 'COCO 2018 Panoptic Dataset',
            'url': 'http://cocodataset.org',
            'version': '1.0',
            'year': 2018,
            'contributor': 'https://arxiv.org/abs/1801.00868',
            'date_created': '2018-06-01 00:00:00.0',
        },
    }

    with open(output_json_file, 'w') as f:
        json.dump(output_json, f, indent=4)

    print(f"Done! Saved {len(output_annotations)} annotations for {len(output_img_infos)} images to {output_json_file}")
    return output_json_file


def main():
    parser = argparse.ArgumentParser(description="Download COCONUT annotations from Hugging Face")
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["relabeled_coco_val", "coconut_s", "coconut_b", "all"],
        help="Which split to download (default: all = val + train)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../dataset/coco/annotations",
        help="Output directory for annotations",
    )
    parser.add_argument(
        "--save_masks",
        action="store_true",
        help="Also save panoptic mask images",
    )
    args = parser.parse_args()

    if args.split == "all":
        # Download val and train (coconut_s)
        download_coconut("relabeled_coco_val", args.output_dir, args.save_masks)
        download_coconut("coconut_s", args.output_dir, args.save_masks)
    else:
        download_coconut(args.split, args.output_dir, args.save_masks)

    print("Download complete!")


if __name__ == "__main__":
    main()
