import json
import numpy as np
from cubercnn import data
from detectron2.structures import BoxMode

def get_filter_settings():
    return {
        'visibility_thres': 0.33333333,
        'truncation_thres': 0.33333333,
        'min_height_thres': 0.0625,
        'max_depth': 100000000.0,
        'category_names': [],  # Will be set based on category_path
        'ignore_names': ['dontcare', 'ignore', 'void'],
        'trunc_2D_boxes': False,
        'modal_2D_boxes': False,
        'max_height_thres': 1.5,
    }

def is_ignore(anno, filter_settings, image_height):
    ignore = anno['behind_camera'] 
    ignore |= (not bool(anno['valid3D']))

    if ignore:
        return ignore

    ignore |= anno['dimensions'][0] <= 0
    ignore |= anno['dimensions'][1] <= 0
    ignore |= anno['dimensions'][2] <= 0
    ignore |= anno['center_cam'][2] > filter_settings['max_depth']
    ignore |= (anno['lidar_pts'] == 0)
    ignore |= (anno['segmentation_pts'] == 0)
    ignore |= (anno['depth_error'] > 0.5)
    
    # Get 2D bounding box
    bbox2D = get_bbox2D(anno)
    ignore |= bbox2D[3] <= filter_settings['min_height_thres']*image_height
    ignore |= bbox2D[3] >= filter_settings['max_height_thres']*image_height
        
    ignore |= (anno['truncation'] >=0 and anno['truncation'] >= filter_settings['truncation_thres'])
    ignore |= (anno['visibility'] >= 0 and anno['visibility'] <= filter_settings['visibility_thres'])
    
    if 'ignore_names' in filter_settings:
        ignore |= anno['category_name'] in filter_settings['ignore_names']

    return ignore

def get_bbox2D(anno):
    # Priority 1: tightly annotated 2D boxes
    if 'bbox2D_tight' in anno and anno['bbox2D_tight'][0] != -1:
        return BoxMode.convert(anno['bbox2D_tight'], BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    
    # Priority 2: truncated projected 2D boxes
    elif 'bbox2D_trunc' in anno and not np.all([val==-1 for val in anno['bbox2D_trunc']]):
        return BoxMode.convert(anno['bbox2D_trunc'], BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    
    # Priority 3: projected 3D --> 2D box
    elif 'bbox2D_proj' in anno:
        return BoxMode.convert(anno['bbox2D_proj'], BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    
    # Priority 4: default bbox
    else:
        return anno['bbox']

def convert_coco3d_to_2d(coco3d_path, output_path):
    # Load COCO3D dataset
    dataset = data.Omni3D([coco3d_path])
    imgIds = dataset.getImgIds()
    imgs = dataset.loadImgs(imgIds)
    
    # Get filter settings
    filter_settings = get_filter_settings()
    
    # Load category mapping
    with open("configs/category_meta.json", "r") as f:
        meta = json.load(f)
    id_mapping = meta['thing_dataset_id_to_contiguous_id']
    
    # Initialize output data structure
    output_data = []
    
    # Process each image
    for img in imgs:
        # Get annotations for current image
        annIds = dataset.getAnnIds(imgIds=img['id'])
        anns = dataset.loadAnns(annIds)
        
        # Create image entry
        image_entry = {
            "image_id": img['id'],
            "K": img['K'],
            "width": img['width'],
            "height": img['height'],
            "instances": []
        }
        
        # Process each annotation
        for ann in anns:
            # Skip ignored annotations
            if is_ignore(ann, filter_settings, img['height']):
                continue
                
            # Get the appropriate 2D bounding box
            bbox2D = get_bbox2D(ann)
            
            # Map category_id from dataset_id to contiguous_id
            dataset_id = str(ann['category_id'])
            if dataset_id not in id_mapping:
                print(f"Warning: Category ID {dataset_id} not found in mapping. Skipping this annotation.")
                continue
            contiguous_id = id_mapping[dataset_id]
            
            # Create instance entry
            instance = {
                "category_id": contiguous_id,
                "category_name": ann['category_name'],
                "bbox": bbox2D,
                "score": 1.0  # Default score since original data doesn't have confidence scores
            }
            image_entry["instances"].append(instance)
        
        # Always add the image entry, even if it has no instances
        output_data.append(image_entry)
    
    # Save to output file
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco3d_path", type=str, default="datasets/Omni3D/COCO3D_val.json")
    parser.add_argument("--output_path", type=str, default="datasets/Omni3D/gt_coco3d_base_oracle_2d.json")
    args = parser.parse_args()
    convert_coco3d_to_2d(args.coco3d_path, args.output_path)