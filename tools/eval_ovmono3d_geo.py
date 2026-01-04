import logging
import os
import sys
import torch
import numpy as np
from collections import OrderedDict
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
import detectron2.utils.comm as comm

sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)
from cubercnn.data import (
    get_filter_settings_from_cfg,
    simple_register,
    get_omni3d_categories
)
from cubercnn.evaluation import Omni3DEvaluationHelper
from cubercnn import util

logger = logging.getLogger(__name__)

def setup_categories(category_path):
    """Setup category mapping"""
    metadata = util.load_json(category_path)
    thing_classes = metadata['thing_classes']
    id_map = {int(key):val for key, val in metadata['thing_dataset_id_to_contiguous_id'].items()}
    MetadataCatalog.get('omni3d_model').thing_classes = thing_classes
    MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id = id_map

def evaluate_predictions(
    dataset_names,
    prediction_paths,
    filter_settings,
    output_dir,
    category_path,
    eval_mode="novel",
    iter_label='final'
):
    """
    Evaluate predictions from pre-computed prediction files.
    
    Args:
        dataset_names (list): List of dataset names to evaluate
        prediction_paths (dict): Dictionary mapping dataset names to prediction file paths
        filter_settings (dict): Filter settings for evaluation
        output_dir (str): Output directory for evaluation results
        category_path (str): Path to category metadata json file
        eval_mode (str): Evaluation mode, either "novel" or "base"
        iter_label (str): Label for the iteration being evaluated
    """
    # Setup logging
    os.makedirs(output_dir, exist_ok=True)
    setup_logger(output=output_dir, name="cubercnn")

    # Setup categories
    setup_categories(category_path)

    # Initialize evaluation helper
    thing_classes = ['monitor', 'bag', 'dresser', 'board', 'printer', 'keyboard', 'painting', 'drawers', 'microwave', 'computer', 'kitchen pan', 'potted plant', 'tissues', 'rack', 'tray', 'toys', 'phone', 'podium', 'cart', 'soundsystem', 'fireplace', 'tram']
    filter_settings['category_names'] = thing_classes
    eval_helper = Omni3DEvaluationHelper(
        dataset_names=dataset_names,
        filter_settings=filter_settings,
        output_folder=output_dir,
        iter_label=iter_label,
        only_2d=False,
        eval_categories=thing_classes
    )

    # Load and evaluate predictions for each dataset
    for dataset_name in dataset_names:
        logger.info(f"Evaluating predictions for {dataset_name}")
        # to get the thing_classes and thing_dataset_id_to_contiguous_id for the MetadataCatalog.get(dataset_name)
        DatasetCatalog.get(dataset_name)
        # Load predictions
        pred_path = prediction_paths[dataset_name]
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Prediction file not found: {pred_path}")
        
        with PathManager.open(pred_path, "rb") as f:
            predictions = torch.load(f)
        
        # Add predictions to evaluator
        eval_helper.add_predictions(dataset_name, predictions)
        
        # Run evaluation
        eval_helper.evaluate(dataset_name)
        
        # Save predictions if needed
        eval_helper.save_predictions(dataset_name)

    # Summarize results
    eval_helper.summarize_all()

def main():
    """Main function demonstrating how to use the evaluation script"""

    dataset_names = ["SUNRGBD_test_novel", "KITTI_test_novel", "ARKitScenes_test_novel"] 
    prediction_paths = {
        "SUNRGBD_test_novel": "./output/ovmono3d_geo/SUNRGBD_test_novel.pth",
        "KITTI_test_novel": "./output/ovmono3d_geo/KITTI_test_novel.pth",
        "ARKitScenes_test_novel": "./output/ovmono3d_geo/ARKitScenes_test_novel.pth"
    }
    
    # Setup filter settings
    filter_settings = {
        'visibility_thres': 0.33333333,
        'truncation_thres': 0.33333333,
        'min_height_thres': 0.0625,
        'max_depth': 100000000.0,        
        'category_names': None,  # Will be set based on category_path
        'ignore_names': ['dontcare', 'ignore', 'void'],
        'trunc_2D_boxes': True,
        'modal_2D_boxes': False,
        'max_height_thres': 1.5,
    }
    
    # Set paths
    output_dir = "./output/ovmono3d_geo"
    category_path = "./configs/category_meta.json"
    
    # Run evaluation
    evaluate_predictions(
        dataset_names=dataset_names,
        prediction_paths=prediction_paths,
        filter_settings=filter_settings,
        output_dir=output_dir,
        category_path=category_path,
        eval_mode="novel",
        iter_label='final'
    )

if __name__ == "__main__":
    main() 