"""
COCONUT annotation loader.

Loads COCONUT instance segmentation annotations from coconut_val.json / coconut_train.json.

Usage:
    from coconut_loader import CoconutLoader

    loader = CoconutLoader(split="val")
    images = loader.get_images()
    annotations = loader.get_annotations(image_id)
"""

import json
import os
from typing import Dict, List, Optional, Any


class CoconutLoader:
    """Load COCONUT instance segmentation annotations."""

    def __init__(self, split: str = "val", annotations_dir: str = "../dataset/coco/annotations"):
        """
        Initialize the loader.

        Args:
            split: "val" or "train"
            annotations_dir: Path to the annotations directory
        """
        self.split = split

        if split == "val":
            json_path = os.path.join(annotations_dir, "coconut_val.json")
        else:
            json_path = os.path.join(annotations_dir, "coconut_train.json")

        print(f"Loading COCONUT annotations from {json_path}...")
        with open(json_path, 'r') as f:
            data = json.load(f)

        self.images = data["images"]
        self.categories = data.get("categories", [])

        # Build image_id -> annotations mapping
        # COCONUT instance format: flat list of annotations, each with image_id
        self.annotations_by_image: Dict[int, List[Any]] = {}
        for anno in data["annotations"]:
            img_id = anno["image_id"]
            if img_id not in self.annotations_by_image:
                self.annotations_by_image[img_id] = []
            self.annotations_by_image[img_id].append(anno)

        print(f"Loaded {len(self.images)} images with annotations")

    def get_images(self) -> List[Dict]:
        """Get list of all images."""
        return self.images

    def get_image_by_index(self, index: int) -> Dict:
        """Get image info by index."""
        return self.images[index]

    def get_annotations(self, image_id: int) -> List[Dict]:
        """Get annotations for a specific image."""
        return self.annotations_by_image.get(image_id, [])

    def get_categories(self) -> List[Dict]:
        """Get category definitions."""
        return self.categories

    def __len__(self) -> int:
        """Return number of images."""
        return len(self.images)


def get_dataset_paths(split: str) -> tuple:
    """
    Get dataset paths for a given split.

    Returns:
        (dataset_root, annotations_dir)
    """
    if split == "val":
        dataset_root = "../dataset/coco/images/val2017/"
    else:
        dataset_root = "../dataset/coco/images/train2017/"

    annotations_dir = "../dataset/coco/annotations"

    return dataset_root, annotations_dir
