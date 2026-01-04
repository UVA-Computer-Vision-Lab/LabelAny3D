#!/bin/bash

# Download COCO dataset and COCONUT annotations to dataset/coco/ directory
# Run from repository root: bash src/download_coco.sh
#
# Usage:
#   bash src/download_coco.sh          # Download everything (COCO + COCONUT)
#   bash src/download_coco.sh --coco   # Download only COCO images and annotations
#   bash src/download_coco.sh --coconut # Download only COCONUT annotations

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TARGET_DIR="${REPO_ROOT}/dataset/coco"

download_coco() {
    mkdir -p "${TARGET_DIR}/images"
    cd "${TARGET_DIR}/images"

    echo "=== Downloading COCO images to ${TARGET_DIR}/images ==="
    wget http://images.cocodataset.org/zips/train2017.zip
    wget http://images.cocodataset.org/zips/val2017.zip
    wget http://images.cocodataset.org/zips/test2017.zip

    unzip train2017.zip
    unzip val2017.zip
    unzip test2017.zip

    rm train2017.zip
    rm val2017.zip
    rm test2017.zip

    cd "${TARGET_DIR}"
    echo "=== Downloading COCO annotations to ${TARGET_DIR}/annotations ==="
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
    wget http://images.cocodataset.org/annotations/image_info_test2017.zip

    unzip annotations_trainval2017.zip
    unzip stuff_annotations_trainval2017.zip
    unzip image_info_test2017.zip

    rm annotations_trainval2017.zip
    rm stuff_annotations_trainval2017.zip
    rm image_info_test2017.zip

    echo "COCO dataset downloaded to ${TARGET_DIR}"
}

download_coconut() {
    echo "=== Downloading COCONUT annotations ==="
    cd "${SCRIPT_DIR}"
    python download_coconut.py --output_dir "${TARGET_DIR}/annotations" --split all

    echo "COCONUT annotations downloaded to ${TARGET_DIR}/annotations"
}

# Parse arguments
if [ "$1" == "--coco" ]; then
    download_coco
elif [ "$1" == "--coconut" ]; then
    download_coconut
else
    # Download everything
    download_coco
    download_coconut
fi

echo "=== Download complete ==="
echo "Dataset structure:"
echo "  ${TARGET_DIR}/"
echo "  ├── images/"
echo "  │   ├── train2017/"
echo "  │   ├── val2017/"
echo "  │   └── test2017/"
echo "  └── annotations/"
echo "      ├── instances_*.json (COCO)"
echo "      ├── coconut_train.json (COCONUT)"
echo "      └── coconut_val.json (COCONUT)"