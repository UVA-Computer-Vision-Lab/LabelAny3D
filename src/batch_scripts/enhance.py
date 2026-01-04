import argparse
from omegaconf import OmegaConf
import sys
import os
from tqdm import tqdm
import torch
sys.path = [
    './',
    '../external/InvSR',
] + sys.path
from dataset_model import get_scene
from pathlib import Path
from inference_invsr_us import get_parser, get_configs, InvSamplerSR
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

    invsr_args = get_parser(description="My CLI tool")
    invsr_configs = get_configs(invsr_args)
    sampler = InvSamplerSR(invsr_configs)

    for i in tqdm(range(args.start_index, args.end_index)):
        image_info = loader.get_image_by_index(i)
        img_name = image_info["file_name"]
        image_path = os.path.join(dataset_root, img_name)
        output_dir = os.path.join(args.save_dir, args.split, img_name.split(".")[0].replace("/", "_").replace("-", "_"))

        opt.scene.attributes.img_path = image_path
        scene = get_scene(opt.scene.type, opt.scene.attributes)
        
        out_dir = Path(output_dir)
        print(f"Saving to {out_dir}")
        out_dir.mkdir(exist_ok=True, parents=True)
        (out_dir / "crops").mkdir(exist_ok=True)
        (out_dir / "object_space").mkdir(exist_ok=True)
        (out_dir / "reconstruction").mkdir(exist_ok=True)
        (out_dir / "enhanced").mkdir(exist_ok=True)

        # making directories for result

        scene.image_pil.save(out_dir / 'input.png')

        enhance_path = out_dir / 'enhanced' / 'input.png'
        if os.path.exists(enhance_path):
            continue
        else:
            sampler.inference(f"{out_dir}/input.png", out_path=out_dir / 'enhanced', bs=invsr_args.bs)


        
