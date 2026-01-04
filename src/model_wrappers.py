"""
Unified model wrappers for all external models used in the pipeline.

Models included:
- Reconstruction: TRELLIS, Hunyuan3D, DreamGaussian
- Depth: MoGe, DepthPro
- Matching: MASt3R
- Enhancement: InvSR
- Segmentation (in-the-wild): EntityV2, CLIPSeg, OneFormer
- Tagging (in-the-wild): OVSAM
- Completion: Amodal completion
"""

import os
import sys
import warnings
import numpy as np

warnings.simplefilter('ignore', category=UserWarning)
warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=DeprecationWarning)


# =============================================================================
# Lazy loading state for all models
# =============================================================================
_loaded_models = {}


# =============================================================================
# Segmentation Helper Functions
# =============================================================================
def filter_component_masks(masks, foreground_mask, threshold=0.5):
    """Filter masks based on foreground overlap."""
    all_instances = np.arange(len(masks))
    is_foreground = ((masks & foreground_mask).sum((-1, -2)) + 1e-6) / (masks.sum((-1, -2)) + 1e-6) > threshold
    return all_instances[is_foreground], all_instances[~is_foreground]


def initialize_oneformer(device):
    """Initialize OneFormer model for semantic segmentation."""
    from oneformer import (
        add_oneformer_config,
        add_common_config,
        add_swin_config,
        add_dinat_config,
        add_convnext_config,
    )
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from detectron2.data import MetadataCatalog
    from demo.defaults import DefaultPredictor

    SWIN_CFG_DICT = {"cityscapes": "../external/OneFormer-Colab/configs/cityscapes/oneformer_swin_large_IN21k_384_bs16_90k.yaml",
                     "coco": "../external/OneFormer-Colab/configs/coco/oneformer_swin_large_IN21k_384_bs16_100ep.yaml",
                     "ade20k": "../external/OneFormer-Colab/configs/ade20k/oneformer_swin_large_IN21k_384_bs16_160k.yaml", }

    DINAT_CFG_DICT = {"cityscapes": "../external/OneFormer-Colab/configs/cityscapes/oneformer_dinat_large_bs16_90k.yaml",
                      "coco": "../external/OneFormer-Colab/configs/coco/oneformer_dinat_large_bs16_100ep.yaml",
                      "ade20k": "../external/OneFormer-Colab/configs/ade20k/oneformer_dinat_large_IN21k_384_bs16_160k.yaml", }

    def setup_cfg(dataset, model_path, use_swin):
        # load config from file and command-line arguments
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_common_config(cfg)
        add_swin_config(cfg)
        add_dinat_config(cfg)
        add_convnext_config(cfg)
        add_oneformer_config(cfg)
        if use_swin:
            cfg_path = SWIN_CFG_DICT[dataset]
        else:
            cfg_path = DINAT_CFG_DICT[dataset]
        cfg.merge_from_file(cfg_path)
        cfg.MODEL.DEVICE = device
        cfg.MODEL.WEIGHTS = model_path
        cfg.freeze()
        return cfg

    def setup_modules(dataset, model_path, use_swin):
        cfg = setup_cfg(dataset, model_path, use_swin)
        predictor = DefaultPredictor(cfg)
        metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST_PANOPTIC[0] if len(cfg.DATASETS.TEST_PANOPTIC) else "__unused"
        )
        return predictor, metadata

    predictor, metadata = setup_modules("ade20k", "../external/checkpoints/250_16_dinat_l_oneformer_ade20k_160k.pth", False)

    my_stuff = [
        'window ',
        'door',
        'curtain',
        'mirror',
        'fence',
        'rail',
        'column, pillar',
        'stairs',
        'screen door, screen',
        'bannister, banister, balustrade, balusters, handrail',
        'step, stair',
    ]

    my_thing = [
        'plant',
        'tent',
        'crt screen',
        'cradle',
        'blanket, cover'
    ]

    custom_thing = []
    for thing in metadata.thing_classes:
        if thing not in my_stuff:
            custom_thing.append(metadata.stuff_classes.index(thing))

    for thing in my_thing:
        custom_thing.append(metadata.stuff_classes.index(thing))

    return predictor, metadata, custom_thing


def _ensure_path(external_path):
    """Ensure external path is in sys.path"""
    if external_path not in sys.path:
        sys.path.insert(0, external_path)


# =============================================================================
# TRELLIS - Image to 3D Reconstruction
# =============================================================================
def load_trellis():
    """Load TRELLIS model (lazy loading)"""
    if 'trellis' not in _loaded_models:
        _ensure_path('../external/TRELLIS')
        os.environ['ATTN_BACKEND'] = 'xformers'

        from trellis.pipelines import TrellisImageTo3DPipeline

        pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
        pipeline.cuda()
        _loaded_models['trellis'] = pipeline
        print("TRELLIS model loaded.")

    return _loaded_models['trellis']


def infer_with_trellis(out_dir, obj_id):
    """
    Run TRELLIS inference on a single object.

    Args:
        out_dir: Output directory (Path object)
        obj_id: Object identifier string

    Returns:
        Mesh object, or None if failed
    """
    from pathlib import Path
    from PIL import Image

    _ensure_path('../external/TRELLIS')
    from trellis.utils import postprocessing_utils

    print("Starting TRELLIS inference...")

    try:
        pipeline = load_trellis()

        img_path = (Path(out_dir) / "crops" / f"{obj_id}_rgba.png").absolute()
        image = Image.open(img_path)

        outputs = pipeline.run(image, seed=1)

        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            texture_size=1024,
        )
        glb.export(f"{out_dir}/object_space/{obj_id}.glb")

        print(f"TRELLIS inference complete: {out_dir}/object_space/{obj_id}.glb")
        return outputs['mesh'][0]

    except Exception as e:
        print(f"TRELLIS inference failed: {e}")
        return None


# =============================================================================
# Hunyuan3D - Image to 3D Reconstruction
# =============================================================================
def load_hunyuan3d():
    """Load Hunyuan3D models (lazy loading)"""
    if 'hunyuan3d' not in _loaded_models:
        _ensure_path('../external/Hunyuan3D-1')

        from infer import Removebg, Image2Views, Views2Mesh

        rembg_model = Removebg()
        image_to_views = Image2Views(
            device='cuda:0',
            use_lite=False,
            save_memory=False,
            std_pretrain='../external/Hunyuan3D-1/weights/mvd_std',
        )
        views_to_mesh = Views2Mesh(
            '../external/Hunyuan3D-1/svrm/configs/svrm.yaml',
            '../external/Hunyuan3D-1/weights/svrm/svrm.safetensors',
            'cuda:0',
            use_lite=False,
            save_memory=False
        )

        _loaded_models['hunyuan3d'] = {
            'rembg': rembg_model,
            'image_to_views': image_to_views,
            'views_to_mesh': views_to_mesh,
        }
        print("Hunyuan3D models loaded.")

    return _loaded_models['hunyuan3d']


def infer_with_hunyuan(out_dir, obj_id, gen_seed=0, gen_steps=50, max_faces_num=90000, do_texture_mapping=True):
    """
    Run Hunyuan3D inference on a single object.

    Args:
        out_dir: Output directory (Path object)
        obj_id: Object identifier string
        gen_seed: Random seed for generation
        gen_steps: Number of generation steps
        max_faces_num: Maximum number of faces in the output mesh
        do_texture_mapping: Whether to apply texture mapping

    Returns:
        Path to the generated GLB file, or None if failed
    """
    from pathlib import Path
    from PIL import Image
    import shutil

    print("Starting Hunyuan3D inference...")

    try:
        models = load_hunyuan3d()

        save_path = (Path(out_dir) / "object_space" / f"{obj_id}").absolute()
        img_path = (Path(out_dir) / "crops" / f"{obj_id}_rgba.png").absolute()

        os.makedirs(save_path, exist_ok=True)

        # Load input image
        res_rgb_pil = Image.open(img_path)
        res_rgb_pil.save(os.path.join(save_path, "img_nobg.png"))

        # Stage 1: Image to multi-views
        (views_grid_pil, cond_img), view_pil_list = models['image_to_views'](
            res_rgb_pil,
            seed=gen_seed,
            steps=gen_steps
        )
        views_grid_pil.save(os.path.join(save_path, "views.jpg"))

        # Stage 2: Views to mesh
        models['views_to_mesh'](
            views_grid_pil,
            cond_img,
            seed=gen_seed,
            target_face_count=max_faces_num,
            save_folder=str(save_path),
            do_texture_mapping=do_texture_mapping
        )

        # Move the output mesh to the expected location
        source_mesh = save_path / "mesh.glb"
        target_mesh = Path(out_dir) / "object_space" / f"{obj_id}.glb"

        if source_mesh.exists():
            shutil.copy(str(source_mesh), str(target_mesh))
            print(f"Hunyuan3D inference complete: {target_mesh}")
            return target_mesh
        else:
            print(f"Warning: Expected mesh not found at {source_mesh}")
            return None

    except Exception as e:
        print(f"Hunyuan3D inference failed: {e}")
        return None


# =============================================================================
# MoGe - Monocular Geometry Estimation
# =============================================================================
def load_moge():
    """Load MoGe model (lazy loading)"""
    if 'moge' not in _loaded_models:
        _ensure_path('../external/MoGe')
        from infer_moge import infer_geometry_on_image as _infer_moge
        _loaded_models['moge'] = _infer_moge
        print("MoGe model loaded.")

    return _loaded_models['moge']


def infer_with_moge(image_path, out_dir):
    """
    Run MoGe inference to get depth and camera intrinsics.

    Args:
        image_path: Path to input image
        out_dir: Output directory

    Returns:
        Tuple of (points, depth_map, mask, K)
    """
    infer_fn = load_moge()
    return infer_fn(image_path, out_dir)


# =============================================================================
# DepthPro - Metric Depth Estimation
# =============================================================================
def load_depthpro(device='cuda:0'):
    """Load DepthPro model (lazy loading)"""
    import torch

    if 'depthpro' not in _loaded_models:
        import depth_pro

        model, transform = depth_pro.create_model_and_transforms(
            device=device,
            precision=torch.float16
        )
        model.eval()

        _loaded_models['depthpro'] = {
            'model': model,
            'transform': transform,
        }
        print("DepthPro model loaded.")

    return _loaded_models['depthpro']


def infer_with_depthpro(image_pil, focal_length, device='cuda:0'):
    """
    Run DepthPro inference to get metric depth.

    Args:
        image_pil: PIL Image
        focal_length: Focal length in pixels
        device: CUDA device

    Returns:
        Depth map as numpy array
    """
    models = load_depthpro(device)

    img = models['transform'](image_pil)
    prediction = models['model'].infer(img, f_px=focal_length)
    depth = prediction["depth"]

    return depth.cpu().numpy()


# =============================================================================
# MASt3R - Dense Matching
# =============================================================================
def load_mast3r(device='cuda:0'):
    """Load MASt3R model (lazy loading)"""
    import torch

    if 'mast3r' not in _loaded_models:
        _ensure_path('../external/mast3r')
        from mast3r.model import AsymmetricMASt3R

        model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
        model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
        model.eval()

        _loaded_models['mast3r'] = model
        print("MASt3R model loaded.")

    return _loaded_models['mast3r']


# =============================================================================
# InvSR - Image Super-Resolution
# =============================================================================
def load_invsr():
    """Load InvSR model (lazy loading)"""
    if 'invsr' not in _loaded_models:
        _ensure_path('../external/InvSR')
        from inference_invsr_us import get_parser, get_configs, InvSamplerSR

        args = get_parser(description="InvSR")
        configs = get_configs(args)
        sampler = InvSamplerSR(configs)

        _loaded_models['invsr'] = {
            'sampler': sampler,
            'args': args,
        }
        print("InvSR model loaded.")

    return _loaded_models['invsr']


def enhance_with_invsr(input_path, output_dir):
    """
    Run InvSR super-resolution.

    Args:
        input_path: Path to input image
        output_dir: Output directory

    Returns:
        Path to enhanced image
    """
    models = load_invsr()
    models['sampler'].inference(
        str(input_path),
        out_path=output_dir,
        bs=models['args'].bs
    )
    return output_dir / 'input.png'


# =============================================================================
# Utility functions
# =============================================================================
def unload_model(model_name):
    """Unload a specific model to free GPU memory"""
    import torch

    if model_name in _loaded_models:
        del _loaded_models[model_name]
        torch.cuda.empty_cache()
        print(f"Model '{model_name}' unloaded.")


def unload_all_models():
    """Unload all models to free GPU memory"""
    import torch

    _loaded_models.clear()
    torch.cuda.empty_cache()
    print("All models unloaded.")


# =============================================================================
# DreamGaussian - Image to 3D (legacy)
# =============================================================================
def run_dreamgaussian(scene_dir, obj_id, elevation):
    """
    Run DreamGaussian for 3D reconstruction.

    Args:
        scene_dir: Scene directory (Path object)
        obj_id: Object identifier string
        elevation: Elevation angle in degrees
    """
    from pathlib import Path

    save_path = (scene_dir / "object_space" / f"{obj_id}").absolute()
    main_dir_path = Path.cwd().absolute()
    img_path = (scene_dir / "crops" / f"{obj_id}_rgba.png").absolute()

    os.chdir('../external/dreamgaussian/')
    bash_script = f'python main.py --config configs/image.yaml input={img_path} save_path={save_path} elevation={elevation} force_cuda_rast=True'
    print(bash_script)
    os.system(bash_script)
    os.chdir(main_dir_path)


# =============================================================================
# EntityV2 - Instance Segmentation (in-the-wild mode)
# =============================================================================
def run_entityv2(image, threshold=0.1, max_size=1500):
    """
    Run EntityV2 (CropFormer) for instance segmentation.

    Args:
        image: Input image as numpy array (RGB)
        threshold: Score threshold for mask selection
        max_size: Maximum image size for processing

    Returns:
        Selected masks as numpy array
    """
    import numpy as np
    import cv2
    import torchvision.transforms.functional as Ftv

    from detectron2.data import MetadataCatalog, DatasetCatalog
    from detectron2.modeling import BACKBONE_REGISTRY, SEM_SEG_HEADS_REGISTRY

    # For cleanup
    def_keys = list(DatasetCatalog.keys())

    from detectron2.config import get_cfg
    from demo_cropformer.predictor import VisualizationDemo
    from detectron2.projects.deeplab import add_deeplab_config
    from mask2former import add_maskformer2_config

    CropFormerCfg = {
        'config_file': "../external/detectron2/projects/CropFormer/configs/entityv2/entity_segmentation/cropformer_hornet_3x.yaml",
        'opts': ["MODEL.WEIGHTS", "../external/checkpoints/CropFormer_hornet_3x_03823a.pth"],
    }
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(CropFormerCfg['config_file'])
    cfg.merge_from_list(CropFormerCfg['opts'])
    cfg.freeze()
    segmentor = VisualizationDemo(cfg)

    # CropFormer expects BGR
    orig_size = image.shape
    if max(image.shape) > max_size:
        scale_factor = max_size / max(image.shape[:-1])
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
    img = image.copy()[:, :, ::-1]
    predictions = segmentor.run_on_image(img)

    # Hacky cleanup
    extra_keys = list(DatasetCatalog.keys())
    for key in extra_keys:
        if key not in def_keys:
            DatasetCatalog.remove(key)
            MetadataCatalog.remove(key)
    BACKBONE_REGISTRY._obj_map = {}
    SEM_SEG_HEADS_REGISTRY._obj_map = {}

    pred_masks = predictions["instances"].pred_masks
    pred_scores = predictions["instances"].scores
    selected_indexes = (pred_scores >= threshold)
    selected_masks = pred_masks[selected_indexes]
    selected_masks = Ftv.resize(selected_masks.cpu(), orig_size[:-1]).numpy() > 0.5
    return selected_masks


# =============================================================================
# CLIPSeg - Background/Foreground Classification (in-the-wild mode)
# =============================================================================
def run_clipseg(image, masks):
    """
    Run CLIPSeg for background/foreground classification.

    Args:
        image: Input PIL image
        masks: Instance masks to filter

    Returns:
        Filtered masks (foreground only)
    """
    import torch
    import numpy as np
    from PIL import Image
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    background_prompts = ["background", "floor", "wall", "curtain", "window", "ceiling", "table"]
    foreground_prompts = ["object", "furniture"]

    img = image
    inputs = processor(
        text=background_prompts + foreground_prompts,
        images=[image] * (len(background_prompts) + len(foreground_prompts)),
        padding="max_length", return_tensors="pt"
    )
    predicted = torch.sigmoid(model(**inputs).logits)
    back_pred = (predicted[:len(background_prompts)] > 0.5).any(dim=0)
    fore_pred = (predicted[-len(foreground_prompts):] > 0.1).any(dim=0)
    foreground_mask = torch.logical_or(~back_pred, fore_pred).numpy()
    foreground_mask = np.array(Image.fromarray(foreground_mask).resize(img.size))
    return filter_component_masks(masks, foreground_mask)


# =============================================================================
# OneFormer - Semantic Segmentation (in-the-wild mode)
# =============================================================================
def run_oneformer(image, masks, device):
    """
    Run OneFormer for semantic segmentation filtering.

    Args:
        image: Input PIL image
        masks: Instance masks to filter
        device: CUDA device

    Returns:
        Filtered masks (things only)
    """
    import numpy as np
    from PIL import Image

    predictor, metadata, thing_classes_ids = initialize_oneformer(device)
    W, H = image.size
    f = 640 * 4 / W
    img = np.array(image.resize((int(W * f), int(H * f)), Image.BILINEAR))[:, :, ::-1]
    predictions = predictor(img, "semantic")['sem_seg'].argmax(dim=0).cpu().numpy()
    is_thing = Image.fromarray(np.isin(predictions, thing_classes_ids))
    is_thing = np.array(is_thing.resize((W, H), Image.NEAREST))
    return filter_component_masks(masks, is_thing)


# =============================================================================
# OVSAM - Open-Vocabulary Tagging (in-the-wild mode)
# =============================================================================
def run_ovsam(image, masks, max_samples=5):
    """
    Run OVSAM for open-vocabulary object tagging.

    Args:
        image: Input PIL image
        masks: Instance masks
        max_samples: Maximum samples per mask

    Returns:
        Tuple of (tags, scores)
    """
    from image_tagger import ImageTagger

    tagger = ImageTagger()
    tags, scores = tagger.infer(image, masks, max_samples=max_samples)
    return tags


# =============================================================================
# Amodal Completion (in-the-wild mode)
# =============================================================================
def complete_object(crop, label, model):
    """
    Complete occluded object regions using diffusion model.

    Args:
        crop: Cropped RGBA image
        label: Object category label
        model: Diffusion model (InstructPix2Pix)

    Returns:
        Completed RGB image
    """
    import numpy as np

    image, mask = np.split(np.array(crop) / 255, (3,), axis=-1)
    image[mask[:, :, 0] < 0.5] = 0.5
    completed = model(
        prompt=label,
        image=image,
        num_inference_steps=50,
        image_guidance_scale=1.5,
        guidance_scale=8.5,
        num_images_per_prompt=1
    ).images[0]
    return completed
