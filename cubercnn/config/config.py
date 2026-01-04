# Copyright (c) Meta Platforms, Inc. and affiliates
from detectron2.config import CfgNode as CN

def get_cfg_defaults(cfg):

    cfg.DATASETS.FOLDER_NAME = 'Omni3D'

    # A list of category names which will be used
    cfg.DATASETS.CATEGORY_NAMES = []

    # The category names which will be treated as ignore
    # e.g., not counting as background during training
    # or as false positives during evaluation.
    cfg.DATASETS.IGNORE_NAMES = []

    # Should the datasets appear with the same probabilty
    # in batches (e.g., the imbalance from small and large
    # datasets will be accounted for during sampling)
    cfg.DATALOADER.BALANCE_DATASETS = False

    # The datasets that will be used for metric scale supervision, otherwise the scale-invariant loss will be used
    cfg.DATASETS.METRIC_DATASETS = ['SUNRGBD_train', 'SUNRGBD_val', 'Hypersim_train', 'Hypersim_val', 'ARKitScenes_train', 'ARKitScenes_val', 'Objectron_train', 'Objectron_val', 'nuScenes_train', 'nuScenes_val', 'KITTI_train', 'KITTI_val']

    # The thresholds for when to treat a known box
    # as ignore based on too heavy of truncation or 
    # too low of visibility in the image. This affects
    # both training and evaluation ignores.
    cfg.DATASETS.TRUNCATION_THRES = 0.99
    cfg.DATASETS.VISIBILITY_THRES = 0.01
    cfg.DATASETS.MIN_HEIGHT_THRES = 0.00
    cfg.DATASETS.MAX_DEPTH = 1e8

    # Whether modal 2D boxes should be loaded, 
    # or if the full 3D projected boxes should be used.
    cfg.DATASETS.MODAL_2D_BOXES = False

    # Whether truncated 2D boxes should be loaded, 
    # or if the 3D full projected boxes should be used.
    cfg.DATASETS.TRUNC_2D_BOXES = True

    cfg.DATASETS.TEST_BASE = ('SUNRGBD_test', 'Hypersim_test', 'ARKitScenes_test', 'Objectron_test', 'KITTI_test', 'nuScenes_test') 
    cfg.DATASETS.TEST_NOVEL = ('SUNRGBD_test_novel','ARKitScenes_test_novel', 'KITTI_test_novel') 
    # cfg.DATASETS.CATEGORY_NAMES_BASE = ('chair', 'table', 'cabinet', 'car', 'lamp', 'books', 'sofa', 'pedestrian', 'picture', 'window', 'pillow', 'truck', 'door', 'blinds', 'sink', 'shelves', 'television', 'shoes', 'cup', 'bottle', 'bookcase', 'laptop', 'desk', 'cereal box', 'floor mat', 'traffic cone', 'mirror', 'barrier', 'counter', 'camera', 'bicycle', 'toilet', 'bus', 'bed', 'refrigerator', 'trailer', 'box', 'oven', 'clothes', 'van', 'towel', 'motorcycle', 'night stand', 'stove', 'machine', 'stationery', 'bathtub', 'cyclist', 'curtain', 'bin')
    cfg.DATASETS.CATEGORY_NAMES_NOVEL = ('monitor', 'bag', 'dresser', 'board', 'printer', 'keyboard', 'painting', 'drawers', 'microwave', 'computer', 'kitchen pan', 'potted plant', 'tissues', 'rack', 'tray', 'toys', 'phone', 'podium', 'cart', 'soundsystem', 'fireplace', 'tram')
    cfg.DATASETS.CATEGORY_NAMES_BASE = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
    # cfg.DATASETS.CATEGORY_NAMES_COCO3D = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
    # Oracle 2D files for evaluation
    cfg.DATASETS.ORACLE2D_FILES = CN()
    cfg.DATASETS.ORACLE2D_FILES.EVAL_MODE = 'target_aware'   # 'target_aware' or 'previous_metric' or 'gt'
    
    # Create a configuration for each evaluation mode
    for mode in ['target_aware', 'previous_metric', 'gt']:
        cfg.DATASETS.ORACLE2D_FILES[mode] = CN()
        cfg.DATASETS.ORACLE2D_FILES[mode].novel = CN()
        cfg.DATASETS.ORACLE2D_FILES[mode].base = CN()

        # Oracle 2D file for the Novel class dataset
        novel_datasets = {
            'SUNRGBD_test_novel': 'sunrgbd',
            'ARKitScenes_test_novel': 'arkitscenes', 
            'KITTI_test_novel': 'kitti'
        }
        
        # Oracle 2D file for the Base class dataset
        base_datasets = {
            'SUNRGBD_test': 'sunrgbd',
            'Hypersim_test': 'hypersim',
            'ARKitScenes_test': 'arkitscenes',
            'Objectron_test': 'objectron',
            'KITTI_test': 'kitti',
            'nuScenes_test': 'nuscenes',
            'COCO3D_val': 'coco3d'
        }

        # Set the file path for the novel class
        for dataset, dataset_name in novel_datasets.items():
            if mode == 'previous_metric':
                prefix = 'gdino_novel_previous_metric'
            elif mode == 'gt':
                prefix = 'gt'
            else:
                prefix = 'gdino'
            cfg.DATASETS.ORACLE2D_FILES[mode].novel[dataset] = f'datasets/Omni3D/{prefix}_{dataset_name}_novel_oracle_2d.json'

        # Set the file path for the base class
        for dataset, dataset_name in base_datasets.items():
            if mode == 'previous_metric':
                prefix = 'gdino_previous_eval'
            elif mode == 'gt':
                prefix = 'gt'
            else:
                prefix = 'gdino'
            cfg.DATASETS.ORACLE2D_FILES[mode].base[dataset] = f'datasets/Omni3D/{prefix}_{dataset_name}_base_oracle_2d.json'

    cfg.MODEL.FPN.IN_FEATURE = None
    cfg.MODEL.FPN.SQUARE_PAD = 0
    # Threshold used for matching and filtering boxes
    # inside of ignore regions, within the RPN and ROIHeads
    cfg.MODEL.RPN.IGNORE_THRESHOLD = 0.5

    cfg.MODEL.DINO = CN()
    cfg.MODEL.DINO.NAME = 'dinov2'
    cfg.MODEL.DINO.MODEL_NAME = 'vitb14'
    cfg.MODEL.DINO.OUTPUT = 'dense'
    cfg.MODEL.DINO.LAYER = -1
    cfg.MODEL.DINO.RETURN_MULTILAYER = False

    cfg.MODEL.MAE = CN()
    cfg.MODEL.MAE.CHECKPOINT = 'facebook/vit-mae-base'
    cfg.MODEL.MAE.OUTPUT = 'dense'
    cfg.MODEL.MAE.LAYER = -1
    cfg.MODEL.MAE.RETURN_MULTILAYER = False

    cfg.MODEL.CLIP = CN()
    cfg.MODEL.CLIP.ARCH = 'ViT-B-16'
    cfg.MODEL.CLIP.CHECKPOINT = 'openai'
    cfg.MODEL.CLIP.OUTPUT = 'dense'
    cfg.MODEL.CLIP.LAYER = -1
    cfg.MODEL.CLIP.RETURN_MULTILAYER = False

    cfg.MODEL.MIDAS = CN()
    cfg.MODEL.MIDAS.OUTPUT = 'dense'
    cfg.MODEL.MIDAS.LAYER = -1
    cfg.MODEL.MIDAS.RETURN_MULTILAYER = False

    cfg.MODEL.SAM = CN()
    cfg.MODEL.SAM.OUTPUT = 'dense'
    cfg.MODEL.SAM.LAYER = -1
    cfg.MODEL.SAM.RETURN_MULTILAYER = False
    
    # Configuration for cube head
    cfg.MODEL.ROI_CUBE_HEAD = CN()
    cfg.MODEL.ROI_CUBE_HEAD.NAME = "CubeHead"
    cfg.MODEL.ROI_CUBE_HEAD.POOLER_RESOLUTION = 7
    cfg.MODEL.ROI_CUBE_HEAD.POOLER_SAMPLING_RATIO = 0
    cfg.MODEL.ROI_CUBE_HEAD.POOLER_TYPE = "ROIAlignV2"

    # Settings for the cube head features
    cfg.MODEL.ROI_CUBE_HEAD.NUM_CONV = 0
    cfg.MODEL.ROI_CUBE_HEAD.CONV_DIM = 256
    cfg.MODEL.ROI_CUBE_HEAD.NUM_FC = 2
    cfg.MODEL.ROI_CUBE_HEAD.FC_DIM = 1024
    cfg.MODEL.ROI_CUBE_HEAD.USE_TRANSFORMER = False
    
    # the style to predict Z with currently supported
    # options --> ['direct', 'sigmoid', 'log', 'clusters']
    cfg.MODEL.ROI_CUBE_HEAD.Z_TYPE = "direct"

    # the style to predict pose with currently supported
    # options --> ['6d', 'euler', 'quaternion']
    cfg.MODEL.ROI_CUBE_HEAD.POSE_TYPE = "6d"

    # Whether to scale all 3D losses by inverse depth
    cfg.MODEL.ROI_CUBE_HEAD.INVERSE_Z_WEIGHT = False

    # Virtual depth puts all predictions of depth into
    # a shared virtual space with a shared focal length. 
    cfg.MODEL.ROI_CUBE_HEAD.VIRTUAL_DEPTH = True
    cfg.MODEL.ROI_CUBE_HEAD.VIRTUAL_FOCAL = 512.0

    # If true, then all losses are computed using the 8 corners
    # such that they are all in a shared scale space. 
    # E.g., their scale correlates with their impact on 3D IoU.
    # This way no manual weights need to be set.
    cfg.MODEL.ROI_CUBE_HEAD.DISENTANGLED_LOSS = True

    # When > 1, the outputs of the 3D head will be based on
    # a 2D scale clustering, based on 2D proposal height/width.
    # This parameter describes the number of bins to cluster.
    cfg.MODEL.ROI_CUBE_HEAD.CLUSTER_BINS = 1

    # Whether batch norm is enabled during training. 
    # If false, all BN weights will be frozen. 
    cfg.MODEL.USE_BN = True

    # Whether to predict the pose in allocentric space. 
    # The allocentric space may correlate better with 2D 
    # images compared to egocentric poses. 
    cfg.MODEL.ROI_CUBE_HEAD.ALLOCENTRIC_POSE = True

    # Whether to use chamfer distance for disentangled losses
    # of pose. This avoids periodic issues of rotation but 
    # may prevent the pose "direction" from being interpretable.
    cfg.MODEL.ROI_CUBE_HEAD.CHAMFER_POSE = True

    # Should the prediction heads share FC features or not. 
    # These include groups of uv, z, whl, pose.
    cfg.MODEL.ROI_CUBE_HEAD.SHARED_FC = True

    # Check for stable gradients. When inf is detected, skip the update. 
    # This prevents an occasional bad sample from exploding the model. 
    # The threshold below is the allows percent of bad samples. 
    # 0.0 is off, and 0.01 is recommended for minor robustness to exploding.
    cfg.MODEL.STABILIZE = 0.01
    
    # Whether or not to use the dimension priors
    cfg.MODEL.ROI_CUBE_HEAD.DIMS_PRIORS_ENABLED = True

    # How prior dimensions should be computed? 
    # The supported modes are ["exp", "sigmoid"]
    # where exp is unbounded and sigmoid is bounded
    # between +- 3 standard deviations from the mean.
    cfg.MODEL.ROI_CUBE_HEAD.DIMS_PRIORS_FUNC = 'exp'

    # weight for confidence loss. 0 is off.
    cfg.MODEL.ROI_CUBE_HEAD.USE_CONFIDENCE = 1.0

    # Loss weights for XY, Z, Dims, Pose
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_3D = 1.0
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_XY = 1.0
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_Z = 1.0
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_DIMS = 1.0
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_POSE = 1.0

    cfg.MODEL.DLA = CN()

    # Supported types for DLA backbones are...
    # dla34, dla46_c, dla46x_c, dla60x_c, dla60, dla60x, dla102x, dla102x2, dla169
    cfg.MODEL.DLA.TYPE = 'dla34'

    # Only available for dla34, dla60, dla102
    cfg.MODEL.DLA.TRICKS = False

    # A joint loss for the disentangled loss.
    # All predictions are computed using a corner
    # or chamfers loss depending on chamfer_pose!
    # Recommened to keep this weight small: [0.05, 0.5]
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_JOINT = 1.0

    # sgd, adam, adam+amsgrad, adamw, adamw+amsgrad
    cfg.SOLVER.TYPE = 'sgd'

    cfg.MODEL.RESNETS.TORCHVISION = True
    cfg.TEST.DETECTIONS_PER_IMAGE = 100

    cfg.TEST.VISIBILITY_THRES = 1/2.0
    cfg.TEST.TRUNCATION_THRES = 1/2.0

    # If ORACLE2D is True, the ocacle 2d bboxes and categories will be loaded when evaluation. 
    cfg.TEST.ORACLE2D = True
    cfg.TEST.CAT_MODE = "base" # "base" or "novel" or "all"

    cfg.INPUT.RANDOM_FLIP = "horizontal"
    cfg.INPUT.TRAIN_SET_PERCENTAGE = 1.0
    # When True, we will use localization uncertainty
    # as the new IoUness score in the RPN.
    cfg.MODEL.RPN.OBJECTNESS_UNCERTAINTY = 'IoUness'

    # If > 0.0 this is the scaling factor that will be applied to
    # an RoI 2D box before doing any pooling to give more context. 
    # Ex. 1.5 makes width and height 50% larger. 
    cfg.MODEL.ROI_CUBE_HEAD.SCALE_ROI_BOXES = 0.0

    # weight path specifically for pretraining (no checkpointables will be loaded)
    cfg.MODEL.WEIGHTS_PRETRAIN = ''