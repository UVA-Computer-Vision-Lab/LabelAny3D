# Copyright (c) Meta Platforms, Inc. and affiliates
import sys
sys.path.append('./GroundingDino/')
from cubercnn.modeling.roi_heads.roi_heads import *

from torchvision.ops import nms

# GroundingDINO imports
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span
from transformers import AutoTokenizer


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu", weights_only=True)
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    return model


@ROI_HEADS_REGISTRY.register()
class ROIHeads3DGDINO(ROIHeads3D):

    @configurable
    def __init__(
        self,
        *,
        ignore_thresh: float,
        cube_head: nn.Module,
        cube_pooler: nn.Module,
        loss_w_3d: float,
        loss_w_xy: float,
        loss_w_z: float,
        loss_w_dims: float,
        loss_w_pose: float,
        loss_w_joint: float,
        use_confidence: float,
        inverse_z_weight: bool,
        z_type: str,
        pose_type: str,
        cluster_bins: int,
        priors = None,
        dims_priors_enabled = None,
        dims_priors_func = None,
        disentangled_loss=None,
        virtual_depth=None,
        virtual_focal=None,
        test_scale=None,
        allocentric_pose=None,
        chamfer_pose=None,
        scale_roi_boxes=None,
        **kwargs,
    ):
        super().__init__(
            ignore_thresh=ignore_thresh,
            cube_head=cube_head,
            cube_pooler=cube_pooler,
            loss_w_3d=loss_w_3d,
            loss_w_xy=loss_w_xy,
            loss_w_z=loss_w_z,
            loss_w_dims=loss_w_dims,
            loss_w_pose=loss_w_pose,
            loss_w_joint=loss_w_joint,
            use_confidence=use_confidence,
            inverse_z_weight=inverse_z_weight,
            z_type=z_type,
            pose_type=pose_type,
            cluster_bins=cluster_bins,
            priors=priors,
            dims_priors_enabled=dims_priors_enabled,
            dims_priors_func=dims_priors_func,
            disentangled_loss=disentangled_loss,
            virtual_depth=virtual_depth,
            virtual_focal=virtual_focal,
            test_scale=test_scale,
            allocentric_pose=allocentric_pose,
            chamfer_pose=chamfer_pose,
            scale_roi_boxes=scale_roi_boxes,
            **kwargs
        )

        self.groundingdino_model = load_model(
            "./configs/GroundingDINO_SwinB_cfg.py", 
            "./checkpoints/groundingdino_swinb_cogcoor.pth", 
            cpu_only=False
        )

    def forward(self, images, features, proposals, Ks, im_scales_ratio, targets=None, category_list=None):

        im_dims = [image.shape[1:] for image in images]

        # del images

        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        
        del targets

        if self.training:

            losses = self._forward_box(features, proposals)
            if self.loss_w_3d > 0:
                instances_3d, losses_cube = self._forward_cube(features, proposals, Ks, im_dims, im_scales_ratio)
                losses.update(losses_cube)

            return instances_3d, losses
        
        else:

            # when oracle is available, by pass the box forward.
            # simulate the predicted instances by creating a new 
            # instance for each passed in image.
            if isinstance(proposals, list) and ~np.any([isinstance(p, Instances) for p in proposals]):
                pred_instances = []
                for proposal, im_dim in zip(proposals, im_dims):
                    
                    pred_instances_i = Instances(im_dim)
                    pred_instances_i.pred_boxes = Boxes(proposal['gt_bbox2D'])
                    pred_instances_i.pred_classes =  proposal['gt_classes']
                    pred_instances_i.scores = torch.ones_like(proposal['gt_classes']).float()
                    pred_instances.append(pred_instances_i)
            else:
                pred_instances = self._forward_box(features, proposals)
            
            if category_list:
                filtered_texts = [ [cat]  for cat in  category_list]

            # Return empty Instances object if no valid text is found
            if not filtered_texts:
                target = Instances(pred_instances[0].image_size)
                target.pred_classes = torch.tensor([], dtype=torch.int64)  # Empty class tensor
                target.pred_boxes = Boxes(torch.tensor([], dtype=torch.float32).view(-1, 4))  # Empty boxes tensor
                target.scores = torch.tensor([], dtype=torch.float32)  # Empty scores tensor
                target = target.to(device=pred_instances[0].scores.device)       
            
            else:
                
                # use grounding dino prediction
                configs = {
                    "groundingdino_model": self.groundingdino_model,
                    "image": images[0][[2, 1, 0], :, :],
                    "text_prompt": filtered_texts,
                    "box_threshold": 0.001,
                    "text_threshold": 0.25,
                    "token_spans": None,
                    "cpu_only": False
                }
                
                ov_pred_instances = grounding_dino_inference_detector(configs)
                
                # init target
                target = Instances(pred_instances[0].image_size)
                
                # add classes, 2D boxes, scores
                class_names = ov_pred_instances["labels"]
                # h, w = pred_instances[0].image_size
                target.pred_classes = torch.tensor([filtered_texts.index([class_name])  for class_name in class_names])
                target.pred_boxes = Boxes( ov_pred_instances["bboxes"])
                # max_scores = [torch.max(score_tensor).item() for score_tensor in ov_pred_instances["scores"]]
                # target.scores = torch.tensor(max_scores).float()
                target.scores = ov_pred_instances["scores"]
                target = target.to(device=pred_instances[0].scores.device)

            if self.loss_w_3d > 0:
                pred_instances = self._forward_cube(features, [target,], Ks, im_dims, im_scales_ratio)
            return pred_instances, {}
        

def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    cap_list = [cat[0] for cat in caption ]
    caption = " . ".join(cap_list)
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + " ."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    all_logits = []
    
    # filter output
    if token_spans is None:
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        phrases_logits = get_phrase_logits_from_token_logits(logits, tokenized, tokenlizer, cap_list)
        filt_mask = phrases_logits.max(dim=1)[0] > box_threshold
        im_logits_filt = phrases_logits[filt_mask]
        boxes_filt = boxes[filt_mask].cpu()

        im_pred_scores, im_pred_classes = im_logits_filt.max(dim = -1)
        all_logits = im_pred_scores.cpu()
        pred_phrases = [cap_list[idx] for idx in im_pred_classes]
        
    else:
        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            model.tokenizer(caption),
            token_span=token_spans
        ).to(image.device) # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases
        
    return boxes_filt, pred_phrases, all_logits


def grounding_dino_inference_detector(config):
    image = config["image"]
    text_prompt = config["text_prompt"]
    box_threshold = config["box_threshold"]
    text_threshold = config["text_threshold"]
    token_spans = config["token_spans"]
    cpu_only = config["cpu_only"]
    groundingdino_model = config["groundingdino_model"]
    
    if token_spans is not None:
        text_threshold = None
        print("Using token_spans. Set the text_threshold to None.")
        
    boxes_filt, pred_phrases, all_logits = get_grounding_output(
        groundingdino_model, image, text_prompt, box_threshold, text_threshold, cpu_only, token_spans=eval(f"{token_spans}")
    )
    h, w = image.shape[1:]
    boxes_filt = box_cxcywh_to_xyxy(boxes_filt * torch.tensor([w, h, w, h]))
    nms_idx = nms(boxes_filt, all_logits, 0.5)
    all_logits = all_logits[nms_idx]
    boxes_filt = boxes_filt[nms_idx]
    pred_phrases = [pred_phrases[idx] for idx in nms_idx]
    ov_pred_instances = {}
    ov_pred_instances["scores"] = all_logits
    ov_pred_instances["bboxes"] = boxes_filt
    ov_pred_instances["labels"] = pred_phrases
    
    return ov_pred_instances
    

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def get_phrase_logits_from_token_logits(
    token_logits: torch.Tensor, tokenized: Dict, tokenizer: AutoTokenizer, cap_list: List
):
    if token_logits.dim() == 2:  # (num of query, 256)
        tokenized_phrases = tokenizer(cap_list, add_special_tokens=False)['input_ids']
        begin_id = 1
        phrase_logits = []
        ids = list(range(len(tokenized['input_ids'])))
        phrases_ids = []
        for phrase_tokens in tokenized_phrases:
            end_id = begin_id + len(phrase_tokens)
            assert phrase_tokens == tokenized['input_ids'][begin_id : end_id], "assert error!!!"
            phrases_ids.append(ids[begin_id : end_id])
            begin_id = end_id + 1
        for phrase_ids in phrases_ids:
            # import pdb;pdb.set_trace()
            phrase_logit = token_logits[:, phrase_ids].sum(dim=-1)
            phrase_logits.append(phrase_logit)
        phrase_logits = torch.stack(phrase_logits, dim=1)
        return phrase_logits
    else:
        raise NotImplementedError("token_logits must be 1-dim")