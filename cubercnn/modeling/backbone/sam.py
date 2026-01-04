from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import LastLevelMaxPool, FPN
from detectron2.modeling.backbone.vit import SimpleFeaturePyramid
import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F

from pathlib import Path
from urllib.request import urlretrieve
from segment_anything import sam_model_registry
import numpy as np
import einops as E
import unittest

# reference: https://github.com/mbanani/probe3d/blob/c52d00b069d949b2f00c544d4991716df68d5233/evals/models/sam.py
class SAMBackbone(Backbone):
    def __init__(self, cfg, input_shape, checkpoint="facebook/vit-mae-base", output="dense", layer=-1, return_multilayer=False, out_feature="last_feat",):
        super().__init__()

        assert output in ["cls", "gap", "dense", "dense-cls"]
        self.output = output

        # get model
        ckpt_file = "sam_vit_b_01ec64.pth"
        ckpt_path = Path("checkpoints")  / ckpt_file
        
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not ckpt_path.exists():
            download_path = (
                f"https://dl.fbaipublicfiles.com/segment_anything/{ckpt_file}"
            )
            urlretrieve(download_path, ckpt_path)
        
        sam = sam_model_registry['vit_b'](checkpoint=ckpt_path)
        vit = sam.image_encoder

        feat_dim = vit.neck[0].in_channels
        emb_h, emb_w = vit.pos_embed.shape[1:3]
        self.patch_size = vit.patch_embed.proj.kernel_size[0]
        self.image_size = (emb_h * self.patch_size, emb_w * self.patch_size)
        assert self.patch_size == 16

        self.vit = vit


        num_layers = len(self.vit.blocks)
        multilayers = [
            num_layers // 4 - 1,
            num_layers // 2 - 1,
            num_layers // 4 * 3 - 1,
            num_layers - 1,
        ]

        if return_multilayer:
            self.feat_dim = [feat_dim, feat_dim, feat_dim, feat_dim]
            self.multilayers = multilayers
        else:
            self.feat_dim = feat_dim
            layer = multilayers[-1] if layer == -1 else layer
            self.multilayers = [layer]

        # define layer name (for logging)
        self.layer = "-".join(str(_x) for _x in self.multilayers)

        self._out_feature_channels = {out_feature: feat_dim}
        self._out_feature_strides = {out_feature: self.patch_size}
        self._out_features = [out_feature]

    def resize_pos_embed(self, image_size):
        # get embed size
        h, w = image_size
        h = h // self.patch_size
        w = w // self.patch_size

        # resize embed
        pos_embed = self.vit.pos_embed.data.permute(0, 3, 1, 2)
        pos_embed = torch.nn.functional.interpolate(
            pos_embed, size=(h, w), mode="bicubic"
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        self.vit.pos_embed.data = pos_embed
        self.image_size = image_size

    def forward(self, x):
        _, _, h, w = x.shape
        assert h % self.patch_size == 0 and w % self.patch_size == 0, f"{h}, {w}"
        
        if h != self.image_size[0] or w != self.image_size[1]:
            self.resize_pos_embed(image_size=(h, w))

        # run vit
        x = self.vit.patch_embed(x)
        if self.vit.pos_embed is not None:
            x = x + self.vit.pos_embed

        embeds = []
        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if i in self.multilayers:
                embeds.append(x)
                if len(embeds) == len(self.multilayers):
                    break

        # feat shape is batch x feat_dim x height x width
        embeds = [_emb.permute(0, 3, 1, 2).contiguous() for _emb in embeds]
        outputs = {self._out_features[i]: embeds[i] for i in range(len(self.multilayers))}
        return outputs


@BACKBONE_REGISTRY.register()
def build_sam_backbone(cfg, input_shape: ShapeSpec, priors=None):
    output = cfg.MODEL.SAM.OUTPUT
    layer = cfg.MODEL.SAM.LAYER
    return_multilayer = cfg.MODEL.SAM.RETURN_MULTILAYER

    bottom_up = SAMBackbone(
        cfg,
        input_shape,
        output=output,
        layer=layer,
        return_multilayer=return_multilayer,
    )

    in_feature = cfg.MODEL.FPN.IN_FEATURE
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    scale_factors = (4.0, 2.0, 1.0, 0.5)
    backbone = SimpleFeaturePyramid(
        net=bottom_up,
        in_feature=in_feature,
        out_channels=out_channels,
        scale_factors=scale_factors,
        norm=cfg.MODEL.FPN.NORM,
        top_block=None,
        square_pad=cfg.MODEL.FPN.SQUARE_PAD
    )
    return backbone


class TestSAMBackbone(unittest.TestCase):
    def setUp(self):
        # Mock configuration
        self.cfg = type('', (), {})()
        self.cfg.MODEL = type('', (), {})()
        self.cfg.MODEL.SAM = type('', (), {})()
        self.cfg.MODEL.SAM.OUTPUT = "dense"
        self.cfg.MODEL.SAM.LAYER = -1
        self.cfg.MODEL.SAM.RETURN_MULTILAYER = False
        self.cfg.MODEL.FPN = type('', (), {})()
        self.cfg.MODEL.FPN.IN_FEATURE = 'last_feat'
        self.cfg.MODEL.FPN.OUT_CHANNELS = 256
        self.cfg.MODEL.FPN.NORM = "LN"
        self.cfg.MODEL.FPN.FUSE_TYPE = "sum"
        self.cfg.MODEL.FPN.SQUARE_PAD = 1024
        self.input_shape = ShapeSpec(channels=3, height=1024, width=1024)

    def test_sam_backbone_forward(self):
        # Create the backbone
        backbone = build_sam_backbone(self.cfg, self.input_shape)
        # Generate a random input tensor
        x = torch.randn(2, 3, 1024, 1024)
        # Run forward pass
        outputs = backbone(x)
        print(backbone.net.output_shape())
        for key, output in outputs.items():
            print(key, output.shape)

        # print(backbone.net.vit)


if __name__ == "__main__":
    unittest.main()