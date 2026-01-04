from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.vit import SimpleFeaturePyramid
import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
import numpy as np
import einops as E
import unittest
from cubercnn.modeling.backbone.dino import tokens_to_output
from cubercnn.modeling.backbone.clip import resize_pos_embed
# from dino import tokens_to_output
# from clip import resize_pos_embed

# reference: https://github.com/mbanani/probe3d/blob/c52d00b069d949b2f00c544d4991716df68d5233/evals/models/midas_final.py
class MIDASBackbone(Backbone):
    def __init__(self, cfg, input_shape, output="dense", layer=-1, return_multilayer=False, out_feature="last_feat",):
        super().__init__()

        # get model
        midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        self.vit = midas.pretrained.model

        # set parameters for feature extraction
        self.image_size = (384, 384)
        self.patch_size = 16
        self.output = output
        feat_dim = 1024
        self.feat_dim = 1024

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


    def forward(self, x):
        # update shapes
        h, w = x.shape[2:]
        emb_hw = (h // self.patch_size, w // self.patch_size)
        # assert h == w, f"BeIT can only handle square images, not ({h}, {w})."
        if (h, w) != self.image_size:
            self.image_size = (h, w)
            self.vit.patch_embed.img_size = (h, w)
            # import pdb;pdb.set_trace()
            self.vit.pos_embed.data = resize_pos_embed(self.vit.pos_embed[0], emb_hw, True)[None]

        # actual forward from beit
        x = self.vit.patch_embed(x)
        x = torch.cat((self.vit.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.vit.pos_embed

        x = self.vit.norm_pre(x)

        embeds = []
        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if i in self.multilayers:
                embeds.append(x)
                if i == self.layer:
                    break

        # map tokens to output
        outputs = {}
        for i, x_i in enumerate(embeds):
            x_i = tokens_to_output(self.output, x_i[:, 1:], x_i[:, 0], emb_hw)
            outputs[self._out_features[i]] = x_i

        return outputs


@BACKBONE_REGISTRY.register()
def build_midas_backbone(cfg, input_shape: ShapeSpec, priors=None):
    output = cfg.MODEL.MIDAS.OUTPUT
    layer = cfg.MODEL.MIDAS.LAYER
    return_multilayer = cfg.MODEL.MIDAS.RETURN_MULTILAYER

    bottom_up = MIDASBackbone(
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


class TestMIDASBackbone(unittest.TestCase):
    def setUp(self):
        # Mock configuration
        self.cfg = type('', (), {})()
        self.cfg.MODEL = type('', (), {})()
        self.cfg.MODEL.MIDAS = type('', (), {})()
        self.cfg.MODEL.MIDAS.OUTPUT = "dense"
        self.cfg.MODEL.MIDAS.LAYER = -1
        self.cfg.MODEL.MIDAS.RETURN_MULTILAYER = False
        self.cfg.MODEL.FPN = type('', (), {})()
        self.cfg.MODEL.FPN.IN_FEATURE = 'last_feat'
        self.cfg.MODEL.FPN.OUT_CHANNELS = 256
        self.cfg.MODEL.FPN.NORM = "LN"
        self.cfg.MODEL.FPN.FUSE_TYPE = "sum"
        self.cfg.MODEL.FPN.SQUARE_PAD = 1024
        self.input_shape = ShapeSpec(channels=3, height=1024, width=1024)

    def test_midas_backbone_forward(self):
        # Create the backbone
        backbone = build_midas_backbone(self.cfg, self.input_shape)
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