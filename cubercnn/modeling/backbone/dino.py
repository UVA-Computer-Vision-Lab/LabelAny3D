from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import LastLevelMaxPool, FPN
from detectron2.modeling.backbone.vit import SimpleFeaturePyramid
import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
import einops as E
import unittest

# reference: https://github.com/mbanani/probe3d/blob/c52d00b069d949b2f00c544d4991716df68d5233/evals/models/dino.py
class DINOBackbone(Backbone):
    def __init__(self, cfg, input_shape, dino_name="dino", model_name="vitb16", output="dense", layer=-1, return_multilayer=False, out_feature="last_feat",):
        super().__init__()
        feat_dims = {
            "vitb8": 768,
            "vitb16": 768,
            "vitb14": 768,
            "vitb14_reg": 768,
            "vitl14": 1024,
            "vitg14": 1536,
        }

        # get model
        self.model_name = dino_name
        self.checkpoint_name = f"{dino_name}_{model_name}"
        dino_vit = torch.hub.load(f"facebookresearch/{dino_name}", self.checkpoint_name)
        self.vit = dino_vit
        self.has_registers = "_reg" in model_name

        assert output in ["cls", "gap", "dense", "dense-cls"]
        self.output = output
        self.patch_size = self.vit.patch_embed.proj.kernel_size[0]

        feat_dim = feat_dims[model_name]
        feat_dim = feat_dim * 2 if output == "dense-cls" else feat_dim

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

    def forward(self, images):
        h, w = images.shape[-2:]
        h, w = h // self.patch_size, w // self.patch_size

        if self.model_name == "dinov2":
            x = self.vit.prepare_tokens_with_masks(images, None)
        else:
            x = self.vit.prepare_tokens(images)

        embeds = []
        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if i in self.multilayers:
                embeds.append(x)
                if len(embeds) == len(self.multilayers):
                    break

        num_spatial = h * w
        outputs = {}
        for idx, x_i in enumerate(embeds):
            cls_tok = x_i[:, 0]
            spatial = x_i[:, -1 * num_spatial:]
            x_i = tokens_to_output(self.output, spatial, cls_tok, (h, w))
            outputs[self._out_features[idx]] = x_i

        return outputs


@BACKBONE_REGISTRY.register()
def build_dino_backbone(cfg, input_shape: ShapeSpec, priors=None):
    dino_name = cfg.MODEL.DINO.NAME
    model_name = cfg.MODEL.DINO.MODEL_NAME
    output = cfg.MODEL.DINO.OUTPUT
    layer = cfg.MODEL.DINO.LAYER
    return_multilayer = cfg.MODEL.DINO.RETURN_MULTILAYER

    bottom_up = DINOBackbone(
        cfg,
        input_shape,
        dino_name=dino_name,
        model_name=model_name,
        output=output,
        layer=layer,
        return_multilayer=return_multilayer,
    )

    in_feature = cfg.MODEL.FPN.IN_FEATURE
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    scale_factors = (2.0, 1.0, 0.5)
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

def tokens_to_output(output_type, dense_tokens, cls_token, feat_hw):
    if output_type == "cls":
        assert cls_token is not None
        output = cls_token
    elif output_type == "gap":
        output = dense_tokens.mean(dim=1)
    elif output_type == "dense":
        h, w = feat_hw
        dense_tokens = E.rearrange(dense_tokens, "b (h w) c -> b c h w", h=h, w=w)
        output = dense_tokens.contiguous()
    elif output_type == "dense-cls":
        assert cls_token is not None
        h, w = feat_hw
        dense_tokens = E.rearrange(dense_tokens, "b (h w) c -> b c h w", h=h, w=w)
        cls_token = cls_token[:, :, None, None].repeat(1, 1, h, w)
        output = torch.cat((dense_tokens, cls_token), dim=1).contiguous()
    else:
        raise ValueError()

    return output

class TestDINOBackbone(unittest.TestCase):
    def setUp(self):
        # Mock configuration
        self.cfg = type('', (), {})()
        self.cfg.MODEL = type('', (), {})()
        self.cfg.MODEL.DINO = type('', (), {})()
        self.cfg.MODEL.DINO.NAME = "dino"
        self.cfg.MODEL.DINO.MODEL_NAME = "vitb16"
        self.cfg.MODEL.DINO.OUTPUT = "dense"
        self.cfg.MODEL.DINO.LAYER = -1
        self.cfg.MODEL.DINO.RETURN_MULTILAYER = False
        self.cfg.MODEL.FPN = type('', (), {})()
        self.cfg.MODEL.FPN.IN_FEATURE = 'last_feat'
        self.cfg.MODEL.FPN.OUT_CHANNELS = 256
        self.cfg.MODEL.FPN.NORM = "LN"
        self.cfg.MODEL.FPN.FUSE_TYPE = "sum"
        self.input_shape = ShapeSpec(channels=3, height=512, width=512)

    def test_dino_backbone_forward(self):
        # Create the backbone
        backbone = build_dino_backbone(self.cfg, self.input_shape)
        # Generate a random input tensor
        x = torch.randn(1, 3, 512, 512)
        # Run forward pass
        outputs = backbone(x)
        print(backbone.net.output_shape())
        for key, output in outputs.items():
            print(key, output.shape)

        # print(backbone.net.vit)


if __name__ == "__main__":
    unittest.main()