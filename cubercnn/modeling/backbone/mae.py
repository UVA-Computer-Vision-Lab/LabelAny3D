from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import LastLevelMaxPool, FPN
from detectron2.modeling.backbone.vit import SimpleFeaturePyramid
import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
from transformers import ViTMAEForPreTraining
from transformers.models.vit_mae.modeling_vit_mae import (
    get_2d_sincos_pos_embed_from_grid,
)
import numpy as np
import einops as E
import unittest
from cubercnn.modeling.backbone.dino import tokens_to_output

# reference: https://github.com/mbanani/probe3d/blob/c52d00b069d949b2f00c544d4991716df68d5233/evals/models/mae.py
class MAEBackbone(Backbone):
    def __init__(self, cfg, input_shape, checkpoint="facebook/vit-mae-base", output="dense", layer=-1, return_multilayer=False, out_feature="last_feat",):
        super().__init__()

        # get model
        self.checkpoint_name = checkpoint.split("/")[1]
        self.vit = ViTMAEForPreTraining.from_pretrained(checkpoint).vit

        assert output in ["cls", "gap", "dense", "dense-cls"]
        self.output = output
        self.patch_size = self.vit.config.patch_size

        self.image_size = self.vit.embeddings.patch_embeddings.image_size
        self.feat_h = self.image_size[0] // self.patch_size
        self.feat_w = self.image_size[1] // self.patch_size

        feat_dim = self.vit.config.hidden_size

        num_layers = len(self.vit.encoder.layer)
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
        assert image_size[0] % self.patch_size == 0
        assert image_size[1] % self.patch_size == 0
        self.feat_h = image_size[0] // self.patch_size
        self.feat_w = image_size[1] // self.patch_size
        embed_dim = self.vit.config.hidden_size
        self.vit.embeddings.patch_embeddings.image_size = image_size
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim, (self.feat_h, self.feat_w), add_cls_token=True
        )
        # there should be an easier way ... TODO
        device = self.vit.embeddings.patch_embeddings.projection.weight.device
        self.vit.embeddings.position_embeddings = nn.Parameter(
            torch.from_numpy(pos_embed).float().unsqueeze(0).to(device=device),
            requires_grad=False,
        )

    def embed_forward(self, embedder, pixel_values):
        # No masking here ...
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = embedder.patch_embeddings(pixel_values)

        # add position embeddings w/o cls token
        embeddings = embeddings + embedder.position_embeddings[:, 1:, :]

        # append cls token
        cls_token = embedder.cls_token + embedder.position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings

    def forward(self, images):
        # check if positional embeddings are correct
        if self.image_size != images.shape[-2:]:
            self.resize_pos_embed(images.shape[-2:])

        # from MAE implementation
        head_mask = self.vit.get_head_mask(None, self.vit.config.num_hidden_layers)

        # ---- hidden ----
        embedding_output = self.embed_forward(self.vit.embeddings, images)
        encoder_outputs = self.vit.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=self.vit.config.output_attentions,
            output_hidden_states=True,
            return_dict=self.vit.config.return_dict,
        )

        outputs = {}
        for idx, layer_i in enumerate(self.multilayers):
            x_i = encoder_outputs.hidden_states[layer_i]
            x_i = tokens_to_output(
                self.output, x_i[:, 1:], x_i[:, 0], (self.feat_h, self.feat_w)
            )
            outputs[self._out_features[idx]] = x_i

        return outputs


@BACKBONE_REGISTRY.register()
def build_mae_backbone(cfg, input_shape: ShapeSpec, priors=None):
    checkpoint = cfg.MODEL.MAE.CHECKPOINT
    output = cfg.MODEL.MAE.OUTPUT
    layer = cfg.MODEL.MAE.LAYER
    return_multilayer = cfg.MODEL.MAE.RETURN_MULTILAYER

    bottom_up = MAEBackbone(
        cfg,
        input_shape,
        checkpoint=checkpoint,
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

def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """
    COPIED FROM TRANSFORMERS PACKAGE AND EDITED TO ALLOW FOR DIFFERENT WIDTH-HEIGHT
    Create 2D sin/cos positional embeddings.

    Args:
        embed_dim (`int`):
            Embedding dimension.
        grid_size (`int`):
            The grid height and width.
        add_cls_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add a classification (CLS) token.

    Returns:
        (`torch.FloatTensor` of shape (grid_size*grid_size, embed_dim) or
        (1+grid_size*grid_size, embed_dim): the
        position embeddings (with or without classification token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

class TestMAEBackbone(unittest.TestCase):
    def setUp(self):
        # Mock configuration
        self.cfg = type('', (), {})()
        self.cfg.MODEL = type('', (), {})()
        self.cfg.MODEL.MAE = type('', (), {})()
        self.cfg.MODEL.MAE.CHECKPOINT = "facebook/vit-mae-base"
        self.cfg.MODEL.MAE.OUTPUT = "dense"
        self.cfg.MODEL.MAE.LAYER = -1
        self.cfg.MODEL.MAE.RETURN_MULTILAYER = False
        self.cfg.MODEL.FPN = type('', (), {})()
        self.cfg.MODEL.FPN.IN_FEATURE = 'last_feat'
        self.cfg.MODEL.FPN.OUT_CHANNELS = 256
        self.cfg.MODEL.FPN.NORM = "LN"
        self.cfg.MODEL.FPN.FUSE_TYPE = "sum"
        self.cfg.MODEL.FPN.SQUARE_PAD = 1024
        self.input_shape = ShapeSpec(channels=3, height=1024, width=1024)

    def test_mae_backbone_forward(self):
        # Create the backbone
        backbone = build_mae_backbone(self.cfg, self.input_shape)
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