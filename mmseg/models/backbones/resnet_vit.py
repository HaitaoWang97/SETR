import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_plugin_layer,
                      constant_init, kaiming_init)
from mmcv.runner import load_checkpoint
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmseg.utils import get_root_logger
from ..builder import BACKBONES
from ..utils import ResLayer
from mmseg.models.backbones.vit import VisionTransformer

@BACKBONES.register_module()
class ResNetVit(nn.Module):
    """
    use resNet stem to get a feature map for latter transformer block
    resNet stem downsaple 4x and transformer patch is set as 2 or 4 (maybe 2 cause cuda error)
    transformer block num is set as 12
    """
    def __init__(self,
                 in_channels=3,
                 stem_channels=64,
                 embed_channels=256,
                 img_size=192,
                 patch=2,
                 trans_depth=12,
                 num_heads=8,
                 style='pytorch',
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False):
        super(ResNetVit, self).__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.stem_channels = stem_channels
        self.img_size = img_size
        self.patch = patch
        self.trans_depth = trans_depth
        self.num_heads = num_heads
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.stem = nn.Sequential(
            build_conv_layer(
                self.conv_cfg,
                self.in_channels,
                self.stem_channels // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False),
            build_norm_layer(self.norm_cfg, self.stem_channels // 2)[1],
            nn.ReLU(inplace=True),
            build_conv_layer(
                self.conv_cfg,
                self.stem_channels // 2,
                self.stem_channels // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            build_norm_layer(self.norm_cfg, self.stem_channels // 2)[1],
            nn.ReLU(inplace=True),
            build_conv_layer(
                self.conv_cfg,
                self.stem_channels // 2,
                self.stem_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            build_norm_layer(self.norm_cfg, self.stem_channels)[1],
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.vit = VisionTransformer(model_name='vit_feature_map_192', img_size=self.img_size, patch_size=self.patch,
                                     in_chans=self.stem_channels, embed_dim=self.embed_channels, depth=self.trans_depth,
                                     num_heads=self.num_heads)
    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.stem(x)
        #print("=================== input of trans block ", x.shape, "===================")
        out = self.vit(x)
        return out
