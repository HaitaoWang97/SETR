from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
from itertools import repeat
from torch._six import container_abcs
import warnings
from ..backbones.vit import VisionTransformer
from mmcv.cnn import build_norm_layer

from ..builder import NECKS
# from .helpers import load_pretrained
# from .layers import DropPath, to_2tuple, trunc_normal_

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


@NECKS.register_module()
class VitUpV2(nn.Module):
    def __init__(self, model_name=None, img_size=48, patch_size=1, in_chans=2048, embed_dim=256, depth=12, num_heads=8, num_classes=19,
                 norm_cfg=None, **kwargs):
        super(VitUpV2, self).__init__(**kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.norm_cfg = norm_cfg

        self.vit = VisionTransformer(model_name=model_name, img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_chans,
                                     embed_dim=self.embed_dim, depth=self.depth, num_heads=self.num_heads,
                                     num_classes=self.num_classes, norm_cfg=self.norm_cfg)
        # self.conv_0 = nn.Conv2d(
        #     self.embed_dim, 1024, kernel_size=1, stride=1, padding=0)
        self.conv_1 = nn.Conv2d(
            256, 128, kernel_size=3, stride=1, padding=2)
        self.conv_2 = nn.Conv2d(
            512, 128, kernel_size=3, stride=1, padding=2)
        self.conv_3 = nn.Conv2d(
            128, 128, kernel_size=1, stride=1, padding=0)
        self.conv_4 = nn.Conv2d(
            128, 128, kernel_size=1, stride=1, padding=0)

        # _, self.syncbn_fc_0 = build_norm_layer(self.norm_cfg, 1024)
        _, self.syncbn_fc_1 = build_norm_layer(self.norm_cfg, 128)
        _, self.syncbn_fc_2 = build_norm_layer(self.norm_cfg, 128)
        _, self.syncbn_fc_3 = build_norm_layer(self.norm_cfg, 128)
        _, self.syncbn_fc_4 = build_norm_layer(self.norm_cfg, 128)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        trans_out = self.vit(x[-1])
        # trans_out = self.conv_0(trans_out)
        # trans_out = self.syncbn_fc_0(trans_out)
        # trans_out = F.relu(trans_out, inplace=True)

        first_out = self.conv_1(x[0])
        first_out = self.syncbn_fc_1(first_out)
        first_out = F.relu(first_out, inplace=True)

        second_out = self.conv_2(x[1])
        second_out = self.syncbn_fc_2(second_out)
        second_out = F.relu(second_out, inplace=True)
        second_out = F.interpolate(second_out, size=second_out.shape[-1] * 2, mode='bilinear', align_corners=False)

        out = torch.add(first_out, second_out)
        out = self.conv_3(out)
        out = self.syncbn_fc_3(out)
        out = F.relu(out, inplace=True)
        out = F.interpolate(out, size=out.shape[-1] * 2, mode='bilinear', align_corners=False)
        out = self.conv_4(out)
        out = self.syncbn_fc_4(out)
        out = F.relu(out, inplace=True)
        out = F.interpolate(out, size=out.shape[-1] * 2, mode='bilinear', align_corners=False)

        print("======================", out.shape,"================")    # 768x768x128
        print("======================", trans_out.shape, "================") # 48x48x512
        return tuple([out, trans_out])