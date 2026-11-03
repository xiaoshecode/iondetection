"""Small fully convolutional center-probability and subpixel-offset model."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU())


class PointNet(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        if not isinstance(channels, int) or channels < 4 or channels > 128:
            raise ValueError("channels must be an integer in [4,128]")
        self.channels = channels
        self.enc = _block(1, channels)
        self.low = _block(channels, channels * 2)
        self.fuse = _block(channels * 3, channels)
        self.heatmap = nn.Conv2d(channels, 1, 1)
        self.offset = nn.Conv2d(channels, 2, 1)
        nn.init.constant_(self.heatmap.bias, math.log(.01 / .99))

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 1 or min(image.shape[-2:]) < 2:
            raise ValueError("PointNet expects Bx1xHxW images with H,W >= 2")
        high = self.enc(image)
        low = self.low(F.avg_pool2d(high, 2))
        # A fixed 2x interpolation retains the global sampling lattice even
        # for odd-sized images; size-based resizing would stretch each tile.
        up = F.interpolate(low, scale_factor=2, mode="bilinear", align_corners=False)
        up = F.pad(up, (0, high.shape[-1] - up.shape[-1], 0, high.shape[-2] - up.shape[-2]), mode="replicate")
        features = self.fuse(torch.cat((high, up), 1))
        return self.heatmap(features), .5 * torch.tanh(self.offset(features))


def point_loss(logits, offsets, target, target_offsets, centers):
    """CenterNet-style focal heatmap plus offsets supervised only at real centers."""
    probability = torch.sigmoid(logits).clamp(1e-5, 1 - 1e-5)
    positive = (target == 1).to(logits.dtype)
    negative = 1 - positive
    positive_loss = -torch.log(probability) * (1 - probability).square() * positive
    negative_loss = -torch.log1p(-probability) * probability.square() * (1 - target).pow(4) * negative
    normalizer = centers.sum().clamp(min=1)
    detection = (positive_loss.sum() + negative_loss.sum()) / normalizer
    regression = (F.smooth_l1_loss(offsets, target_offsets, reduction="none") * centers).sum() / normalizer
    return detection + regression, detection, regression
