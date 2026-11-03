"""Shared masked/diffusion networks; no pretrained natural-image weights required."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels):
    return next(n for n in (8, 4, 2, 1) if channels % n == 0)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension

    def forward(self, t):
        half = self.dimension // 2
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angles = t.float()[:, None] * frequencies[None, :]
        return torch.cat((angles.sin(), angles.cos()), dim=1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dimension=0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.time = nn.Linear(time_dimension, out_channels) if time_dimension else None

    def forward(self, x, time=None):
        x = F.silu(self.norm1(self.conv1(x)))
        if self.time is not None:
            x = x + self.time(time)[:, :, None, None]
        return F.silu(self.norm2(self.conv2(x)))


class SmallUNet(nn.Module):
    """Two downsampling levels, with interpolation to support odd image sizes.

    In diffusion mode, inputs are [x_t, observed_image] and the output is epsilon.
    In masked mode, input/output are normalized single-channel intensity images.
    """

    def __init__(self, base_channels=16, diffusion=False):
        super().__init__()
        if int(base_channels) < 4:
            raise ValueError("model.base_channels must be at least 4")
        base_channels = int(base_channels)
        self.diffusion = diffusion
        time_dimension = base_channels * 4 if diffusion else 0
        self.time_embedding = (
            nn.Sequential(
                SinusoidalTimeEmbedding(time_dimension),
                nn.Linear(time_dimension, time_dimension), nn.SiLU(),
                nn.Linear(time_dimension, time_dimension),
            ) if diffusion else None
        )
        c = base_channels
        self.encoder1 = ConvBlock(2 if diffusion else 1, c, time_dimension)
        self.encoder2 = ConvBlock(c, c * 2, time_dimension)
        self.center = ConvBlock(c * 2, c * 4, time_dimension)
        self.decoder2 = ConvBlock(c * 6, c * 2, time_dimension)
        self.decoder1 = ConvBlock(c * 3, c, time_dimension)
        self.output = nn.Conv2d(c, 1, 1)

    def forward(self, x, t=None):
        if min(x.shape[-2:]) < 4:
            raise ValueError("Images must be at least 4 x 4 pixels")
        if self.diffusion and t is None:
            raise ValueError("A diffusion timestep is required")
        time = self.time_embedding(t) if self.diffusion else None
        e1 = self.encoder1(x, time)
        e2 = self.encoder2(F.avg_pool2d(e1, 2), time)
        center = self.center(F.avg_pool2d(e2, 2), time)
        d2 = self.decoder2(torch.cat((F.interpolate(center, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2), dim=1), time)
        d1 = self.decoder1(torch.cat((F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1), dim=1), time)
        return self.output(d1)


def diffusion_schedule(timesteps=100, schedule="cosine", beta_start=0.0001, beta_end=0.02):
    """Return beta, alpha and cumulative alpha; the last noise level is checked.

    A 100-step linear 1e-4..0.02 schedule is NOT sufficiently noisy at its end
    to initialize reverse sampling with pure Gaussian noise. Cosine is default.
    """
    timesteps = int(timesteps)
    if timesteps < 2:
        raise ValueError("diffusion.timesteps must be at least 2")
    if schedule == "cosine":
        x = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
        abar = torch.cos(((x / timesteps + 0.008) / 1.008) * math.pi / 2).square()
        abar = abar / abar[0]
        betas = (1 - abar[1:] / abar[:-1]).clamp(0.00001, 0.999).float()
    elif schedule == "linear":
        if not 0 < beta_start < beta_end < 1:
            raise ValueError("Linear beta schedule requires 0 < beta_start < beta_end < 1")
        betas = torch.linspace(beta_start, beta_end, timesteps)
    else:
        raise ValueError("diffusion.schedule must be cosine or linear")
    alphas = 1 - betas
    cumulative = torch.cumprod(alphas, dim=0)
    if cumulative[-1].item() > 0.01:
        raise ValueError("Final alpha_bar is too large for Gaussian initialization; use schedule: cosine or more linear timesteps")
    return betas, alphas, cumulative


def predict_tiled(model, image, t=None, tile_size=256, overlap=32):
    """Overlap-add network predictions. Diffusion keeps one global noisy image.

    Thus tiles do not independently sample incompatible noise at their borders.
    Tiling reduces activation memory, but can differ from full-image GroupNorm.
    """
    height, width = image.shape[-2:]
    tile_size, overlap = int(tile_size), int(overlap)
    if tile_size <= 0 or (height <= tile_size and width <= tile_size):
        return model(image, t)
    if tile_size < 16 or not 0 <= overlap < tile_size // 2:
        raise ValueError("Use tile_size >= 16 and 0 <= overlap < tile_size / 2")

    def origins(size):
        if size <= tile_size:
            return [0]
        return sorted(set(list(range(0, size - tile_size + 1, tile_size - overlap)) + [size - tile_size]))

    total = image.new_zeros((image.shape[0], 1, height, width))
    weights = image.new_zeros((1, 1, height, width))
    for y in origins(height):
        for x in origins(width):
            patch = image[..., y:y + tile_size, x:x + tile_size]
            prediction = model(patch, t)
            ph, pw = patch.shape[-2:]
            wy = torch.hann_window(ph, periodic=False, device=image.device).clamp_min(0.05)
            wx = torch.hann_window(pw, periodic=False, device=image.device).clamp_min(0.05)
            weight = (wy[:, None] * wx[None, :])[None, None, ...]
            total[..., y:y + ph, x:x + pw] += prediction * weight
            weights[..., y:y + ph, x:x + pw] += weight
    return total / weights
