"""Pixel-space conditional diffusion; no VAE, DiT or downloaded weights."""
from ..self_supervised.models import SmallUNet, diffusion_schedule, predict_tiled


def build_model(config):
    """Create an epsilon predictor taking [noisy_target, observed_image]."""
    return SmallUNet(base_channels=int(config["model"]["base_channels"]), diffusion=True)


def alpha_bar(config):
    return diffusion_schedule(timesteps=int(config["diffusion"]["timesteps"]), schedule="cosine")[2]


__all__ = ["build_model", "alpha_bar", "predict_tiled"]
