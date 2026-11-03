"""Signal-supported learned centroids and exact discrete image transforms."""
import numpy as np
import torch
from torch import nn


def transform_image(image, quarter_turns=0, reflect=False, dx=0, dy=0):
    """Return image and 3x3 forward map, for zero-based [x, y, 1] points.

    Rotation is np.rot90 counterclockwise in image coordinates; reflection is
    horizontal AFTER rotation; integer translation uses constant boundary fill.
    Inputs must have enough border margin to avoid clipping the object.
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or image.shape[0] != image.shape[1] or not np.isfinite(image).all():
        raise ValueError("Transforms require a finite square grayscale image")
    for name, value in (("quarter_turns", quarter_turns), ("dx", dx), ("dy", dy)):
        if not isinstance(value, (int, np.integer)):
            raise ValueError(name + " must be an integer")
    n = image.shape[0]
    if abs(dx) >= n or abs(dy) >= n:
        raise ValueError("Translation must be smaller than the image")
    rotation = np.array([[0., 1., 0.], [-1., 0., n - 1.], [0., 0., 1.]])
    matrix = np.linalg.matrix_power(rotation, int(quarter_turns) % 4)
    transformed = np.rot90(image, int(quarter_turns) % 4).copy()
    if reflect:
        transformed = transformed[:, ::-1].copy()
        matrix = np.array([[-1., 0., n - 1.], [0., 1., 0.], [0., 0., 1.]]) @ matrix
    fill = float(np.median(np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]))))
    shifted = np.full_like(transformed, fill)
    sx0, sx1 = max(0, -dx), min(n, n - dx)
    sy0, sy1 = max(0, -dy), min(n, n - dy)
    shifted[sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = transformed[sy0:sy1, sx0:sx1]
    matrix = np.array([[1., 0., dx], [0., 1., dy], [0., 0., 1.]]) @ matrix
    return shifted, matrix.astype(np.float32)


def invert_points(points, matrices):
    """Inverse-map one predicted [x,y] per orthogonal affine transform."""
    if points.ndim != 2 or points.shape[1] != 2 or matrices.shape != (len(points), 3, 3):
        raise ValueError("Expected points Bx2 and transforms Bx3x3")
    # D4 transforms are orthogonal; inverse translation then transpose rotation.
    return torch.bmm(matrices[:, :2, :2].transpose(1, 2),
                     (points - matrices[:, :2, 2]).unsqueeze(2)).squeeze(2)


def equivariance_loss(points, matrices, batch_size, views):
    if views < 2 or batch_size < 1 or len(points) != batch_size * views:
        raise ValueError("Equivariance requires at least two transformed views per sample")
    canonical = invert_points(points, matrices).reshape(batch_size, views, 2)
    return ((canonical - canonical.mean(dim=1, keepdim=True)) ** 2).mean()


class SignalCentroidNet(nn.Module):
    """Bounded learned reweighting of foreground pixels; no position channels.

    Nonnegative signal support and a positive weight floor prevent the network
    from cancelling every foreground pixel and returning a fixed patch center.
    A uniform learned map reduces to an image-dependent foreground centroid,
    which is an intentional baseline, not a successful learned localization.
    """
    def __init__(self, channels=16, weight_floor=0.1):
        super().__init__()
        if not isinstance(channels, int) or channels < 4 or channels > 128:
            raise ValueError("channels must be an integer from 4 to 128")
        if not np.isfinite(weight_floor) or not 0 < weight_floor <= 1:
            raise ValueError("weight_floor must be in (0,1]")
        self.channels = channels
        self.weight_floor = float(weight_floor)
        self.features = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(channels, 1, 1))
        # Start from a physically interpretable centroid. No coordinates used.
        nn.init.zeros_(self.features[-1].weight)
        nn.init.zeros_(self.features[-1].bias)

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 1 or min(image.shape[-2:]) < 5:
            raise ValueError("Expected Bx1xHxW grayscale patches, at least 5 pixels wide")
        background = torch.quantile(image.flatten(2), 0.2, dim=2).unsqueeze(-1).unsqueeze(-1)
        signal = (image - background).clamp_min(0.)
        weights = signal * (self.weight_floor + torch.sigmoid(self.features(image)))
        denominator = weights.sum(dim=(2, 3)).clamp_min(1e-12)
        h, w = image.shape[-2:]
        yy, xx = torch.meshgrid(torch.arange(h, device=image.device, dtype=image.dtype),
                                torch.arange(w, device=image.device, dtype=image.dtype), indexing="ij")
        x = (weights * xx).sum(dim=(2, 3)) / denominator
        y = (weights * yy).sum(dim=(2, 3)) / denominator
        return torch.cat((x, y), dim=1)
