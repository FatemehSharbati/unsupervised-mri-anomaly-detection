"""Convolutional autoencoder for unsupervised anomaly detection in brain MRI.

The autoencoder is trained exclusively on healthy brain slices. At inference
time, regions with high reconstruction error are flagged as candidate
anomalies: the network never learned to reconstruct pathology, so abnormal
tissue stands out in the error map.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two 3x3 convolutions, each followed by BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BrainAutoencoder(nn.Module):
    """Autoencoder for 2D axial brain-MRI slices.

    Encoder: three conv blocks (16 -> 32 -> 64 channels), each followed by
    2x2 max-pooling, then global average pooling and a dense bottleneck of
    128 units.
    Decoder: mirrored transposed-convolution upsampling path ending with a
    1x1 convolution and sigmoid.

    Args:
        in_channels: input channels (1 for grayscale MRI).
        bottleneck_dim: size of the dense bottleneck (default 128).
        input_size: spatial size of square input slices; must be divisible
            by 8 (three 2x2 pooling steps).
    """

    def __init__(
        self,
        in_channels: int = 1,
        bottleneck_dim: int = 128,
        input_size: int = 64,
    ):
        super().__init__()
        if input_size % 8 != 0:
            raise ValueError("input_size must be divisible by 8")
        self.input_size = input_size
        feat = input_size // 8  # spatial extent after three pooling steps

        self.enc1 = ConvBlock(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(32, 64)
        self.pool3 = nn.MaxPool2d(2)

        self.to_bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, bottleneck_dim),
            nn.ReLU(inplace=True),
        )
        self.from_bottleneck = nn.Sequential(
            nn.Linear(bottleneck_dim, 64 * feat * feat),
            nn.ReLU(inplace=True),
        )
        self._feat = feat

        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(32, 32)
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(16, 16)
        self.up1 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(16, 16)
        self.out_conv = nn.Conv2d(16, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x = self.pool1(self.enc1(x))
        x = self.pool2(self.enc2(x))
        x = self.pool3(self.enc3(x))
        z = self.to_bottleneck(x)
        x = self.from_bottleneck(z).view(batch, 64, self._feat, self._feat)
        x = self.dec3(self.up3(x))
        x = self.dec2(self.up2(x))
        x = self.dec1(self.up1(x))
        return torch.sigmoid(self.out_conv(x))


def reconstruction_error_map(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Per-pixel reconstruction error, averaged over channels.

    Args:
        x: original slices, shape (B, C, H, W).
        x_hat: reconstructions, same shape as x.

    Returns:
        Error map of shape (B, 1, H, W). Higher values mark regions the
        autoencoder failed to reconstruct, i.e. candidate anomalies.
    """
    return ((x - x_hat) ** 2).mean(dim=1, keepdim=True)


def anomaly_scores(error_map: torch.Tensor) -> torch.Tensor:
    """Collapse an error map to one anomaly score per slice (mean error)."""
    return error_map.flatten(start_dim=1).mean(dim=1)
