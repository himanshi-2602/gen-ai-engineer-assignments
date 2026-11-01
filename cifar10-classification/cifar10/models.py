"""Model architectures: a plain CNN baseline and the same CNN with residual blocks.

Both models share one backbone (`CifarCNN`). The only difference is the block used in
each stage, so a comparison between them isolates the effect of the skip connection.
"""
from __future__ import annotations

from typing import Sequence, Type

import torch
from torch import nn


class PlainBlock(nn.Module):
    """Two 3x3 conv -> BatchNorm -> ReLU layers (baseline)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class ResidualBlock(nn.Module):
    """Simplified ResNet basic block (Part 2, Option A).

    out = ReLU( BN(conv(ReLU(BN(conv(x))))) + shortcut(x) )

    The shortcut is the identity when the channel count is unchanged, otherwise a
    1x1 conv + BatchNorm projection so the two branches can be added.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class CifarCNN(nn.Module):
    """Small CNN for 32x32 images.

    3 stages of [block -> MaxPool(2) -> Dropout], halving the resolution each time
    (32 -> 16 -> 8 -> 4), then a classification head:
    global average pooling -> Dropout -> Linear(num_classes).
    """

    def __init__(
        self,
        block: Type[nn.Module],
        widths: Sequence[int] = (64, 128, 256),
        num_classes: int = 10,
        dropout: float = 0.1,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        stages = []
        in_channels = 3
        for width in widths:
            stages += [block(in_channels, width), nn.MaxPool2d(2), nn.Dropout(dropout)]
            in_channels = width
        self.features = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(head_dropout),
            nn.Linear(in_channels, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


BLOCKS = {"baseline": PlainBlock, "residual": ResidualBlock}


def build_model(name: str, **kwargs) -> CifarCNN:
    """Create a model by name: 'baseline' (plain CNN) or 'residual' (CNN with residual blocks)."""
    if name not in BLOCKS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(BLOCKS)}")
    return CifarCNN(BLOCKS[name], **kwargs)
