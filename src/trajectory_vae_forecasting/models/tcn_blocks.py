import torch
from torch import nn
import torch.nn.functional as F


class ResidualTemporalBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=1,
            padding=(dilation, 1),
            dilation=(dilation, 1),
        )
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=1,
            padding=(dilation, 1),
            dilation=(dilation, 1),
        )
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.dropout(F.silu(self.norm1(self.conv1(x))))
        out = self.dropout(F.silu(self.norm2(self.conv2(out))))
        return out + residual
