from torch import nn
import torch

from .tcn_blocks import ResidualTemporalBlock2D


class TCN2DEncoder(nn.Module):
    def __init__(self, in_channels: int = 5, channels: tuple[int, ...] = (32, 64, 128, 256), dropout: float = 0.1):
        super().__init__()
        layers = []
        curr = in_channels
        for i, out in enumerate(channels):
            layers.append(ResidualTemporalBlock2D(curr, out, dilation=2**i, dropout=dropout))
            curr = out
        self.blocks = nn.ModuleList(layers)
        self.final = nn.Conv2d(curr, curr, kernel_size=1)
        self.out_channels = curr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.permute(0, 3, 1, 2)
        for block in self.blocks:
            out = block(out)
        return self.final(out)
