from __future__ import annotations

import torch
from torch import Tensor, nn

from models.time_embedding import SinusoidalTimeEmbedding


class SimpleUNet(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64, time_dim: int = 128) -> None:
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, base_channels),
            nn.SiLU(),
            nn.Linear(base_channels, base_channels),
        )

        self.in_block = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
        )
        self.mid_block = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
        )
        self.out_block = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        h = self.in_block(x)
        t_emb = self.time_proj(self.time_emb(t)).unsqueeze(-1).unsqueeze(-1)
        h = h + t_emb
        h = self.mid_block(h)
        return self.out_block(h)
