from __future__ import annotations

import torch
from torch import nn


class GraphConvLayer(nn.Module):
    """Simple message passing layer without torch_geometric dependency."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src = edge_index[0]
        dst = edge_index[1]

        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])

        deg = torch.zeros((x.shape[0],), dtype=x.dtype, device=x.device)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        deg = deg.clamp(min=1.0).unsqueeze(-1)
        mean_neigh = agg / deg

        return self.lin_self(x) + self.lin_neigh(mean_neigh)


class AnchorGNN(nn.Module):
    """Node classifier for anchor prediction (0=normal,1=left,2=right)."""

    def __init__(
        self,
        in_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_classes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList([GraphConvLayer(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index)
            h = norm(h)
            h = torch.relu(h)
            h = self.dropout(h)
        return self.classifier(h)
