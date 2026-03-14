from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn

from models.anchor_gnn import GraphConvLayer
from models.time_embedding import SinusoidalTimeEmbedding


def mean_pool_by_batch(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Mean-pool node embeddings into graph embeddings using a batch vector."""

    if x.ndim != 2:
        raise ValueError(f"Expected x to be rank-2, got shape {tuple(x.shape)}")
    if batch.ndim != 1 or batch.shape[0] != x.shape[0]:
        raise ValueError("batch must be rank-1 with one entry per node")

    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    if num_graphs == 0:
        return x.new_zeros((0, x.shape[1]))

    pooled = x.new_zeros((num_graphs, x.shape[1]))
    counts = x.new_zeros((num_graphs, 1))
    pooled.index_add_(0, batch, x)
    counts.index_add_(0, batch, torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype))
    return pooled / counts.clamp(min=1.0)


def apply_condition_dropout(
    left_ctx: torch.Tensor,
    right_ctx: torch.Tensor,
    *,
    drop_prob: float,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop fragment conditioning per graph, keeping time conditioning untouched."""

    if left_ctx.shape != right_ctx.shape:
        raise ValueError("left_ctx and right_ctx must share shape for condition dropout")
    if drop_prob <= 0.0 or not training:
        return left_ctx, right_ctx

    if drop_prob >= 1.0:
        keep_mask = left_ctx.new_zeros((left_ctx.shape[0], 1))
    else:
        keep_prob = 1.0 - drop_prob
        keep_mask = (torch.rand((left_ctx.shape[0], 1), device=left_ctx.device) < keep_prob).to(left_ctx.dtype)
    return left_ctx * keep_mask, right_ctx * keep_mask


def sinusoidal_position_encoding(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build a standard sinusoidal position embedding table [L, D]."""

    if length <= 0:
        raise ValueError("length must be > 0")
    positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    half = dim // 2
    scale = math.log(10000.0) / max(half - 1, 1)
    inv_freq = torch.exp(torch.arange(half, device=device, dtype=dtype) * (-scale))
    angles = positions * inv_freq.unsqueeze(0)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class FragmentGraphEncoder(nn.Module):
    """Encode a fragment graph into one embedding per graph."""

    def __init__(
        self,
        in_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList([GraphConvLayer(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index)
            h = norm(h)
            h = torch.relu(h)
            h = self.dropout(h)
        pooled = mean_pool_by_batch(h, batch=batch)
        return self.out_proj(pooled)


class FragmentConditionedNodeDenoiser(nn.Module):
    """Predict linker-node noise conditioned on left/right fragment graphs."""

    def __init__(
        self,
        in_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 4,
        time_dim: int = 128,
        dropout: float = 0.1,
        condition_dropout: float = 0.0,
        node_type_classes: int = 3,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if not (0.0 <= condition_dropout <= 1.0):
            raise ValueError("condition_dropout must be in [0, 1]")

        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fragment_encoder = FragmentGraphEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=max(1, num_layers - 1),
            dropout=dropout,
        )
        self.condition_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_proj = nn.Linear(in_dim + node_type_classes + 1, hidden_dim)
        self.convs = nn.ModuleList([GraphConvLayer(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, in_dim)
        self.node_type_classes = node_type_classes
        self.condition_dropout = float(condition_dropout)

    def _time_per_graph(self, t: torch.Tensor, num_graphs: int) -> torch.Tensor:
        time_h = self.time_proj(self.time_emb(t))
        if time_h.shape[0] == num_graphs:
            return time_h
        if time_h.shape[0] == 1:
            return time_h.expand(num_graphs, -1)
        raise ValueError(f"Expected one timestep or one timestep per graph, got {time_h.shape[0]} for {num_graphs} graphs")

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        linker_graph: Dict[str, torch.Tensor],
        left_graph: Dict[str, torch.Tensor],
        right_graph: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if x.ndim != 3 or x.shape[0] != 1:
            raise ValueError(f"Expected x shape [1, N, F], got {tuple(x.shape)}")

        linker_x = x.squeeze(0)
        linker_batch = linker_graph["batch"].long()
        num_graphs = int(linker_graph["graph_ptr"].shape[0] - 1)
        if num_graphs <= 0:
            raise ValueError("linker_graph must contain at least one graph")

        left_ctx = self.fragment_encoder(
            left_graph["x"],
            left_graph["edge_index"],
            left_graph["batch"].long(),
        )
        right_ctx = self.fragment_encoder(
            right_graph["x"],
            right_graph["edge_index"],
            right_graph["batch"].long(),
        )
        if left_ctx.shape[0] != num_graphs or right_ctx.shape[0] != num_graphs:
            raise ValueError("Fragment graph counts must match linker graph count")
        left_ctx, right_ctx = apply_condition_dropout(
            left_ctx,
            right_ctx,
            drop_prob=self.condition_dropout,
            training=self.training,
        )

        time_ctx = self._time_per_graph(t, num_graphs=num_graphs)
        graph_ctx = self.condition_proj(torch.cat([left_ctx, right_ctx, time_ctx], dim=1))
        node_ctx = graph_ctx[linker_batch]

        node_type = linker_graph["node_type"].long()
        node_type_one_hot = torch.nn.functional.one_hot(
            node_type.clamp(min=0, max=self.node_type_classes - 1),
            num_classes=self.node_type_classes,
        ).to(dtype=linker_x.dtype)
        is_fixed = (node_type > 0).to(dtype=linker_x.dtype).unsqueeze(-1)

        h = self.input_proj(torch.cat([linker_x, node_type_one_hot, is_fixed], dim=1))
        h = h + node_ctx
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, linker_graph["edge_index"].long())
            h = h + node_ctx
            h = norm(h)
            h = torch.relu(h)
            h = self.dropout(h)
        out = self.out_proj(h)
        return out.unsqueeze(0)


class FragmentConditionedEdgeDenoiser(nn.Module):
    """Predict linker-edge noise conditioned on left/right fragment graphs."""

    def __init__(
        self,
        node_in_dim: int = 4,
        edge_in_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 4,
        time_dim: int = 128,
        dropout: float = 0.1,
        condition_dropout: float = 0.0,
        node_type_classes: int = 3,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if not (0.0 <= condition_dropout <= 1.0):
            raise ValueError("condition_dropout must be in [0, 1]")

        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fragment_encoder = FragmentGraphEncoder(
            in_dim=node_in_dim,
            hidden_dim=hidden_dim,
            num_layers=max(1, num_layers - 1),
            dropout=dropout,
        )
        self.condition_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.node_input_proj = nn.Linear(node_in_dim + node_type_classes + 1, hidden_dim)
        self.node_convs = nn.ModuleList([GraphConvLayer(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.node_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        self.edge_input_proj = nn.Linear(edge_in_dim + hidden_dim * 3, hidden_dim)
        self.edge_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(max(1, num_layers - 1))
            ]
        )
        self.edge_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(max(1, num_layers - 1))])
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, edge_in_dim)
        self.node_type_classes = node_type_classes
        self.condition_dropout = float(condition_dropout)

    def _time_per_graph(self, t: torch.Tensor, num_graphs: int) -> torch.Tensor:
        time_h = self.time_proj(self.time_emb(t))
        if time_h.shape[0] == num_graphs:
            return time_h
        if time_h.shape[0] == 1:
            return time_h.expand(num_graphs, -1)
        raise ValueError(f"Expected one timestep or one timestep per graph, got {time_h.shape[0]} for {num_graphs} graphs")

    def _encode_graph_context(
        self,
        t: torch.Tensor,
        *,
        left_graph: Dict[str, torch.Tensor],
        right_graph: Dict[str, torch.Tensor],
        num_graphs: int,
    ) -> torch.Tensor:
        left_ctx = self.fragment_encoder(
            left_graph["x"],
            left_graph["edge_index"],
            left_graph["batch"].long(),
        )
        right_ctx = self.fragment_encoder(
            right_graph["x"],
            right_graph["edge_index"],
            right_graph["batch"].long(),
        )
        if left_ctx.shape[0] != num_graphs or right_ctx.shape[0] != num_graphs:
            raise ValueError("Fragment graph counts must match linker graph count")
        left_ctx, right_ctx = apply_condition_dropout(
            left_ctx,
            right_ctx,
            drop_prob=self.condition_dropout,
            training=self.training,
        )
        time_ctx = self._time_per_graph(t, num_graphs=num_graphs)
        return self.condition_proj(torch.cat([left_ctx, right_ctx, time_ctx], dim=1))

    def _encode_linker_nodes(
        self,
        linker_graph: Dict[str, torch.Tensor],
        graph_ctx: torch.Tensor,
    ) -> torch.Tensor:
        linker_x = linker_graph["x"]
        linker_batch = linker_graph["batch"].long()
        node_type = linker_graph["node_type"].long()
        node_type_one_hot = torch.nn.functional.one_hot(
            node_type.clamp(min=0, max=self.node_type_classes - 1),
            num_classes=self.node_type_classes,
        ).to(dtype=linker_x.dtype)
        is_fixed = (node_type > 0).to(dtype=linker_x.dtype).unsqueeze(-1)
        node_ctx = graph_ctx[linker_batch]

        h = self.node_input_proj(torch.cat([linker_x, node_type_one_hot, is_fixed], dim=1))
        h = h + node_ctx
        for conv, norm in zip(self.node_convs, self.node_norms):
            h = conv(h, linker_graph["edge_index"].long())
            h = h + node_ctx
            h = norm(h)
            h = torch.relu(h)
            h = self.dropout(h)
        return h

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        linker_graph: Dict[str, torch.Tensor],
        left_graph: Dict[str, torch.Tensor],
        right_graph: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if x.ndim != 4 or x.shape[0] != 1:
            raise ValueError(f"Expected x shape [1, N, N, F], got {tuple(x.shape)}")

        edge_x = x.squeeze(0)
        num_graphs = int(linker_graph["graph_ptr"].shape[0] - 1)
        if num_graphs <= 0:
            raise ValueError("linker_graph must contain at least one graph")

        graph_ctx = self._encode_graph_context(
            t,
            left_graph=left_graph,
            right_graph=right_graph,
            num_graphs=num_graphs,
        )
        node_h = self._encode_linker_nodes(linker_graph, graph_ctx=graph_ctx)
        linker_batch = linker_graph["batch"].long()

        row_h = node_h.unsqueeze(1).expand(-1, node_h.shape[0], -1)
        col_h = node_h.unsqueeze(0).expand(node_h.shape[0], -1, -1)
        pair_ctx = graph_ctx[linker_batch].unsqueeze(1).expand(-1, node_h.shape[0], -1)

        h = self.edge_input_proj(torch.cat([edge_x, row_h, col_h, pair_ctx], dim=-1))
        h = h + pair_ctx
        for block, norm in zip(self.edge_blocks, self.edge_norms):
            h = block(h) + pair_ctx
            h = norm(h)
            h = torch.relu(h)
            h = self.dropout(h)
        out = self.out_proj(h)
        out = 0.5 * (out + out.transpose(0, 1))
        return out.unsqueeze(0)


class FragmentConditionedTokenDenoiser(nn.Module):
    """Predict linker token-embedding noise conditioned on left/right fragment graphs."""

    def __init__(
        self,
        embed_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        time_dim: int = 128,
        dropout: float = 0.1,
        condition_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not (0.0 <= condition_dropout <= 1.0):
            raise ValueError("condition_dropout must be in [0, 1]")

        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fragment_encoder = FragmentGraphEncoder(
            in_dim=4,
            hidden_dim=hidden_dim,
            num_layers=max(1, num_layers - 1),
            dropout=dropout,
        )
        self.condition_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, embed_dim)
        self.condition_dropout = float(condition_dropout)

    def _time_per_graph(self, t: torch.Tensor, num_graphs: int) -> torch.Tensor:
        time_h = self.time_proj(self.time_emb(t))
        if time_h.shape[0] == num_graphs:
            return time_h
        if time_h.shape[0] == 1:
            return time_h.expand(num_graphs, -1)
        raise ValueError(f"Expected one timestep or one timestep per graph, got {time_h.shape[0]} for {num_graphs} graphs")

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        left_graph: Dict[str, torch.Tensor],
        right_graph: Dict[str, torch.Tensor],
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x shape [B, L, D], got {tuple(x.shape)}")

        batch_size, seq_len, _ = x.shape
        if batch_size <= 0:
            raise ValueError("token batch must contain at least one sample")

        left_ctx = self.fragment_encoder(
            left_graph["x"],
            left_graph["edge_index"],
            left_graph["batch"].long(),
        )
        right_ctx = self.fragment_encoder(
            right_graph["x"],
            right_graph["edge_index"],
            right_graph["batch"].long(),
        )
        if left_ctx.shape[0] != batch_size or right_ctx.shape[0] != batch_size:
            raise ValueError("Fragment graph counts must match token batch size")
        left_ctx, right_ctx = apply_condition_dropout(
            left_ctx,
            right_ctx,
            drop_prob=self.condition_dropout,
            training=self.training,
        )

        time_ctx = self._time_per_graph(t, num_graphs=batch_size)
        graph_ctx = self.condition_proj(torch.cat([left_ctx, right_ctx, time_ctx], dim=1))
        pos = sinusoidal_position_encoding(
            length=seq_len,
            dim=graph_ctx.shape[1],
            device=x.device,
            dtype=x.dtype,
        )

        if token_mask is None:
            token_mask = torch.ones((batch_size, seq_len), device=x.device, dtype=torch.bool)
        else:
            token_mask = token_mask.to(device=x.device, dtype=torch.bool)

        h = self.input_proj(x) + graph_ctx.unsqueeze(1) + pos.unsqueeze(0)
        h = torch.where(token_mask.unsqueeze(-1), h, torch.zeros_like(h))
        h = self.encoder(h, src_key_padding_mask=~token_mask)
        h = self.dropout(h)
        out = self.out_proj(h)
        return torch.where(token_mask.unsqueeze(-1), out, torch.zeros_like(out))
