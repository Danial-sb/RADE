# graph_classification/models_gc.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GCNConv,
    GINConv,
    GINEConv,
    global_mean_pool,
    global_add_pool,
)

try:
    from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
except Exception:
    AtomEncoder = None
    BondEncoder = None

from rade_convs_gc import BatchedGraphCache, RADEGCNConvGC, RADEGINConvGC


def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )


@dataclass
class GraphMPNNConfig:
    gnn: str = "gcn"                 # {"gcn","gin"}
    pooling: str = "mean"            # {"mean","sum"}
    local_layers: int = 3
    hidden_channels: int = 128
    dropout: float = 0.0
    bn: bool = False                 # <--- NEW: optional BatchNorm
    use_ogb_encoders: bool = False
    use_edge_attr: bool = True

    # RADE controls (graph classification)
    unbiased: bool = False
    unbiased_mode: str = "rade"      # {"rade","rade-ic"}
    correct_self_loop: bool = True


class GraphMPNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cfg: GraphMPNNConfig):
        super().__init__()
        self.cfg = cfg
        self.gnn = cfg.gnn.lower().strip()
        self.pooling = cfg.pooling.lower().strip()

        if self.pooling not in {"mean", "sum"}:
            raise ValueError(f"pooling must be 'mean' or 'sum'. Got {self.pooling}")
        if self.gnn not in {"gcn", "gin"}:
            raise ValueError(f"Unsupported gnn={self.gnn}. Use 'gcn' or 'gin'.")
        if cfg.unbiased_mode not in {"rade", "rade-ic"}:
            raise ValueError(f"unbiased_mode must be 'rade' or 'rade-ic'. Got {cfg.unbiased_mode}")

        hidden = int(cfg.hidden_channels)

        # -------------------------
        # Node/edge encoders
        # -------------------------
        self.use_ogb_encoders = bool(cfg.use_ogb_encoders) and (AtomEncoder is not None)
        if self.use_ogb_encoders:
            self.node_encoder = AtomEncoder(emb_dim=hidden)
        else:
            self.node_encoder = nn.Linear(int(in_channels), hidden)

        self.use_edge_attr = bool(cfg.use_edge_attr) and self.use_ogb_encoders and (BondEncoder is not None)
        self.edge_encoder = BondEncoder(emb_dim=hidden) if self.use_edge_attr else None

        # Important: current RADE convs here are node-feature-only.
        if bool(cfg.unbiased) and self.use_edge_attr and (self.gnn == "gin"):
            raise ValueError(
                "RADE-GC (current step) does not yet support edge_attr-aware GIN/GINE. "
                "Set --use_edge_attr False (or disable --unbiased) for now."
            )

        # -------------------------
        # Convs
        # -------------------------
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        use_ic = (cfg.unbiased_mode == "rade-ic")

        for _ in range(int(cfg.local_layers)):
            if self.gnn == "gcn":
                if cfg.unbiased:
                    conv = RADEGCNConvGC(
                        hidden,
                        hidden,
                        bias=True,
                        correct_self_loop=bool(cfg.correct_self_loop),
                        ic_mode=use_ic,
                    )
                else:
                    conv = GCNConv(hidden, hidden)
            else:  # gin
                mlp = _make_mlp(hidden, hidden, hidden)
                if cfg.unbiased:
                    conv = RADEGINConvGC(mlp, ic_mode=use_ic)
                else:
                    if self.use_edge_attr:
                        conv = GINEConv(nn=mlp, edge_dim=hidden)
                    else:
                        conv = GINConv(nn=mlp)

            self.convs.append(conv)
            self.norms.append(nn.BatchNorm1d(hidden) if bool(cfg.bn) else nn.Identity())

        # -------------------------
        # Graph head
        # -------------------------
        self.pred = nn.Linear(hidden, int(out_channels))

    def _pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return global_mean_pool(x, batch)
        return global_add_pool(x, batch)

    def set_graph(self, edge_index_clean: torch.Tensor, batch: torch.Tensor) -> None:
        cache = BatchedGraphCache.from_edge_index(edge_index_clean, batch)
        for conv in self.convs:
            if hasattr(conv, "set_graph"):
                conv.set_graph(cache)

    def forward(
        self,
        x: torch.Tensor,
        edge_index_clean: torch.Tensor,
        batch: torch.Tensor,
        *,
        edge_attr: Optional[torch.Tensor] = None,
        edge_index_keep: Optional[torch.Tensor] = None,
        edge_index_add: Optional[torch.Tensor] = None,
        p: float = 0.0,
        q: float = 0.0,
        return_embeddings: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:

        # -------------------------
        # Encode nodes (and edges if enabled)
        # -------------------------
        if self.use_ogb_encoders:
            x = x.to(torch.long)
            h = self.node_encoder(x).to(torch.float32)
            e = None
            if self.use_edge_attr and edge_attr is not None:
                e = self.edge_encoder(edge_attr.to(torch.long)).to(torch.float32)
        else:
            if not torch.is_floating_point(x):
                x = x.to(torch.float32)
            h = self.node_encoder(x).to(torch.float32)
            e = None  # not used

        # -------------------------
        # Set per-batch cache for RADE convs
        # -------------------------
        if bool(self.cfg.unbiased):
            self.set_graph(edge_index_clean, batch)

        rade_active = (
            bool(self.cfg.unbiased)
            and self.training
            and (edge_index_keep is not None or edge_index_add is not None)
            and (float(p) > 0.0 or float(q) > 0.0)
        )

        for conv, norm in zip(self.convs, self.norms):
            if rade_active and hasattr(conv, "set_graph"):
                h = conv(
                    h,
                    edge_index_clean,
                    edge_index_keep=edge_index_keep,
                    edge_index_add=edge_index_add,
                    p=float(p),
                    q=float(q),
                )
            else:
                # Clean path; RADE-IC convs may add inference correction in eval using p,q.
                if hasattr(conv, "set_graph"):
                    h = conv(h, edge_index_clean, p=float(p), q=float(q))
                else:
                    if (self.gnn == "gin") and self.use_edge_attr and (e is not None):
                        h = conv(h, edge_index_clean, e)
                    else:
                        h = conv(h, edge_index_clean)

            h = norm(h)  # BN or Identity
            h = F.relu(h, inplace=True)
            h = F.dropout(h, p=float(self.cfg.dropout), training=self.training)

        g = self._pool(h, batch)
        out = self.pred(g)

        if return_embeddings:
            return out, h
        return out
