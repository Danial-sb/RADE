# graph_classification/models_gc.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_add_pool, global_mean_pool, GCNConv, GINConv

from rade_convs_gc import RADEGCNConvGC, RADEGINConvGC, BatchedGraphCache

try:
    from .dropmessage_convs_gc import DropMessageGCNConvGC, DropMessageGINConvGC
except Exception:
    from dropmessage_convs_gc import DropMessageGCNConvGC, DropMessageGINConvGC


EdgeIndexObj = Union[torch.Tensor, List[torch.Tensor]]


@dataclass
class GraphMPNNConfig:
    gnn: str = "gin"                 # 'gin' or 'gcn'
    pooling: str = "sum"             # 'sum' or 'mean'
    local_layers: int = 3
    hidden_channels: int = 64
    dropout: float = 0.0
    bn: bool = False

    use_ogb_encoders: bool = False
    use_edge_attr: bool = False

    pre_linear: bool = False

    # RADE
    unbiased: bool = False
    unbiased_mode: str = "rade"      # 'rade' or 'rade-ic'
    correct_self_loop: bool = True

    # linear
    linear: bool = False

    # augmentation selector (NC style)
    aug_tech: str = "rade"           # 'rade' | 'dropmessage' | 'dropnode' | 'none'

    def __post_init__(self) -> None:
        self.gnn = str(self.gnn).lower().strip()
        self.pooling = str(self.pooling).lower().strip()
        self.unbiased_mode = str(self.unbiased_mode).lower().strip()
        self.aug_tech = str(self.aug_tech).lower().strip()

        if self.gnn not in {"gin", "gcn"}:
            raise ValueError(f"gnn must be 'gin' or 'gcn'. Got gnn={self.gnn}")
        if self.pooling not in {"sum", "mean"}:
            raise ValueError(f"pooling must be 'sum' or 'mean'. Got pooling={self.pooling}")
        if self.unbiased_mode not in {"rade", "rade-ic"}:
            raise ValueError(f"unbiased_mode must be 'rade' or 'rade-ic'. Got {self.unbiased_mode}")
        if self.aug_tech not in {"rade", "dropmessage", "dropnode", "none"}:
            raise ValueError(f"aug_tech must be one of {{rade, dropmessage, dropnode, none}}. Got {self.aug_tech}")

        if self.linear:
            self.bn = False
            self.dropout = 0.0


def _make_gin_mlp(in_dim: int, out_dim: int, *, hidden_dim: int, dropout: float, linear: bool) -> nn.Module:
    if linear:
        return nn.Linear(in_dim, out_dim)
    layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    if float(dropout) > 0.0:
        layers.append(nn.Dropout(p=float(dropout)))
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)

class GraphMPNN(nn.Module):
    """
    Graph classification MPNN:
      model(x, edge_index, batch, edge_attr=..., edge_index_keep=..., edge_index_add=..., p=..., q=...,
            dropmessage_rate=..., dropnode_rate=..., return_embeddings=...)
    """

    def __init__(self, *, in_channels: int, out_channels: int, cfg: GraphMPNNConfig):
        super().__init__()
        self.cfg = cfg

        H = int(cfg.hidden_channels)
        L = int(cfg.local_layers)
        if L <= 0:
            raise ValueError(f"local_layers must be >= 1. Got {L}")

        ic_mode = (cfg.unbiased and cfg.unbiased_mode == "rade-ic")

        # optional pre-linear projection
        self.lin_in = nn.Linear(in_channels, H) if bool(cfg.pre_linear) else None
        first_in = H if self.lin_in is not None else int(in_channels)

        self.convs = nn.ModuleList()

        # -------------------------
        # Layer construction logic (NC style)
        # -------------------------
        if cfg.unbiased:
            # RADE convs
            if cfg.gnn == "gcn":
                self.convs.append(
                    RADEGCNConvGC(first_in, H, correct_self_loop=cfg.correct_self_loop, ic_mode=ic_mode)
                )
                for _ in range(L - 1):
                    self.convs.append(
                        RADEGCNConvGC(H, H, correct_self_loop=cfg.correct_self_loop, ic_mode=ic_mode)
                    )
            else:
                mlp0 = _make_gin_mlp(first_in, H, hidden_dim=H, dropout=cfg.dropout, linear=cfg.linear)
                self.convs.append(RADEGINConvGC(mlp=mlp0, eps=0.0, train_eps=False, ic_mode=ic_mode))
                for _ in range(L - 1):
                    mlp = _make_gin_mlp(H, H, hidden_dim=H, dropout=cfg.dropout, linear=cfg.linear)
                    self.convs.append(RADEGINConvGC(mlp=mlp, eps=0.0, train_eps=False, ic_mode=ic_mode))

        else:
            # baselines / clean
            if cfg.aug_tech == "dropmessage":
                if cfg.gnn == "gcn":
                    self.convs.append(DropMessageGCNConvGC(first_in, H, bias=True))
                    for _ in range(L - 1):
                        self.convs.append(DropMessageGCNConvGC(H, H, bias=True))
                else:
                    mlp0 = _make_gin_mlp(first_in, H, hidden_dim=H, dropout=cfg.dropout, linear=cfg.linear)
                    self.convs.append(DropMessageGINConvGC(mlp=mlp0, eps=0.0, train_eps=False))
                    for _ in range(L - 1):
                        mlp = _make_gin_mlp(H, H, hidden_dim=H, dropout=cfg.dropout, linear=cfg.linear)
                        self.convs.append(DropMessageGINConvGC(mlp=mlp, eps=0.0, train_eps=False))
            else:
                # clean model (or DropNode, which is applied on x, not in conv)
                if cfg.gnn == "gcn":
                    self.convs.append(GCNConv(first_in, H, cached=False, normalize=True))
                    for _ in range(L - 1):
                        self.convs.append(GCNConv(H, H, cached=False, normalize=True))
                else:
                    mlp0 = _make_gin_mlp(first_in, H, hidden_dim=H, dropout=cfg.dropout, linear=cfg.linear)
                    self.convs.append(GINConv(mlp0))
                    for _ in range(L - 1):
                        mlp = _make_gin_mlp(H, H, hidden_dim=H, dropout=cfg.dropout, linear=cfg.linear)
                        self.convs.append(GINConv(mlp))

        # BN (disabled automatically if linear)
        self.bns = nn.ModuleList()
        if cfg.bn:
            for _ in range(L):
                self.bns.append(nn.BatchNorm1d(H))

        self.pred = nn.Linear(H, int(out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.lin_in is not None:
            self.lin_in.reset_parameters()
        for c in self.convs:
            if hasattr(c, "reset_parameters"):
                c.reset_parameters()
        for b in self.bns:
            b.reset_parameters()
        self.pred.reset_parameters()

    def _pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.cfg.pooling == "sum":
            return global_add_pool(x, batch)
        if self.cfg.pooling == "mean":
            return global_mean_pool(x, batch)
        raise ValueError(f"Unsupported pooling={self.cfg.pooling}")

    @staticmethod
    def _apply_dropnode(x: torch.Tensor, dropnode_rate: float, training: bool) -> torch.Tensor:
        """
        DropNode (current NC style):
          mask_i ~ Bernoulli(1-r) shared across feature dims, applied at train only.
          rescale by 1/(1-r) so E[x_masked] = x.
        """
        r = float(dropnode_rate)
        if (not training) or r <= 0.0:
            return x
        if r >= 1.0:
            raise ValueError(f"dropnode_rate must be in [0,1). Got {r}")
        keep = 1.0 - r
        mask = torch.empty((x.size(0), 1), device=x.device, dtype=x.dtype).bernoulli_(keep)
        return x * (mask / keep)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        *,
        edge_attr: Optional[torch.Tensor] = None,
        edge_index_keep: Optional[EdgeIndexObj] = None,
        edge_index_add: Optional[EdgeIndexObj] = None,
        p: float = 0.0,
        q: float = 0.0,
        dropmessage_rate: float = 0.0,
        dropnode_rate: float = 0.0,
        return_embeddings: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        _ = edge_attr  # unused (topology-only)

        edge_index = edge_index.to(torch.long)
        batch = batch.to(torch.long)
        if not torch.is_floating_point(x):
            x = x.float()
        if self.lin_in is not None:
            x = self.lin_in(x)
            if (not self.cfg.linear) and (float(self.cfg.dropout) > 0.0):
                x = F.dropout(x, p=float(self.cfg.dropout), training=self.training)

        # DropNode baseline: apply on x before message passing (only when selected)
        if (not self.cfg.unbiased) and (self.cfg.aug_tech == "dropnode"):
            x = self._apply_dropnode(x, dropnode_rate=float(dropnode_rate), training=self.training)

        # Only build/set graph cache if convs support set_graph (RADE convs)
        needs_cache = any(hasattr(c, "set_graph") for c in self.convs)
        if needs_cache:
            num_graphs = int(batch.max().item() + 1) if batch.numel() > 0 else 0
            graph_cache = BatchedGraphCache.from_edge_index(edge_index, batch, num_graphs=num_graphs)
            for conv in self.convs:
                if hasattr(conv, "set_graph"):
                    conv.set_graph(graph_cache)

        L = len(self.convs)

        def _pick_per_layer(obj: Optional[EdgeIndexObj], ell: int) -> Optional[torch.Tensor]:
            if obj is None:
                return None
            if isinstance(obj, (list, tuple)):
                if len(obj) != L:
                    raise ValueError(
                        f"layerwise edge_index list must have length {L} (got {len(obj)}). "
                        f"Ensure main_gc passes a list of length local_layers."
                    )
                t = obj[ell]
                return None if (t is None or t.numel() == 0) else t.to(torch.long)
            t = obj
            return None if t.numel() == 0 else t.to(torch.long)

        # RADE train path requirements (unchanged)
        if self.training and self.cfg.unbiased and ((float(p) > 0.0) or (float(q) > 0.0)):
            if edge_index_keep is None:
                raise RuntimeError(
                    "RADE training path requires edge_index_keep (and optionally edge_index_add). "
                    "Pass edge_index_keep=... and edge_index_add=... from main_gc.py."
                )
            if isinstance(edge_index_keep, (list, tuple)) and (len(edge_index_keep) != L):
                raise ValueError(f"edge_index_keep list must have length {L} (got {len(edge_index_keep)})")

        h = x

        for ell, conv in enumerate(self.convs):
            keep_l = _pick_per_layer(edge_index_keep, ell)
            add_l = _pick_per_layer(edge_index_add, ell)

            # ---- Forward routing per conv type ----
            if hasattr(conv, "set_graph"):
                # RADE convs
                if keep_l is None:
                    h = conv(h, edge_index, p=float(p), q=float(q))
                else:
                    h = conv(
                        h,
                        edge_index,
                        edge_index_keep=keep_l,
                        edge_index_add=add_l,
                        p=float(p),
                        q=float(q),
                    )
            elif getattr(conv, "supports_dropmessage", False):
                # DropMessage baselines
                h = conv(h, edge_index, drop_rate=float(dropmessage_rate))
            else:
                # Vanilla PyG convs
                h = conv(h, edge_index)

            # nonlinearity / BN / dropout
            if not self.cfg.linear:
                if self.cfg.bn:
                    h = self.bns[ell](h)
                h = F.relu(h)
                if float(self.cfg.dropout) > 0.0:
                    h = F.dropout(h, p=float(self.cfg.dropout), training=self.training)

        node_emb = h
        g = self._pool(node_emb, batch)
        logits = self.pred(g)

        if return_embeddings:
            return logits, node_emb
        return logits
