# nc_minibatch/mb_models.py
from __future__ import annotations

from typing import Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv

from .mb_rade_convs import BatchGraphCache, RADEGCNConvMB, RADEGINConvMB
from ..full_batch.dropmessage_convs import DropMessageGCNConv, DropMessageGINConv

EdgeIndexObj = Union[torch.Tensor, List[torch.Tensor]]


class MPNNsMB(nn.Module):
    """
    Mini-batch MPNNs for node classification.

    Differences vs full_batch:
      - RADE caches are batch-local and are rebuilt per forward(batch).
      - Complement C(i) is the complement within the batch node set.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        local_layers: int = 3,
        dropout: float = 0.5,
        pre_linear: bool = False,
        res: bool = False,
        ln: bool = False,
        bn: bool = False,
        jk: bool = False,
        gnn: str = "gcn",
        unbiased: bool = False,
        unbiased_mode: str = "rade",
        linear: bool = False,
        aug_tech: str = "rade",
        mb_rade_mode: str = "local",
    ):
        super().__init__()

        assert gnn in ("gcn", "gin"), "gnn must be 'gcn' or 'gin'"
        assert local_layers >= 1, "local_layers must be >= 1"
        if unbiased_mode not in ("rade", "rade-ic"):
            raise ValueError(f"unbiased_mode must be 'rade' or 'rade-ic'. Got {unbiased_mode}")

        self.linear = bool(linear)
        if self.linear:
            dropout = 0.0
            ln = False
            bn = False

        if ln and bn:
            raise ValueError("Choose at most one normalization: ln=True or bn=True (not both).")

        self.dropout = float(dropout)
        self.pre_linear = bool(pre_linear)
        self.res = bool(res)
        self.ln = bool(ln)
        self.bn = bool(bn)
        self.jk = bool(jk)
        self.gnn = str(gnn).lower().strip()
        self.unbiased = bool(unbiased)
        self.unbiased_mode = str(unbiased_mode).lower().strip()

        self.mb_rade_mode = str(mb_rade_mode).lower().strip()
        if self.mb_rade_mode not in {"local", "disabled"}:
            raise ValueError(f"mb_rade_mode must be 'local' or 'disabled'. Got {self.mb_rade_mode}")

        self.aug_tech = str(aug_tech).lower().strip()
        if self.aug_tech not in {"rade", "dropmessage", "dropnode", "none"}:
            raise ValueError(f"aug_tech must be one of {{rade, dropmessage, dropnode, none}}. Got {self.aug_tech}")

        self.lin_in = nn.Linear(in_channels, hidden_channels) if self.pre_linear else None
        self.local_convs = nn.ModuleList()
        self.lins = nn.ModuleList() if self.res else None
        self.lns = nn.ModuleList() if self.ln else None
        self.bns = nn.ModuleList() if self.bn else None


        def make_gin_mlp(in_dim: int, out_dim: int) -> nn.Module:
            if self.linear:
                return nn.Linear(in_dim, out_dim)
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
            )

        first_in = hidden_channels if self.pre_linear else in_channels
        L = int(local_layers)
        use_ic = (self.unbiased and self.unbiased_mode == "rade-ic")

        for layer_idx in range(L):
            in_dim = first_in if layer_idx == 0 else hidden_channels
            out_dim = hidden_channels

            if self.gnn == "gin":
                mlp = make_gin_mlp(in_dim, out_dim)
                if self.aug_tech == "rade":
                    conv = RADEGINConvMB(
                        mlp,
                        ic_mode=use_ic,
                        apply_corrections=self.unbiased,
                    )
                elif self.aug_tech == "dropmessage":
                    conv = DropMessageGINConv(mlp)
                else:
                    conv = GINConv(mlp)
            else:
                if self.aug_tech == "rade":
                    conv = RADEGCNConvMB(
                        in_dim,
                        out_dim,
                        bias=True,
                        correct_self_loop=True,
                        ic_mode=use_ic,
                        apply_corrections=self.unbiased,
                    )
                elif self.aug_tech == "dropmessage":
                    conv = DropMessageGCNConv(in_dim, out_dim, bias=True)
                else:
                    conv = GCNConv(in_dim, out_dim, cached=False, normalize=True)

            self.local_convs.append(conv)

            if self.res:
                self.lins.append(nn.Linear(in_dim, out_dim))
            if self.ln:
                self.lns.append(nn.LayerNorm(out_dim))
            if self.bn:
                self.bns.append(nn.BatchNorm1d(out_dim))

        self.pred_local = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self) -> None:
        for conv in self.local_convs:
            if hasattr(conv, "reset_parameters"):
                conv.reset_parameters()
        if self.lins is not None:
            for lin in self.lins:
                lin.reset_parameters()
        if self.lns is not None:
            for ln in self.lns:
                ln.reset_parameters()
        if self.bns is not None:
            for bn in self.bns:
                bn.reset_parameters()
        if self.lin_in is not None:
            self.lin_in.reset_parameters()
        self.pred_local.reset_parameters()

    @staticmethod
    def _apply_dropnode(x: torch.Tensor, dropnode_rate: float, training: bool) -> torch.Tensor:
        r = float(dropnode_rate)
        if (not training) or r <= 0.0:
            return x
        if r >= 1.0:
            raise ValueError(f"dropnode_rate must be in [0,1). Got {r}")
        keep = 1.0 - r
        mask = torch.empty((x.size(0), 1), device=x.device, dtype=x.dtype).bernoulli_(keep)
        return x * (mask / keep)

    def _set_batch_graph(self, edge_index_clean: torch.Tensor, num_nodes: int) -> None:
        """
        Mini-batch critical: graph cache must be rebuilt for each batch because degrees and complements are batch-local.
        """
        cache = BatchGraphCache.from_edge_index(edge_index_clean, int(num_nodes))
        for conv in self.local_convs:
            if hasattr(conv, "set_graph"):
                conv.set_graph(cache)

    def forward(
        self,
        x: torch.Tensor,
        edge_index_clean: torch.Tensor,
        *,
        edge_index_keep: Optional[EdgeIndexObj] = None,
        edge_index_add: Optional[EdgeIndexObj] = None,
        p: float = 0.0,
        q: float = 0.0,
        dropmessage_rate: float = 0.0,
        dropnode_rate: float = 0.0,
        return_embeddings: bool = False,
    ):
        # If we have RADE convs in the stack, set batch cache every forward.
        if self.aug_tech == "rade":
            self._set_batch_graph(edge_index_clean=edge_index_clean, num_nodes=x.size(0))

        # Pre-linear (optional)
        if self.lin_in is not None:
            x = self.lin_in(x)
            if (not self.linear) and (self.dropout > 0.0):
                x = F.dropout(x, p=self.dropout, training=self.training)

        # DropNode (baseline): only when selected
        if self.aug_tech == "dropnode":
            x = self._apply_dropnode(x, dropnode_rate=float(dropnode_rate), training=self.training)

        def _is_list(t):
            return isinstance(t, (list, tuple))

        L = len(self.local_convs)

        rade_active = (
            (self.aug_tech == "rade")
            and (self.mb_rade_mode == "local")
            and self.training
            and (edge_index_keep is not None or edge_index_add is not None)
            and (float(p) > 0.0 or float(q) > 0.0)
        )

        def _pick_layer_edges(obj: Optional[EdgeIndexObj], i: int) -> Optional[torch.Tensor]:
            if obj is None:
                return None
            if _is_list(obj):
                if len(obj) != L:
                    raise ValueError(f"Expected list length={L}, got {len(obj)}")
                t = obj[i]
                return None if (t is None or t.numel() == 0) else t
            return None if obj.numel() == 0 else obj

        x_final = None

        for i, conv in enumerate(self.local_convs):
            x_in = x

            apply_rade_here = rade_active and hasattr(conv, "set_graph")

            if apply_rade_here:
                layer_keep = _pick_layer_edges(edge_index_keep, i)
                layer_add = _pick_layer_edges(edge_index_add, i)
                x = conv(
                    x,
                    edge_index_clean,
                    edge_index_keep=layer_keep,
                    edge_index_add=layer_add,
                    p=float(p),
                    q=float(q),
                )
            else:
                # RADE convs may use p,q in eval for RADE-IC inference correction
                if hasattr(conv, "set_graph"):
                    x = conv(x, edge_index_clean, p=float(p), q=float(q))
                elif getattr(conv, "supports_dropmessage", False):
                    x = conv(x, edge_index_clean, drop_rate=float(dropmessage_rate))
                else:
                    x = conv(x, edge_index_clean)

            if self.res:
                x = x + self.lins[i](x_in)

            if not self.linear:
                if self.ln:
                    x = self.lns[i](x)
                elif self.bn:
                    x = self.bns[i](x)

                x = F.relu(x)
                if self.dropout > 0.0:
                    x = F.dropout(x, p=self.dropout, training=self.training)

            if self.jk:
                x_final = x if x_final is None else (x_final + x)
            else:
                x_final = x

        logits = self.pred_local(x_final)
        if return_embeddings:
            return logits, x_final
        return logits


# Convenience alias if you want to import the same name in parse code:
MPNNs = MPNNsMB
