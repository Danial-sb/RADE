import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, GINConv
from typing import Optional
from rade_convs import GraphCache, RADEGCNConv, RADEGINConv


class MPNNs(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        local_layers=3,
        dropout=0.5,
        pre_linear=False,
        res=False,
        ln=False,
        bn=False,
        jk=False,
        gnn="gcn",
        unbiased: bool = False,
        unbiased_mode: str = "rade",
    ):
        super(MPNNs, self).__init__()

        assert gnn in ("gcn", "gin"), "gnn must be 'gcn' or 'gin'"
        assert local_layers >= 1, "local_layers must be >= 1"
        if ln and bn:
            raise ValueError("Choose at most one normalization: ln=True or bn=True (not both).")

        if unbiased_mode not in ("rade", "rade-ic"):
            raise ValueError(f"unbiased_mode must be 'rade' or 'rade-ic'. Got {unbiased_mode}")

        self.dropout = float(dropout)
        self.pre_linear = bool(pre_linear)
        self.res = bool(res)
        self.ln = bool(ln)
        self.bn = bool(bn)
        self.jk = bool(jk)
        self.gnn = gnn
        self.unbiased = bool(unbiased)
        self.unbiased_mode = str(unbiased_mode)

        self.local_convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList() if self.res else None
        self.lns = torch.nn.ModuleList() if self.ln else None
        self.bns = torch.nn.ModuleList() if self.bn else None

        self.lin_in = torch.nn.Linear(in_channels, hidden_channels) if self.pre_linear else None

        # ----- helpers for GIN -----
        def make_gin_mlp(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
            )

        first_in = hidden_channels if self.pre_linear else in_channels
        L = int(local_layers)

        use_ic = (self.unbiased_mode == "rade-ic")

        # -------------------------
        # Build layers
        # -------------------------
        for layer_idx in range(L):
            in_dim = first_in if layer_idx == 0 else hidden_channels
            out_dim = hidden_channels

            if gnn == "gin":
                if self.unbiased:
                    conv = RADEGINConv(make_gin_mlp(in_dim, out_dim), ic_mode=use_ic)
                else:
                    conv = GINConv(make_gin_mlp(in_dim, out_dim))
            else:
                if self.unbiased:
                    conv = RADEGCNConv(in_dim, out_dim, bias=True, correct_self_loop=True, ic_mode=use_ic)
                else:
                    conv = GCNConv(in_dim, out_dim, cached=False, normalize=True)

            self.local_convs.append(conv)

            if self.res:
                self.lins.append(nn.Linear(in_dim, out_dim))
            if self.ln:
                self.lns.append(nn.LayerNorm(out_dim))
            if self.bn:
                self.bns.append(nn.BatchNorm1d(out_dim))

        self.pred_local = torch.nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for conv in self.local_convs:
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

    def set_graph(self, edge_index_clean: torch.Tensor, num_nodes: int) -> None:
        cache = GraphCache.from_edge_index(edge_index_clean, num_nodes)
        for conv in self.local_convs:
            if hasattr(conv, "set_graph"):
                conv.set_graph(cache)

    def forward(
        self,
        x: torch.Tensor,
        edge_index_clean: torch.Tensor,
        *,
        edge_index_keep: Optional[torch.Tensor] = None,
        edge_index_add: Optional[torch.Tensor] = None,
        p: float = 0.0,
        q: float = 0.0,
        return_embeddings: bool = False,
    ):
        if self.lin_in is not None:
            x = self.lin_in(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        def _is_list(t):
            return isinstance(t, (list, tuple))

        L = len(self.local_convs)

        # RADE active only if training + keep/add provided + p/q active
        rade_active = (
            self.unbiased
            and self.training
            and (edge_index_keep is not None or edge_index_add is not None)
            and (p > 0.0 or q > 0.0)
        )

        def _pick_layer_edges(obj, i: int):
            # FIX: if obj is a tensor (shared), return it directly
            if obj is None:
                return None
            if _is_list(obj):
                if len(obj) != L:
                    raise ValueError(f"Expected list length={L}, got {len(obj)}")
                return obj[i]
            return obj

        x_final = None

        for i, conv in enumerate(self.local_convs):
            x_in = x  # for residual

            apply_rade_here = rade_active and hasattr(conv, "set_graph")  # RADE conv module

            if apply_rade_here:
                layer_keep = _pick_layer_edges(edge_index_keep, i)
                layer_add = _pick_layer_edges(edge_index_add, i)
                x = conv(
                    x,
                    edge_index_clean,
                    edge_index_keep=layer_keep,
                    edge_index_add=layer_add,
                    p=p,
                    q=q,
                )
            else:
                # Clean path:
                # - For RADE-IC, the conv may add inference correction in eval mode using p,q.
                if hasattr(conv, "set_graph"):
                    x = conv(x, edge_index_clean, p=p, q=q)
                else:
                    x = conv(x, edge_index_clean)

            if self.res:
                x = x + self.lins[i](x_in)
            if self.ln:
                x = self.lns[i](x)
            elif self.bn:
                x = self.bns[i](x)

            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.jk:
                x_final = x if x_final is None else (x_final + x)
            else:
                x_final = x

        logits = self.pred_local(x_final)
        if return_embeddings:
            return logits, x_final
        return logits

