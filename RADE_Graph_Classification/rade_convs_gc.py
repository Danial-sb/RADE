# graph_classification/rade_convs_gc.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import reset


def _scatter_add_rows(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    # values: [E, F], index: [E] -> out: [dim_size, F]
    out = torch.zeros((dim_size, values.size(1)), device=values.device, dtype=values.dtype)
    out.index_add_(0, index, values)
    return out


def _scatter_add_scalar(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    # values: [E], index: [E] -> out: [dim_size]
    out = torch.zeros((dim_size,), device=values.device, dtype=values.dtype)
    out.index_add_(0, index, values)
    return out


def _scatter_add_graph_rows(values: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    # values: [N,F], batch: [N] -> out: [G,F]
    out = torch.zeros((num_graphs, values.size(1)), device=values.device, dtype=values.dtype)
    out.index_add_(0, batch, values)
    return out


def _scatter_add_graph_scalar(values: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    # values: [N], batch: [N] -> out: [G]
    out = torch.zeros((num_graphs,), device=values.device, dtype=values.dtype)
    out.index_add_(0, batch, values)
    return out


@dataclass
class BatchedGraphCache:
    """
    Cache derived from the CLEAN batched graph (disjoint union).
    All complement operations are WITHIN each graph in the batch.

    - edge_index_clean_dir: directed (as provided), no self-loops
    - deg_clean: per-node clean degree (excluding self-loops)
    - comp_size: |C(i)| = (n_g - 1 - d_i) within the node's graph
    """
    edge_index_clean_dir: torch.Tensor  # [2, E_dir]
    batch: torch.Tensor                 # [N]
    num_nodes: int
    num_graphs: int
    deg_clean: torch.Tensor             # [N] float32
    comp_size: torch.Tensor             # [N] float32
    n_per_node: torch.Tensor            # [N] float32 (graph size for each node)

    @staticmethod
    def from_edge_index(edge_index_clean: torch.Tensor, batch: torch.Tensor, num_graphs: Optional[int] = None) -> "BatchedGraphCache":
        batch = batch.to(torch.long)
        N = int(batch.numel())
        G = int(batch.max().item() + 1) if num_graphs is None else int(num_graphs)

        row, col = edge_index_clean[0], edge_index_clean[1]
        mask = row != col
        row, col = row[mask], col[mask]
        edge_index_clean_dir = torch.stack([row, col], dim=0)

        # deg_clean: float32 for delta approximations
        ones_e = torch.ones(col.numel(), dtype=torch.float32, device=col.device)
        deg = torch.zeros((N,), dtype=torch.float32, device=col.device)
        deg.index_add_(0, col, ones_e)

        # graph sizes
        ones_n = torch.ones((N,), dtype=torch.float32, device=batch.device)
        n_g = torch.zeros((G,), dtype=torch.float32, device=batch.device)
        n_g.index_add_(0, batch, ones_n)
        n_per_node = n_g[batch]

        comp = (n_per_node - 1.0 - deg).clamp(min=0.0)

        return BatchedGraphCache(
            edge_index_clean_dir=edge_index_clean_dir,
            batch=batch,
            num_nodes=N,
            num_graphs=G,
            deg_clean=deg,
            comp_size=comp,
            n_per_node=n_per_node,
        )

    def complement_sum_unweighted(self, x: torch.Tensor) -> torch.Tensor:
        """
        S_i = sum_{j in C(i)} x_j, computed within each graph.
        """
        N, G = self.num_nodes, self.num_graphs
        row, col = self.edge_index_clean_dir.to(x.device)
        b = self.batch.to(x.device)

        neigh_sum = _scatter_add_rows(x[row], col, N)              # sum over N(i)
        total_g = _scatter_add_graph_rows(x, b, G)                 # sum over nodes in each graph
        total_i = total_g[b]                                       # broadcast to nodes

        return total_i - x - neigh_sum

    def complement_mean_unweighted(self, x: torch.Tensor) -> torch.Tensor:
        """
        mu_i = mean_{j in C(i)} x_j within each graph.
        """
        comp_sum = self.complement_sum_unweighted(x)
        denom = self.comp_size.to(x.device).clamp(min=1.0).unsqueeze(1)
        return comp_sum / denom

    def complement_sum_weighted(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        S_i = sum_{j in C(i)} w_j x_j, within each graph.
        We assume w depends only on node j.
        """
        N, G = self.num_nodes, self.num_graphs
        row, col = self.edge_index_clean_dir.to(x.device)
        b = self.batch.to(x.device)

        w = w.to(x.device)
        wx = w.unsqueeze(1) * x

        neigh_wx = _scatter_add_rows(wx[row], col, N)
        total_g_wx = _scatter_add_graph_rows(wx, b, G)
        total_i_wx = total_g_wx[b]

        return total_i_wx - wx - neigh_wx

    def complement_mean_weighted(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        mu_i = (sum_{j in C(i)} w_j x_j) / (sum_{j in C(i)} w_j), within each graph.
        """
        N, G = self.num_nodes, self.num_graphs
        row, col = self.edge_index_clean_dir.to(x.device)
        b = self.batch.to(x.device)

        w = w.to(x.device)
        wx = w.unsqueeze(1) * x

        total_g_wx = _scatter_add_graph_rows(wx, b, G)
        total_g_w = _scatter_add_graph_scalar(w, b, G)

        neigh_wx = _scatter_add_rows(wx[row], col, N)
        neigh_w = _scatter_add_scalar(w[row], col, N)

        num = total_g_wx[b] - wx - neigh_wx
        den = (total_g_w[b] - w - neigh_w).clamp(min=1e-12).unsqueeze(1)
        return num / den


def _delta_E_inv_caseA(deg_clean: torch.Tensor, comp_size: torch.Tensor, p: float, q: float, eps: float = 1e-12) -> torch.Tensor:
    d_minus = (deg_clean - 1.0).clamp(min=0.0)
    mu = (1.0 - p) * d_minus + q * comp_size + 2.0
    var = d_minus * (1.0 - p) * p + comp_size * q * (1.0 - q)
    mu = mu.clamp(min=eps)
    return mu.pow(-1.0) + var * mu.pow(-3.0)


def _delta_E_inv_caseB(deg_clean: torch.Tensor, comp_size: torch.Tensor, p: float, q: float, eps: float = 1e-12) -> torch.Tensor:
    c_minus = (comp_size - 1.0).clamp(min=0.0)
    mu = (1.0 - p) * deg_clean + q * c_minus + 2.0
    var = deg_clean * (1.0 - p) * p + c_minus * q * (1.0 - q)
    mu = mu.clamp(min=eps)
    return mu.pow(-1.0) + var * mu.pow(-3.0)


def _delta_E_inv_sqrt_caseA(deg_clean: torch.Tensor, comp_size: torch.Tensor, p: float, q: float) -> torch.Tensor:
    d_minus = (deg_clean - 1.0).clamp(min=0.0)
    A = (1.0 - p) * d_minus + q * comp_size + 2.0
    B = d_minus * (1.0 - p) * p + comp_size * q * (1.0 - q)
    return A.pow(-0.5) + (3.0 / 8.0) * B * A.pow(-2.5)


def _delta_E_inv_sqrt_caseB(deg_clean: torch.Tensor, comp_size: torch.Tensor, p: float, q: float) -> torch.Tensor:
    c_minus = (comp_size - 1.0).clamp(min=0.0)
    A = (1.0 - p) * deg_clean + q * c_minus + 2.0
    B = deg_clean * (1.0 - p) * p + c_minus * q * (1.0 - q)
    return A.pow(-0.5) + (3.0 / 8.0) * B * A.pow(-2.5)


def _delta_E_inv_self(deg_clean: torch.Tensor, comp_size: torch.Tensor, p: float, q: float) -> torch.Tensor:
    mu = 1.0 + (1.0 - p) * deg_clean + q * comp_size
    var = deg_clean * (1.0 - p) * p + comp_size * q * (1.0 - q)
    return mu.pow(-1.0) + var * mu.pow(-3.0)


class RADEGCNConvGC(nn.Module):
    """
    Batched-GC version of RADEGCNConv:
      - complements and sizes are computed within each graph in the batch.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        correct_self_loop: bool = True,
        ic_mode: bool = False,
    ):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.correct_self_loop = bool(correct_self_loop)
        self.ic_mode = bool(ic_mode)
        self.graph_cache: Optional[BatchedGraphCache] = None

    def set_graph(self, graph_cache: BatchedGraphCache) -> None:
        self.graph_cache = graph_cache

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index_clean: torch.Tensor,
        *,
        edge_index_keep: Optional[torch.Tensor] = None,
        edge_index_add: Optional[torch.Tensor] = None,
        p: float = 0.0,
        q: float = 0.0,
    ) -> torch.Tensor:
        assert self.graph_cache is not None, "Call model.set_graph(...) per batch before forward."
        x = self.lin(x)
        N = x.size(0)

        # -------------------------
        # CLEAN path
        # -------------------------
        if edge_index_keep is None:
            row, col = edge_index_clean[0], edge_index_clean[1]
            ar = torch.arange(N, device=x.device)
            row = torch.cat([row, ar])
            col = torch.cat([col, ar])

            w = torch.ones(row.numel(), device=x.device, dtype=x.dtype)
            deg = _scatter_add_scalar(w, col, N)
            inv_sqrt = deg.clamp(min=1.0).pow(-0.5)

            gamma = inv_sqrt[row] * inv_sqrt[col]
            out = _scatter_add_rows(gamma.unsqueeze(1) * x[row], col, N)

            # RADE-IC inference correction (eval only)
            if self.ic_mode and (not self.training) and (q > 0.0):
                deg_clean = self.graph_cache.deg_clean.to(x.device)
                comp_size = self.graph_cache.comp_size.to(x.device)

                tB = _delta_E_inv_sqrt_caseB(deg_clean, comp_size, p=p, q=q).to(x.device, dtype=x.dtype)  # [N]
                comp_wx = self.graph_cache.complement_sum_weighted(x, w=tB)
                out = out + (q * tB).unsqueeze(1) * comp_wx

            if self.bias is not None:
                out = out + self.bias
            return out

        # -------------------------
        # RADE / RADE-IC TRAIN path
        # -------------------------
        row_k, col_k = edge_index_keep[0], edge_index_keep[1]
        if edge_index_add is not None and edge_index_add.numel() > 0:
            row_a, col_a = edge_index_add[0], edge_index_add[1]
        else:
            row_a = row_k.new_empty((0,))
            col_a = col_k.new_empty((0,))

        ar = torch.arange(N, device=x.device)
        row_all = torch.cat([row_k, row_a, ar])
        col_all = torch.cat([col_k, col_a, ar])

        w_all = torch.ones(row_all.numel(), device=x.device, dtype=x.dtype)
        deg_aug = _scatter_add_scalar(w_all, col_all, N).clamp(min=1.0)
        inv_sqrt_aug = deg_aug.pow(-0.5)

        Gamma_k = inv_sqrt_aug[row_k] * inv_sqrt_aug[col_k]
        Gamma_a = inv_sqrt_aug[row_a] * inv_sqrt_aug[col_a] if row_a.numel() else None
        Gamma_self = inv_sqrt_aug * inv_sqrt_aug

        deg_clean = self.graph_cache.deg_clean.to(x.device)
        comp_size = self.graph_cache.comp_size.to(x.device)

        deg_hat = (deg_clean + 1.0).clamp(min=1.0)
        inv_sqrt_hat = deg_hat.pow(-0.5)

        alpha_k = inv_sqrt_hat[row_k] * inv_sqrt_hat[col_k]
        alpha_self = inv_sqrt_hat * inv_sqrt_hat

        # E[Gamma_ij] for clean edges (Case A)
        tA = _delta_E_inv_sqrt_caseA(deg_clean, comp_size, p=p, q=q)
        E_Gamma_k = (1.0 - p) * tA[row_k] * tA[col_k]

        corr_k = (alpha_k / E_Gamma_k.clamp(min=1e-12))
        out_k = _scatter_add_rows((Gamma_k * corr_k).unsqueeze(1) * x[row_k], col_k, N)

        # additions
        if row_a.numel():
            sum_a = _scatter_add_rows(Gamma_a.unsqueeze(1) * x[row_a], col_a, N)

            if self.ic_mode:
                out_a = sum_a
            else:
                tB = _delta_E_inv_sqrt_caseB(deg_clean, comp_size, p=p, q=q).to(x.device, dtype=x.dtype)  # [N]
                mu = self.graph_cache.complement_mean_weighted(x, w=tB)  # [N,F]
                sum_g = _scatter_add_scalar(Gamma_a, col_a, N).unsqueeze(1)
                out_a = sum_a - mu * sum_g
        else:
            out_a = torch.zeros_like(out_k)

        # self-loop correction
        if self.correct_self_loop:
            E_inv = _delta_E_inv_self(deg_clean, comp_size, p=p, q=q).to(x.device, dtype=x.dtype)
            corr_self = (alpha_self / E_inv.clamp(min=1e-12))
            out_self = (Gamma_self * corr_self).unsqueeze(1) * x
        else:
            out_self = Gamma_self.unsqueeze(1) * x

        out = out_k + out_a + out_self
        if self.bias is not None:
            out = out + self.bias
        return out


class RADEGINConvGC(nn.Module):
    """
    Batched-GC version of RADEGINConv (sum aggregation).
    """
    def __init__(self, mlp: nn.Module, eps: float = 0.0, train_eps: bool = False, ic_mode: bool = False):
        super().__init__()
        self.mlp = mlp
        self.initial_eps = float(eps)
        self.ic_mode = bool(ic_mode)

        if train_eps:
            self.eps = nn.Parameter(torch.tensor([self.initial_eps], dtype=torch.float32))
        else:
            self.register_buffer("eps", torch.tensor([self.initial_eps], dtype=torch.float32))

        self.graph_cache: Optional[BatchedGraphCache] = None

    def set_graph(self, graph_cache: BatchedGraphCache) -> None:
        self.graph_cache = graph_cache

    def reset_parameters(self) -> None:
        reset(self.mlp)
        self.eps.data.fill_(self.initial_eps)

    def forward(
        self,
        x: torch.Tensor,
        edge_index_clean: torch.Tensor,
        *,
        edge_index_keep: Optional[torch.Tensor] = None,
        edge_index_add: Optional[torch.Tensor] = None,
        p: float = 0.0,
        q: float = 0.0,
    ) -> torch.Tensor:
        assert self.graph_cache is not None, "Call model.set_graph(...) per batch before forward."
        N = x.size(0)

        # -------------------------
        # CLEAN path
        # -------------------------
        if edge_index_keep is None:
            row, col = edge_index_clean[0], edge_index_clean[1]
            neigh_sum = _scatter_add_rows(x[row], col, N)

            if self.ic_mode and (not self.training) and (q > 0.0):
                comp_sum = self.graph_cache.complement_sum_unweighted(x)
                neigh_sum = neigh_sum + q * comp_sum

            out = (1.0 + self.eps.to(x.device)) * x + neigh_sum
            return self.mlp(out)

        # -------------------------
        # RADE / RADE-IC TRAIN path
        # -------------------------
        row_k, col_k = edge_index_keep[0], edge_index_keep[1]
        if edge_index_add is not None and edge_index_add.numel() > 0:
            row_a, col_a = edge_index_add[0], edge_index_add[1]
        else:
            row_a = row_k.new_empty((0,))
            col_a = col_k.new_empty((0,))

        alpha = 1.0 / max(1e-12, (1.0 - float(p)))
        keep_sum = _scatter_add_rows(x[row_k], col_k, N) * alpha

        if row_a.numel() > 0:
            add_sum_raw = _scatter_add_rows(x[row_a], col_a, N)

            if self.ic_mode:
                add_sum = add_sum_raw
            else:
                mu = self.graph_cache.complement_mean_unweighted(x)  # [N,F]
                add_deg = _scatter_add_scalar(
                    torch.ones(row_a.numel(), device=x.device, dtype=x.dtype),
                    col_a,
                    N,
                ).unsqueeze(1)
                add_sum = add_sum_raw - mu * add_deg
        else:
            add_sum = torch.zeros_like(keep_sum)

        h = keep_sum + add_sum
        out = (1.0 + self.eps.to(x.device)) * x + h
        return self.mlp(out)
