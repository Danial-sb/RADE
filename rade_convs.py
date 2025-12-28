from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
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


@dataclass
class GraphCache:
    """
    Cache derived from the CLEAN graph (no self-loops in edge_index_clean).
    - edge_index_clean_dir: directed (both directions), no self-loops
    - deg_clean: clean degree d_i (excluding self-loop)
    - comp_size: |C(i)| = n - 1 - d_i
    """
    edge_index_clean_dir: torch.Tensor  # [2, E_dir]
    num_nodes: int
    deg_clean: torch.Tensor             # [N]
    comp_size: torch.Tensor             # [N]

    @staticmethod
    def from_edge_index(edge_index_clean: torch.Tensor, num_nodes: int) -> "GraphCache":
        row, col = edge_index_clean[0], edge_index_clean[1]
        mask = row != col
        row, col = row[mask], col[mask]
        edge_index_clean_dir = torch.stack([row, col], dim=0)

        ones = torch.ones(col.numel(), dtype=torch.float32, device=col.device)
        deg = torch.zeros((num_nodes,), dtype=torch.float32, device=col.device)
        deg.index_add_(0, col, ones)  # in-degree == degree for undirected (with both directions)

        comp = (num_nodes - 1.0 - deg).clamp(min=0.0)
        return GraphCache(edge_index_clean_dir=edge_index_clean_dir, num_nodes=num_nodes, deg_clean=deg, comp_size=comp)

    def complement_mean_unweighted(self, x: torch.Tensor) -> torch.Tensor:
        """
        μ_i = mean_{j∈C(i)} x_j
        computed exactly in O(E+N) using global sums.
        """
        N = self.num_nodes
        row, col = self.edge_index_clean_dir.to(x.device)
        neigh_sum = _scatter_add_rows(x[row], col, N)               # sum over N(i)
        total_sum = x.sum(dim=0, keepdim=True)                      # sum over all nodes
        comp_sum = total_sum - x - neigh_sum                        # sum over C(i)

        denom = self.comp_size.to(x.device).clamp(min=1.0).unsqueeze(1)
        return comp_sum / denom

    def complement_mean_weighted(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Weighted centering for GCN case:
        μ_i = (∑_{j∈C(i)} w_j x_j) / (∑_{j∈C(i)} w_j),
        where w_j depends only on j (your decoupled approximation reduces weights to node-wise terms).
        """
        N = self.num_nodes
        row, col = self.edge_index_clean_dir.to(x.device)

        wx = w.unsqueeze(1) * x
        total_wx = wx.sum(dim=0, keepdim=True)                      # [1,F]
        total_w = w.sum()                                           # scalar

        neigh_wx = _scatter_add_rows(wx[row], col, N)               # ∑_{j∈N(i)} w_j x_j
        neigh_w = _scatter_add_scalar(w[row], col, N)               # ∑_{j∈N(i)} w_j

        num = total_wx - wx - neigh_wx                              # ∑_{j∈C(i)} w_j x_j
        den = (total_w - w - neigh_w).clamp(min=1e-12).unsqueeze(1) # ∑_{j∈C(i)} w_j
        return num / den


def _delta_E_inv_sqrt_caseA(deg_clean: torch.Tensor, comp_size: torch.Tensor, p: float, q: float) -> torch.Tensor:
    """
    Lemma D.1 (Case A): edge (i,j) is a CLEAN edge.
    E[(S_i+1)^(-1/2)] ≈ 1/sqrt(A) + (3/8) * B / A^(5/2)
    where
      A = (1-p)(d_i-1) + q|C(i)| + 2
      B = (d_i-1)(1-p)p + |C(i)| q(1-q)
    """
    d_minus = (deg_clean - 1.0).clamp(min=0.0)
    A = (1.0 - p) * d_minus + q * comp_size + 2.0
    B = d_minus * (1.0 - p) * p + comp_size * q * (1.0 - q)
    return A.pow(-0.5) + (3.0 / 8.0) * B * A.pow(-2.5)


def _delta_E_inv_sqrt_caseB(deg_clean: torch.Tensor, comp_size: torch.Tensor, p: float, q: float) -> torch.Tensor:
    """
    Lemma D.1 (Case B): edge (i,j) is a CLEAN non-edge.
    A = (1-p)d_i + q(|C(i)|-1) + 2
    B = d_i(1-p)p + (|C(i)|-1) q(1-q)
    """
    c_minus = (comp_size - 1.0).clamp(min=0.0)
    A = (1.0 - p) * deg_clean + q * c_minus + 2.0
    B = deg_clean * (1.0 - p) * p + c_minus * q * (1.0 - q)
    return A.pow(-0.5) + (3.0 / 8.0) * B * A.pow(-2.5)


def _delta_E_inv_self(deg_clean: torch.Tensor, comp_size: torch.Tensor, p: float, q: float) -> torch.Tensor:
    """
    Appendix (g(x)=x^{-1} delta method):
    For X = 1 + Bin(d_i,1-p) + Bin(|C(i)|, q),
    E[X^{-1}] ≈ μ^{-1} + σ^2 μ^{-3}
    where μ = 1 + (1-p)d_i + q|C(i)|, σ^2 = d_i(1-p)p + |C(i)| q(1-q).
    """
    mu = 1.0 + (1.0 - p) * deg_clean + q * comp_size
    var = deg_clean * (1.0 - p) * p + comp_size * q * (1.0 - q)
    return mu.pow(-1.0) + var * mu.pow(-3.0)


class RADEGCNConv(nn.Module):
    """
    GCN symmetric normalization with RADE expectation-preserving corrections.

    Train-time uses:
      - realized Γ_ij = 1/sqrt(d'_i d'_j) on the augmented graph (kept+added+I),
      - neighbor correction factor α_ij / E[Γ_ij] with E[Γ_ij] from Lemma D.1 (Case A),
      - weighted centering on additions using weights ~ E[(S_j+2)^(-1/2)] from Lemma D.1 (Case B),
      - optional self-loop correction using E[(S_i+1)^{-1}] delta approx.

    Eval-time (or if edge_index_keep is None): standard clean GCN on edge_index_clean.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True, correct_self_loop: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        self.correct_self_loop = correct_self_loop

        self.graph_cache: Optional[GraphCache] = None  # set via set_graph()

    def set_graph(self, graph_cache: GraphCache) -> None:
        self.graph_cache = graph_cache

    def reset_parameters(self) -> None:
        # Match PyG-style behavior: reset trainable parameters only.
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
        """
        - If edge_index_keep is None -> clean forward (standard GCN)
        - Else -> RADE forward using keep/add and (p,q)
        """
        assert self.graph_cache is not None, "Call model.set_graph(...) once after loading data."

        x = self.lin(x)
        N = x.size(0)

        # CLEAN path (exact): symmetric normalization on clean graph with self-loops.
        if edge_index_keep is None:
            row, col = edge_index_clean[0], edge_index_clean[1]
            # add self-loops
            ar = torch.arange(N, device=x.device)
            row = torch.cat([row, ar])
            col = torch.cat([col, ar])

            # compute deg_hat and inv_sqrt_hat
            w = torch.ones(row.numel(), device=x.device, dtype=x.dtype)
            deg = _scatter_add_scalar(w, col, N)
            inv_sqrt = deg.clamp(min=1.0).pow(-0.5)

            gamma = inv_sqrt[row] * inv_sqrt[col]  # [E+N]
            out = _scatter_add_rows(gamma.unsqueeze(1) * x[row], col, N)

            if self.bias is not None:
                out = out + self.bias
            return out

        # ---- RADE path ----
        # Build augmented edge set for degree computation: keep + add + self-loops
        row_k, col_k = edge_index_keep[0], edge_index_keep[1]
        row_a, col_a = edge_index_add[0], edge_index_add[1] if (edge_index_add is not None) else (row_k.new_empty((0,)), col_k.new_empty((0,)))

        ar = torch.arange(N, device=x.device)
        row_all = torch.cat([row_k, row_a, ar])
        col_all = torch.cat([col_k, col_a, ar])

        # realized degrees d'_i on augmented graph (self-loop weight = 1)
        w_all = torch.ones(row_all.numel(), device=x.device, dtype=x.dtype)
        deg_aug = _scatter_add_scalar(w_all, col_all, N).clamp(min=1.0)
        inv_sqrt_aug = deg_aug.pow(-0.5)

        # realized Γ on kept/add edges (weight=1) and self-loop edges (also weight=1)
        Gamma_k = inv_sqrt_aug[row_k] * inv_sqrt_aug[col_k]                       # [E_keep_dir]
        Gamma_a = inv_sqrt_aug[row_a] * inv_sqrt_aug[col_a] if row_a.numel() else None
        Gamma_self = inv_sqrt_aug * inv_sqrt_aug                                  # [N]

        # clean α_ij from clean degrees with self-loops
        deg_clean = self.graph_cache.deg_clean.to(x.device)                        # d_i
        deg_hat = (deg_clean + 1.0).clamp(min=1.0)                                 # d_i + 1
        inv_sqrt_hat = deg_hat.pow(-0.5)

        alpha_k = inv_sqrt_hat[row_k] * inv_sqrt_hat[col_k]                        # [E_keep_dir]
        alpha_self = inv_sqrt_hat * inv_sqrt_hat                                   # [N]

        # E[Γ_ij] approximations (your appendix)
        deg_clean_f = deg_clean
        comp_size = self.graph_cache.comp_size.to(x.device)

        tA = _delta_E_inv_sqrt_caseA(deg_clean_f, comp_size, p=p, q=q)             # [N]
        E_Gamma_k = (1.0 - p) * tA[row_k] * tA[col_k]                              # [E_keep_dir]

        # neighbor contribution: Γ_ij * (α_ij / E[Γ_ij]) * x_j
        corr_k = (alpha_k / E_Gamma_k.clamp(min=1e-12))                            # [E_keep_dir]
        out_k = _scatter_add_rows((Gamma_k * corr_k).unsqueeze(1) * x[row_k], col_k, N)

        # addition contribution: Γ_ij * (x_j - μ_i) with weighted μ_i (Case B weights)
        if row_a.numel():
            tB = _delta_E_inv_sqrt_caseB(deg_clean_f, comp_size, p=p, q=q)         # [N]
            # weights across j for fixed i reduce to node-wise term on j (your text: r_ij=q constant cancels)
            wj = tB.to(x.device, dtype=x.dtype)                                    # [N]

            mu = self.graph_cache.complement_mean_weighted(x, w=wj)                # [N,F]

            sum_a = _scatter_add_rows(Gamma_a.unsqueeze(1) * x[row_a], col_a, N)   # Σ Γ x_j
            sum_g = _scatter_add_scalar(Gamma_a, col_a, N).unsqueeze(1)            # Σ Γ

            out_a = sum_a - mu * sum_g
        else:
            out_a = torch.zeros_like(out_k)

        # self-loop contribution: Γ_ii * (α_ii / E[Γ_ii]) * x_i  (optional, recommended)
        if self.correct_self_loop:
            E_inv = _delta_E_inv_self(deg_clean_f, comp_size, p=p, q=q).to(x.device, dtype=x.dtype)  # ≈ E[1/d'_i]
            E_Gamma_self = E_inv  # since self-loop edge weight=1 => Γ_ii = 1/d'_i
            corr_self = (alpha_self / E_Gamma_self.clamp(min=1e-12))                               # [N]
            out_self = (Gamma_self * corr_self).unsqueeze(1) * x
        else:
            # leave self-loop as realized (biased in general)
            out_self = Gamma_self.unsqueeze(1) * x

        out = out_k + out_a + out_self
        if self.bias is not None:
            out = out + self.bias
        return out


class RADEGINConv(nn.Module):
    """
    Sum-aggregation GIN-style RADE.

    Clean (eval or no keep/add):
        out_i = MLP( (1+eps) x_i + sum_{j in N(i)} x_j )

    RADE train-time (with keep/add):
        h_i = (1/(1-p)) * sum_{(j->i) in keep} x_j
              + sum_{(j->i) in add} (x_j - mu_i),
        mu_i = mean_{k in C(i)} x_k  (computed from CLEAN graph cache)
        out_i = MLP( (1+eps) x_i + h_i )

    Note: q is accepted for API consistency but not needed in the GIN sum-aggregator correction.
    """

    def __init__(self, mlp: nn.Module, eps: float = 0.0, train_eps: bool = False):
        super().__init__()
        self.mlp = mlp
        self.initial_eps = float(eps)

        if train_eps:
            self.eps = nn.Parameter(torch.tensor([self.initial_eps], dtype=torch.float32))
        else:
            self.register_buffer("eps", torch.tensor([self.initial_eps], dtype=torch.float32))

        self.graph_cache: Optional[GraphCache] = None

    def set_graph(self, graph_cache: GraphCache) -> None:
        self.graph_cache = graph_cache

    def reset_parameters(self) -> None:
        # Reset MLP (PyG-style)
        reset(self.mlp)
        # Reset eps to initial value (works for Parameter or buffer)
        self.eps.data.fill_(self.initial_eps)

    def forward(
        self,
        x: torch.Tensor,
        edge_index_clean: torch.Tensor,
        *,
        edge_index_keep: Optional[torch.Tensor] = None,
        edge_index_add: Optional[torch.Tensor] = None,
        p: float = 0.0,
        q: float = 0.0,   # accepted for unified API; unused here
    ) -> torch.Tensor:
        assert self.graph_cache is not None, "Call model.set_graph(...) once after loading data."
        N = x.size(0)

        # -------------------------
        # CLEAN path (standard GIN sum aggregation)
        # -------------------------
        if edge_index_keep is None:
            row, col = edge_index_clean[0], edge_index_clean[1]
            neigh_sum = _scatter_add_rows(x[row], col, N)  # sum_{j in N(i)} x_j
            out = (1.0 + self.eps.to(x.device)) * x + neigh_sum
            return self.mlp(out)

        # -------------------------
        # RADE
        # -------------------------
        row_k, col_k = edge_index_keep[0], edge_index_keep[1]
        if edge_index_add is not None and edge_index_add.numel() > 0:
            row_a, col_a = edge_index_add[0], edge_index_add[1]
        else:
            row_a = row_k.new_empty((0,))
            col_a = col_k.new_empty((0,))

        # kept-neighbor sum with expectation-preserving rescaling alpha = 1/(1-p)
        alpha = 1.0 / max(1e-12, (1.0 - float(p)))
        keep_sum = _scatter_add_rows(x[row_k], col_k, N) * alpha

        # added non-neighbor sum with centering by mu_i = mean_{C(i)} x_j
        if row_a.numel() > 0:
            mu = self.graph_cache.complement_mean_unweighted(x)  # [N,F]
            add_sum = _scatter_add_rows(x[row_a], col_a, N)
            add_deg = _scatter_add_scalar(
                torch.ones(row_a.numel(), device=x.device, dtype=x.dtype),
                col_a,
                N,
            ).unsqueeze(1)
            add_sum = add_sum - mu * add_deg
        else:
            add_sum = torch.zeros_like(keep_sum)

        h = keep_sum + add_sum
        out = (1.0 + self.eps.to(x.device)) * x + h
        return self.mlp(out)

