from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn.inits import reset


def _scatter_add_rows(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """
    Row-wise scatter-add.

    Args:
        values: [E, F]
        index:  [E]
        dim_size: number of destination rows

    Returns:
        out: [dim_size, F], where out[i] = sum_{e: index[e]=i} values[e]
    """
    out = torch.zeros((dim_size, values.size(1)), device=values.device, dtype=values.dtype)
    out.index_add_(0, index, values)
    return out


def _scatter_add_scalar(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """
    Scalar scatter-add.

    Args:
        values: [E]
        index:  [E]
        dim_size: number of destination entries

    Returns:
        out: [dim_size], where out[i] = sum_{e: index[e]=i} values[e]
    """
    out = torch.zeros((dim_size,), device=values.device, dtype=values.dtype)
    out.index_add_(0, index, values)
    return out


def _validate_rade_variant(rade_variant: str) -> str:
    rade_variant = str(rade_variant).lower().strip()
    if rade_variant not in {"rade-of", "rade-ofs"}:
        raise ValueError(
            f"rade_variant must be one of {{'rade-of', 'rade-ofs'}}. Got {rade_variant}"
        )
    return rade_variant


@dataclass
class GraphCache:
    """
    Cache derived from the clean graph G (without self-loops in edge_index_clean).

    Stored quantities:
      - edge_index_clean_dir: directed clean edges (both directions), no self-loops
      - deg_clean[i]: clean degree d_i in G (excluding self-loop)
      - comp_size[i]: |C(i)| = n - 1 - d_i, where C(i) is the clean non-neighbor set
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
        deg.index_add_(0, col, ones)  # for undirected graphs stored both ways: indegree == degree

        comp = (num_nodes - 1.0 - deg).clamp(min=0.0)
        return GraphCache(
            edge_index_clean_dir=edge_index_clean_dir,
            num_nodes=num_nodes,
            deg_clean=deg,
            comp_size=comp,
        )

    def complement_mean_unweighted(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unweighted clean non-neighbor mean:
            mu_i = (1 / |C(i)|) sum_{j in C(i)} x_j

        Computed exactly in O(E + N) via global sums.
        """
        N = self.num_nodes
        row, col = self.edge_index_clean_dir.to(x.device)
        neigh_sum = _scatter_add_rows(x[row], col, N)   # sum_{j in N(i)} x_j
        total_sum = x.sum(dim=0, keepdim=True)          # sum_j x_j
        comp_sum = total_sum - x - neigh_sum            # sum_{j in C(i)} x_j

        denom = self.comp_size.to(x.device).clamp(min=1.0).unsqueeze(1)
        return comp_sum / denom

    def complement_mean_weighted(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Weighted clean non-neighbor mean:
            mu_i = (sum_{j in C(i)} w_j x_j) / (sum_{j in C(i)} w_j)

        This is used for the GCN Case-B weighted centering approximation, where the
        paper reduces the non-edge weights to node-wise terms w_j under the decoupled
        approximation.
        """
        N = self.num_nodes
        row, col = self.edge_index_clean_dir.to(x.device)

        wx = w.unsqueeze(1) * x
        total_wx = wx.sum(dim=0, keepdim=True)
        total_w = w.sum()

        neigh_wx = _scatter_add_rows(wx[row], col, N)   # sum_{j in N(i)} w_j x_j
        neigh_w = _scatter_add_scalar(w[row], col, N)   # sum_{j in N(i)} w_j

        num = total_wx - wx - neigh_wx                  # sum_{j in C(i)} w_j x_j
        den = (total_w - w - neigh_w).clamp(min=1e-12).unsqueeze(1)
        return num / den

    def complement_sum_unweighted(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unweighted clean non-neighbor sum:
            S_i = sum_{j in C(i)} x_j
        """
        N = self.num_nodes
        row, col = self.edge_index_clean_dir.to(x.device)
        neigh_sum = _scatter_add_rows(x[row], col, N)
        total_sum = x.sum(dim=0, keepdim=True)
        comp_sum = total_sum - x - neigh_sum
        return comp_sum

    def complement_sum_weighted(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Weighted clean non-neighbor sum:
            S_i = sum_{j in C(i)} w_j x_j

        Used in the GCN RADE-OFS inference correction under the
        decoupled Case-B approximation.
        """
        N = self.num_nodes
        row, col = self.edge_index_clean_dir.to(x.device)

        wx = w.unsqueeze(1) * x
        total_wx = wx.sum(dim=0, keepdim=True)
        neigh_wx = _scatter_add_rows(wx[row], col, N)
        comp_wx = total_wx - wx - neigh_wx
        return comp_wx


def _delta_E_inv_caseA(
    deg_clean: torch.Tensor,
    comp_size: torch.Tensor,
    p: float,
    q: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Case A (clean edge): delta-method approximation for E[X^{-1}],
    where X is the self-loop-included sampled degree term appearing in
    GCN normalization for a clean edge contribution.

    Mean and variance:
        mu  = (1-p)(d_i - 1) + q|C(i)| + 2
        var = (d_i - 1)(1-p)p + |C(i)| q(1-q)

    Approximation:
        E[X^{-1}] ≈ mu^{-1} + var * mu^{-3}
    """
    d_minus = (deg_clean - 1.0).clamp(min=0.0)
    mu = (1.0 - p) * d_minus + q * comp_size + 2.0
    var = d_minus * (1.0 - p) * p + comp_size * q * (1.0 - q)
    mu = mu.clamp(min=eps)
    return mu.pow(-1.0) + var * mu.pow(-3.0)


def _delta_E_inv_caseB(
    deg_clean: torch.Tensor,
    comp_size: torch.Tensor,
    p: float,
    q: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Case B (clean non-edge): delta-method approximation for E[X^{-1}],
    where X is the self-loop-included sampled degree term appearing in
    GCN normalization for a non-edge contribution.

    Mean and variance:
        mu  = (1-p)d_i + q(|C(i)| - 1) + 2
        var = d_i(1-p)p + (|C(i)| - 1) q(1-q)

    Approximation:
        E[X^{-1}] ≈ mu^{-1} + var * mu^{-3}
    """
    c_minus = (comp_size - 1.0).clamp(min=0.0)
    mu = (1.0 - p) * deg_clean + q * c_minus + 2.0
    var = deg_clean * (1.0 - p) * p + c_minus * q * (1.0 - q)
    mu = mu.clamp(min=eps)
    return mu.pow(-1.0) + var * mu.pow(-3.0)


def _delta_E_inv_sqrt_caseA(
    deg_clean: torch.Tensor,
    comp_size: torch.Tensor,
    p: float,
    q: float,
) -> torch.Tensor:
    """
    Case A (clean edge): delta-method approximation for E[X^{-1/2}],
    where X is the self-loop-included sampled degree term used in the
    GCN normalization factor for a clean edge contribution.

    Let
        A = (1-p)(d_i - 1) + q|C(i)| + 2
        B = (d_i - 1)(1-p)p + |C(i)| q(1-q)

    Then
        E[X^{-1/2}] ≈ A^{-1/2} + (3/8) B A^{-5/2}.
    """
    d_minus = (deg_clean - 1.0).clamp(min=0.0)
    A = (1.0 - p) * d_minus + q * comp_size + 2.0
    B = d_minus * (1.0 - p) * p + comp_size * q * (1.0 - q)
    return A.pow(-0.5) + (3.0 / 8.0) * B * A.pow(-2.5)


def _delta_E_inv_sqrt_caseB(
    deg_clean: torch.Tensor,
    comp_size: torch.Tensor,
    p: float,
    q: float,
) -> torch.Tensor:
    """
    Case B (clean non-edge): delta-method approximation for E[X^{-1/2}],
    where X is the self-loop-included sampled degree term used in the
    GCN normalization factor for a non-edge contribution.

    Let
        A = (1-p)d_i + q(|C(i)| - 1) + 2
        B = d_i(1-p)p + (|C(i)| - 1) q(1-q)

    Then
        E[X^{-1/2}] ≈ A^{-1/2} + (3/8) B A^{-5/2}.
    """
    c_minus = (comp_size - 1.0).clamp(min=0.0)
    A = (1.0 - p) * deg_clean + q * c_minus + 2.0
    B = deg_clean * (1.0 - p) * p + c_minus * q * (1.0 - q)
    return A.pow(-0.5) + (3.0 / 8.0) * B * A.pow(-2.5)


def _delta_E_inv_self(deg_clean: torch.Tensor, comp_size: torch.Tensor, p: float, q: float) -> torch.Tensor:
    """
    Delta-method approximation for the self-loop normalization term.

    For
        X = 1 + Bin(d_i, 1-p) + Bin(|C(i)|, q),

    we use
        E[X^{-1}] ≈ mu^{-1} + var * mu^{-3},
    where
        mu  = 1 + (1-p)d_i + q|C(i)|,
        var = d_i(1-p)p + |C(i)|q(1-q).
    """
    mu = 1.0 + (1.0 - p) * deg_clean + q * comp_size
    var = deg_clean * (1.0 - p) * p + comp_size * q * (1.0 - q)
    return mu.pow(-1.0) + var * mu.pow(-3.0)


class RADEGCNConv(nn.Module):
    """
    GCN symmetric normalization with paper-aligned RADE variants.

    Arguments:
      - rade_variant='rade-of'   -> overfitting-oriented variant
      - rade_variant='rade-ofs'  -> joint overfitting + over-squashing variant
      - ep_correction=True       -> use expectation-preserving aggregation correction
      - ep_correction=False      -> same sampled keep/add operator path, but without
                                    expectation-preserving correction (matched ablation)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        correct_self_loop: bool = True,
        rade_variant: str = "rade-of",
        ep_correction: bool = True,
    ):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.correct_self_loop = bool(correct_self_loop)
        self.rade_variant = _validate_rade_variant(rade_variant)
        self.ep_correction = bool(ep_correction)

        self.graph_cache: Optional[GraphCache] = None

    def set_graph(self, graph_cache: GraphCache) -> None:
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
        assert self.graph_cache is not None, "Call model.set_graph(...) once after loading data."

        x = self.lin(x)
        N = x.size(0)
        use_ofs = (self.rade_variant == "rade-ofs")

        # ------------------------------------------------------------------
        # Clean-graph path:
        #   - used for ordinary clean inference
        #   - when rade_variant='rade-ofs' and eval, includes the RADE-OFS
        #     inference-time non-edge correction
        # ------------------------------------------------------------------
        if edge_index_keep is None:
            row, col = edge_index_clean[0], edge_index_clean[1]
            ar = torch.arange(N, device=x.device)

            # Add self-loops for standard GCN normalization.
            row = torch.cat([row, ar])
            col = torch.cat([col, ar])

            w = torch.ones(row.numel(), device=x.device, dtype=x.dtype)
            deg = _scatter_add_scalar(w, col, N)
            inv_sqrt = deg.clamp(min=1.0).pow(-0.5)

            gamma = inv_sqrt[row] * inv_sqrt[col]
            out = _scatter_add_rows(gamma.unsqueeze(1) * x[row], col, N)

            # RADE-OFS inference correction:
            #   out += q * tB(i) * sum_{j in C(i)} tB(j) x_j
            # under the decoupled Case-B approximation.
            if use_ofs and (not self.training) and (q > 0.0):
                deg_clean = self.graph_cache.deg_clean.to(x.device)
                comp_size = self.graph_cache.comp_size.to(x.device)

                tB = _delta_E_inv_sqrt_caseB(deg_clean, comp_size, p=p, q=q).to(x.device, dtype=x.dtype)
                comp_wx = self.graph_cache.complement_sum_weighted(x, w=tB)
                out = out + (q * tB).unsqueeze(1) * comp_wx

            if self.bias is not None:
                out = out + self.bias
            return out

        # ------------------------------------------------------------------
        # Sampled-graph training path:
        # edge_index_keep: retained clean edges
        # edge_index_add:  sampled clean non-edges
        # ------------------------------------------------------------------
        row_k, col_k = edge_index_keep[0], edge_index_keep[1]

        if edge_index_add is not None and edge_index_add.numel() > 0:
            row_a, col_a = edge_index_add[0], edge_index_add[1]
        else:
            row_a = row_k.new_empty((0,))
            col_a = row_k.new_empty((0,))

        ar = torch.arange(N, device=x.device)

        # Build sampled graph with self-loops to obtain the realized GCN coefficients.
        row_all = torch.cat([row_k, row_a, ar])
        col_all = torch.cat([col_k, col_a, ar])

        w_all = torch.ones(row_all.numel(), device=x.device, dtype=x.dtype)
        deg_aug = _scatter_add_scalar(w_all, col_all, N).clamp(min=1.0)
        inv_sqrt_aug = deg_aug.pow(-0.5)

        Gamma_k = inv_sqrt_aug[row_k] * inv_sqrt_aug[col_k]
        Gamma_a = inv_sqrt_aug[row_a] * inv_sqrt_aug[col_a] if row_a.numel() else None
        Gamma_self = inv_sqrt_aug * inv_sqrt_aug

        # ------------------------------------------------------------------
        # Matched no-correction path:
        # same sampled keep/add tensors and same operator path, but without
        # expectation-preserving correction
        # ------------------------------------------------------------------
        if not self.ep_correction:
            out_k = _scatter_add_rows(Gamma_k.unsqueeze(1) * x[row_k], col_k, N)

            if row_a.numel():
                out_a = _scatter_add_rows(Gamma_a.unsqueeze(1) * x[row_a], col_a, N)
            else:
                out_a = torch.zeros_like(out_k)

            out_self = Gamma_self.unsqueeze(1) * x

            out = out_k + out_a + out_self
            if self.bias is not None:
                out = out + self.bias
            return out

        # ------------------------------------------------------------------
        # Paper-aligned corrected path:
        #   - rade_variant='rade-of'   : RADE-OF
        #   - rade_variant='rade-ofs'  : RADE-OFS
        # ------------------------------------------------------------------
        deg_clean = self.graph_cache.deg_clean.to(x.device)
        comp_size = self.graph_cache.comp_size.to(x.device)

        # Clean target coefficients (self-loop included clean graph).
        deg_hat = (deg_clean + 1.0).clamp(min=1.0)
        inv_sqrt_hat = deg_hat.pow(-0.5)

        alpha_k = inv_sqrt_hat[row_k] * inv_sqrt_hat[col_k]
        alpha_self = inv_sqrt_hat * inv_sqrt_hat

        # Case A: expected realized coefficient on retained clean edges.
        tA = _delta_E_inv_sqrt_caseA(deg_clean, comp_size, p=p, q=q)
        E_Gamma_k = (1.0 - p) * tA[row_k] * tA[col_k]

        corr_k = alpha_k / E_Gamma_k.clamp(min=1e-12)
        out_k = _scatter_add_rows((Gamma_k * corr_k).unsqueeze(1) * x[row_k], col_k, N)

        # Sampled non-edge contributions.
        if row_a.numel():
            sum_a = _scatter_add_rows(Gamma_a.unsqueeze(1) * x[row_a], col_a, N)

            if use_ofs:
                # RADE-OFS training:
                # keep non-edge contributions uncentered so their effect is carried to inference.
                out_a = sum_a
            else:
                # RADE-OF training:
                # weighted centering under the Case-B decoupled approximation.
                tB = _delta_E_inv_sqrt_caseB(deg_clean, comp_size, p=p, q=q).to(x.device, dtype=x.dtype)
                mu = self.graph_cache.complement_mean_weighted(x, w=tB)

                # sum_g[i] = sum over sampled added non-edges into i of realized Gamma_a
                sum_g = _scatter_add_scalar(Gamma_a, col_a, N).unsqueeze(1)
                out_a = sum_a - mu * sum_g
        else:
            out_a = torch.zeros_like(out_k)

        # Self-loop correction.
        if self.correct_self_loop:
            E_Gamma_self = _delta_E_inv_self(deg_clean, comp_size, p=p, q=q).to(x.device, dtype=x.dtype)
            corr_self = alpha_self / E_Gamma_self.clamp(min=1e-12)
            out_self = (Gamma_self * corr_self).unsqueeze(1) * x
        else:
            out_self = Gamma_self.unsqueeze(1) * x

        out = out_k + out_a + out_self
        if self.bias is not None:
            out = out + self.bias
        return out


class RADEGINConv(nn.Module):
    """
    GIN sum aggregation with paper-aligned RADE variants.

    Arguments:
      - rade_variant='rade-of'   -> overfitting-oriented variant
      - rade_variant='rade-ofs'  -> joint overfitting + over-squashing variant
      - ep_correction=True       -> use expectation-preserving aggregation correction
      - ep_correction=False      -> same sampled keep/add operator path, but without
                                    expectation-preserving correction
    """

    def __init__(
        self,
        mlp: nn.Module,
        eps: float = 0.0,
        train_eps: bool = False,
        rade_variant: str = "rade-of",
        ep_correction: bool = True,
    ):
        super().__init__()
        self.mlp = mlp
        self.initial_eps = float(eps)
        self.rade_variant = _validate_rade_variant(rade_variant)
        self.ep_correction = bool(ep_correction)

        if train_eps:
            self.eps = nn.Parameter(torch.tensor([self.initial_eps], dtype=torch.float32))
        else:
            self.register_buffer("eps", torch.tensor([self.initial_eps], dtype=torch.float32))

        self.graph_cache: Optional[GraphCache] = None

    def set_graph(self, graph_cache: GraphCache) -> None:
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
        assert self.graph_cache is not None, "Call model.set_graph(...) once after loading data."
        N = x.size(0)
        use_ofs = (self.rade_variant == "rade-ofs")

        # ------------------------------------------------------------------
        # Clean-graph path:
        #   - used for ordinary clean inference
        #   - when rade_variant='rade-ofs' and eval, includes the RADE-OFS
        #     inference-time non-edge correction
        # ------------------------------------------------------------------
        if edge_index_keep is None:
            row, col = edge_index_clean[0], edge_index_clean[1]
            neigh_sum = _scatter_add_rows(x[row], col, N)

            # RADE-OFS inference correction:
            #   neigh_sum += q * sum_{j in C(i)} x_j
            if use_ofs and (not self.training) and (q > 0.0):
                comp_sum = self.graph_cache.complement_sum_unweighted(x)
                neigh_sum = neigh_sum + q * comp_sum

            out = (1.0 + self.eps.to(x.device)) * x + neigh_sum
            return self.mlp(out)

        # ------------------------------------------------------------------
        # Sampled-graph training path:
        # edge_index_keep: retained clean edges
        # edge_index_add:  sampled clean non-edges
        # ------------------------------------------------------------------
        row_k, col_k = edge_index_keep[0], edge_index_keep[1]
        if edge_index_add is not None and edge_index_add.numel() > 0:
            row_a, col_a = edge_index_add[0], edge_index_add[1]
        else:
            row_a = row_k.new_empty((0,))
            col_a = col_k.new_empty((0,))

        # ------------------------------------------------------------------
        # Matched no-correction path:
        # same sampled keep/add path, but without expectation-preserving correction
        # ------------------------------------------------------------------
        if not self.ep_correction:
            keep_sum = _scatter_add_rows(x[row_k], col_k, N)

            if row_a.numel() > 0:
                add_sum = _scatter_add_rows(x[row_a], col_a, N)
            else:
                add_sum = torch.zeros_like(keep_sum)

            h = keep_sum + add_sum
            out = (1.0 + self.eps.to(x.device)) * x + h
            return self.mlp(out)

        # ------------------------------------------------------------------
        # Corrected path:
        #   - rade_variant='rade-of'
        #   - rade_variant='rade-ofs'
        # ------------------------------------------------------------------

        # Retained clean edges: expectation-preserving rescaling by 1 / (1-p).
        alpha = 1.0 / max(1e-12, (1.0 - float(p)))
        keep_sum = _scatter_add_rows(x[row_k], col_k, N) * alpha

        if row_a.numel() > 0:
            add_sum_raw = _scatter_add_rows(x[row_a], col_a, N)

            if use_ofs:
                # RADE-OFS training:
                # keep non-edge contributions uncentered.
                add_sum = add_sum_raw
            else:
                # RADE-OF training:
                # subtract mu_i from each sampled non-edge contribution.
                mu = self.graph_cache.complement_mean_unweighted(x)
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