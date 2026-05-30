from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn.inits import reset
from torch_geometric.utils import softmax


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


def _scatter_add_heads(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """
    Scatter-add for multi-head messages.

    Args:
        values: [E, H, F]
        index:  [E]

    Returns:
        out: [N, H, F]
    """
    if values.numel() == 0:
        return torch.zeros(
            (dim_size, values.size(1), values.size(2)),
            device=values.device,
            dtype=values.dtype,
        )
    flat = values.reshape(values.size(0), values.size(1) * values.size(2))
    out = _scatter_add_rows(flat, index, dim_size)
    return out.view(dim_size, values.size(1), values.size(2))


def _validate_rade_variant(rade_variant: str) -> str:
    rade_variant = str(rade_variant).lower().strip()
    if rade_variant not in {"rade-of", "rade-ofs"}:
        raise ValueError(
            f"rade_variant must be one of {{'rade-of', 'rade-ofs'}}. Got {rade_variant}"
        )
    return rade_variant


def _validate_ep_expectation_mode(ep_expectation_mode: str) -> str:
    ep_expectation_mode = str(ep_expectation_mode).lower().strip()
    if ep_expectation_mode not in {"analytic", "empirical_ema"}:
        raise ValueError(
            f"ep_expectation_mode must be one of {{'analytic', 'empirical_ema'}}. "
            f"Got {ep_expectation_mode}"
        )
    return ep_expectation_mode


def _validate_ep_emp_average_mode(ep_emp_average_mode: str) -> str:
    ep_emp_average_mode = str(ep_emp_average_mode).lower().strip()
    if ep_emp_average_mode not in {"ema", "running_mean"}:
        raise ValueError(
            f"ep_emp_average_mode must be one of {{'ema', 'running_mean'}}. "
            f"Got {ep_emp_average_mode}"
        )
    return ep_emp_average_mode


def _directed_edge_keys(row: torch.Tensor, col: torch.Tensor, num_nodes: int) -> torch.Tensor:
    return row.to(torch.long) * int(num_nodes) + col.to(torch.long)


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
      - ep_expectation_mode='analytic'      -> original paper expectation
      - ep_expectation_mode='empirical_ema' -> lagged online EMA ablation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        correct_self_loop: bool = True,
        rade_variant: str = "rade-of",
        ep_correction: bool = True,
        ep_expectation_mode: str = "analytic",
        ep_emp_average_mode: str = "ema",
        ep_emp_beta: float = 0.1,
        ep_emp_eps: float = 1e-12,
    ):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.correct_self_loop = bool(correct_self_loop)
        self.rade_variant = _validate_rade_variant(rade_variant)
        self.ep_correction = bool(ep_correction)
        self.ep_expectation_mode = _validate_ep_expectation_mode(ep_expectation_mode)
        self.ep_emp_average_mode = _validate_ep_emp_average_mode(ep_emp_average_mode)
        self.ep_emp_beta = float(ep_emp_beta)
        self.ep_emp_eps = float(ep_emp_eps)

        self.graph_cache: Optional[GraphCache] = None

        self.register_buffer("emp_clean_edge_raw", torch.empty(0, dtype=torch.float32))
        self.register_buffer("emp_clean_edge_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("emp_clean_edge_sq_raw", torch.empty(0, dtype=torch.float32))
        self.register_buffer("emp_clean_edge_sq_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("emp_self_raw", torch.empty(0, dtype=torch.float32))
        self.register_buffer("emp_self_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("emp_self_sq_raw", torch.empty(0, dtype=torch.float32))
        self.register_buffer("emp_self_sq_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("emp_nonedge_row", torch.empty(0, dtype=torch.long))
        self.register_buffer("emp_nonedge_col", torch.empty(0, dtype=torch.long))
        self.register_buffer("emp_nonedge_raw", torch.empty(0, dtype=torch.float32))
        self.register_buffer("emp_nonedge_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("emp_nonedge_sq_raw", torch.empty(0, dtype=torch.float32))
        self.register_buffer("emp_nonedge_sq_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("emp_num_updates", torch.tensor(0, dtype=torch.long))

        self._clean_edge_lookup: dict[int, int] = {}
        self._emp_nonedge_lookup: dict[int, int] = {}

    def set_graph(self, graph_cache: GraphCache) -> None:
        self.graph_cache = graph_cache
        row, col = graph_cache.edge_index_clean_dir[0], graph_cache.edge_index_clean_dir[1]
        keys = _directed_edge_keys(row, col, graph_cache.num_nodes).detach().cpu().tolist()
        self._clean_edge_lookup = {int(key): idx for idx, key in enumerate(keys)}
        self._rebuild_emp_nonedge_lookup()

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def _reset_emp_nonedge_sparse(self) -> None:
        device = self.emp_nonedge_scale.device
        self.emp_nonedge_row = torch.empty(0, dtype=torch.long, device=device)
        self.emp_nonedge_col = torch.empty(0, dtype=torch.long, device=device)
        self.emp_nonedge_raw = torch.empty(0, dtype=torch.float32, device=device)
        self.emp_nonedge_scale.fill_(1.0)
        self.emp_nonedge_sq_raw = torch.empty(0, dtype=torch.float32, device=device)
        self.emp_nonedge_sq_scale.fill_(1.0)
        self._emp_nonedge_lookup = {}

    def _rebuild_emp_nonedge_lookup(self) -> None:
        if self.graph_cache is None:
            self._emp_nonedge_lookup = {}
            return
        if self.emp_nonedge_row.numel() == 0:
            self._emp_nonedge_lookup = {}
            return
        keys = _directed_edge_keys(
            self.emp_nonedge_row.detach().cpu(),
            self.emp_nonedge_col.detach().cpu(),
            self.graph_cache.num_nodes,
        ).tolist()
        self._emp_nonedge_lookup = {int(key): idx for idx, key in enumerate(keys)}

    def _gcn_sample_coeffs(
        self,
        edge_index_keep: Optional[torch.Tensor],
        edge_index_add: Optional[torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        assert self.graph_cache is not None, "Call model.set_graph(...) before using empirical EP stats."

        N = int(self.graph_cache.num_nodes)

        if edge_index_keep is not None and edge_index_keep.numel() > 0:
            row_k = edge_index_keep[0].to(device)
            col_k = edge_index_keep[1].to(device)
        else:
            row_k = torch.empty(0, dtype=torch.long, device=device)
            col_k = torch.empty(0, dtype=torch.long, device=device)

        if edge_index_add is not None and edge_index_add.numel() > 0:
            row_a = edge_index_add[0].to(device)
            col_a = edge_index_add[1].to(device)
        else:
            row_a = torch.empty(0, dtype=torch.long, device=device)
            col_a = torch.empty(0, dtype=torch.long, device=device)

        ar = torch.arange(N, device=device)
        row_all = torch.cat([row_k, row_a, ar], dim=0)
        col_all = torch.cat([col_k, col_a, ar], dim=0)

        w_all = torch.ones(row_all.numel(), device=device, dtype=dtype)
        deg_aug = _scatter_add_scalar(w_all, col_all, N).clamp(min=1.0)
        inv_sqrt_aug = deg_aug.pow(-0.5)

        gamma_k = inv_sqrt_aug[row_k] * inv_sqrt_aug[col_k]
        gamma_a = inv_sqrt_aug[row_a] * inv_sqrt_aug[col_a] if row_a.numel() else torch.empty(0, device=device, dtype=dtype)
        gamma_self = inv_sqrt_aug * inv_sqrt_aug
        return row_k, col_k, gamma_k, row_a, col_a, gamma_a, gamma_self

    def _append_nonedge_raw(
        self,
        row_a: torch.Tensor,
        col_a: torch.Tensor,
        gamma_a: torch.Tensor,
        *,
        new_mult: float,
    ) -> None:
        if row_a.numel() == 0:
            return
        assert self.graph_cache is not None, "Call model.set_graph(...) before using empirical EP stats."

        scale = float(self.emp_nonedge_scale.item())
        if scale <= 0.0:
            raise RuntimeError("emp_nonedge_scale must be positive when beta < 1.")

        num_nodes = int(self.graph_cache.num_nodes)
        gamma_detached = gamma_a.detach().to(dtype=torch.float32)
        raw_updates = (float(new_mult) * gamma_detached / scale).cpu().tolist()
        sq_scale = float(self.emp_nonedge_sq_scale.item())
        if sq_scale <= 0.0:
            raise RuntimeError("emp_nonedge_sq_scale must be positive when beta < 1.")
        sq_raw_updates = (float(new_mult) * gamma_detached.square() / sq_scale).cpu().tolist()
        row_list = row_a.detach().cpu().tolist()
        col_list = col_a.detach().cpu().tolist()

        new_rows = []
        new_cols = []
        new_vals = []
        new_sq_vals = []
        existing_indices = []
        existing_updates = []
        existing_sq_updates = []

        for row_i, col_i, raw_i, sq_raw_i in zip(row_list, col_list, raw_updates, sq_raw_updates):
            key = int(row_i) * num_nodes + int(col_i)
            idx = self._emp_nonedge_lookup.get(key)
            if idx is None:
                idx = int(self.emp_nonedge_raw.numel()) + len(new_rows)
                self._emp_nonedge_lookup[key] = idx
                new_rows.append(int(row_i))
                new_cols.append(int(col_i))
                new_vals.append(float(raw_i))
                new_sq_vals.append(float(sq_raw_i))
            else:
                existing_indices.append(int(idx))
                existing_updates.append(float(raw_i))
                existing_sq_updates.append(float(sq_raw_i))

        if existing_indices:
            idx_t = torch.tensor(existing_indices, dtype=torch.long, device=self.emp_nonedge_raw.device)
            upd_t = torch.tensor(existing_updates, dtype=torch.float32, device=self.emp_nonedge_raw.device)
            self.emp_nonedge_raw.index_add_(0, idx_t, upd_t)
            sq_upd_t = torch.tensor(existing_sq_updates, dtype=torch.float32, device=self.emp_nonedge_sq_raw.device)
            self.emp_nonedge_sq_raw.index_add_(0, idx_t.to(self.emp_nonedge_sq_raw.device), sq_upd_t)

        if new_rows:
            dev = self.emp_nonedge_raw.device
            row_t = torch.tensor(new_rows, dtype=torch.long, device=dev)
            col_t = torch.tensor(new_cols, dtype=torch.long, device=dev)
            val_t = torch.tensor(new_vals, dtype=torch.float32, device=dev)
            sq_val_t = torch.tensor(new_sq_vals, dtype=torch.float32, device=self.emp_nonedge_sq_raw.device)
            self.emp_nonedge_row = torch.cat([self.emp_nonedge_row, row_t], dim=0)
            self.emp_nonedge_col = torch.cat([self.emp_nonedge_col, col_t], dim=0)
            self.emp_nonedge_raw = torch.cat([self.emp_nonedge_raw, val_t], dim=0)
            self.emp_nonedge_sq_raw = torch.cat([self.emp_nonedge_sq_raw, sq_val_t], dim=0)

    def _get_clean_edge_emp(
        self,
        row: torch.Tensor,
        col: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if row.numel() == 0:
            return torch.empty(0, dtype=dtype, device=device)
        assert self.graph_cache is not None, "Call model.set_graph(...) before using empirical EP stats."

        keys = _directed_edge_keys(row.detach().cpu(), col.detach().cpu(), self.graph_cache.num_nodes).tolist()
        idx = [self._clean_edge_lookup[int(key)] for key in keys]
        idx_t = torch.tensor(idx, dtype=torch.long, device=self.emp_clean_edge_raw.device)
        vals = self.emp_clean_edge_scale * self.emp_clean_edge_raw.index_select(0, idx_t)
        return vals.to(device=device, dtype=dtype)

    def _apply_emp_nonedge_expectation(self, x: torch.Tensor) -> torch.Tensor:
        if self.emp_nonedge_raw.numel() == 0:
            return torch.zeros((x.size(0), x.size(1)), device=x.device, dtype=x.dtype)

        rows = self.emp_nonedge_row.to(x.device)
        cols = self.emp_nonedge_col.to(x.device)
        vals = (self.emp_nonedge_scale * self.emp_nonedge_raw).to(x.device, dtype=x.dtype)
        return _scatter_add_rows(vals.unsqueeze(1) * x[rows], cols, x.size(0))

    def empirical_gcn_variance_proxy(
        self,
        m: torch.Tensor,
        m_sq: torch.Tensor,
        *,
        p: float,
        q: float,
        eps: float,
    ) -> Optional[torch.Tensor]:
        if self.ep_expectation_mode != "empirical_ema":
            return None
        assert self.graph_cache is not None, "Call model.set_graph(...) before using empirical EP stats."

        row, col = self.graph_cache.edge_index_clean_dir[0], self.graph_cache.edge_index_clean_dir[1]
        if self.emp_clean_edge_raw.numel() != row.numel() or self.emp_clean_edge_sq_raw.numel() != row.numel():
            return torch.zeros_like(m_sq)

        dtype = m_sq.dtype
        device = m_sq.device
        deg = self.graph_cache.deg_clean.to(device=device, dtype=dtype)
        d_hat = (deg + 1.0).clamp_min(1.0)
        row_d = row.to(device)
        col_d = col.to(device)

        e = (self.emp_clean_edge_scale * self.emp_clean_edge_raw).to(device=device, dtype=dtype)
        e2 = (self.emp_clean_edge_sq_scale * self.emp_clean_edge_sq_raw).to(device=device, dtype=dtype)
        var_g = (e2 - e * e).clamp_min(0.0)
        alpha_sq = 1.0 / (d_hat[row_d] * d_hat[col_d])
        factor = alpha_sq * (var_g / (e * e + float(eps)))
        neigh_var = _scatter_add_rows(factor.unsqueeze(1) * m_sq[row_d], col_d, m_sq.size(0))

        if self.emp_nonedge_raw.numel() == 0:
            return neigh_var

        row_a = self.emp_nonedge_row.to(device)
        col_a = self.emp_nonedge_col.to(device)
        e_a = (self.emp_nonedge_scale * self.emp_nonedge_raw).to(device=device, dtype=dtype)
        e2_a = (self.emp_nonedge_sq_scale * self.emp_nonedge_sq_raw).to(device=device, dtype=dtype)
        var_a = (e2_a - e_a * e_a).clamp_min(0.0)
        add_var = _scatter_add_rows(var_a.unsqueeze(1) * m_sq[row_a], col_a, m_sq.size(0))
        return neigh_var + add_var

    def _emp_update_weights(self) -> tuple[float, float]:
        if int(self.emp_num_updates.item()) <= 0:
            return 0.0, 1.0
        if self.ep_emp_average_mode == "ema":
            beta = float(self.ep_emp_beta)
            if beta <= 0.0:
                return 1.0, 0.0
            if beta >= 1.0:
                return 0.0, 1.0
            return 1.0 - beta, beta

        t_prev = int(self.emp_num_updates.item())
        t_new = t_prev + 1
        return float(t_prev) / float(t_new), 1.0 / float(t_new)

    def reset_ep_stats(self, p: float, q: float) -> None:
        assert self.graph_cache is not None, "Call model.set_graph(...) before reset_ep_stats(...)."
        row, col = self.graph_cache.edge_index_clean_dir[0], self.graph_cache.edge_index_clean_dir[1]

        self.emp_clean_edge_raw = torch.zeros(row.numel(), dtype=torch.float32, device=row.device)
        self.emp_clean_edge_scale.fill_(1.0)
        self.emp_clean_edge_sq_raw = torch.zeros_like(self.emp_clean_edge_raw)
        self.emp_clean_edge_sq_scale.fill_(1.0)

        self.emp_self_raw = torch.zeros(self.graph_cache.num_nodes, dtype=torch.float32, device=row.device)
        self.emp_self_scale.fill_(1.0)
        self.emp_self_sq_raw = torch.zeros_like(self.emp_self_raw)
        self.emp_self_sq_scale.fill_(1.0)

        self._reset_emp_nonedge_sparse()
        self.emp_num_updates.zero_()

    def update_ep_stats(
        self,
        *,
        edge_index_keep: Optional[torch.Tensor],
        edge_index_add: Optional[torch.Tensor],
        p: float,
        q: float,
    ) -> None:
        if self.ep_expectation_mode != "empirical_ema":
            return
        assert self.graph_cache is not None, "Call model.set_graph(...) before update_ep_stats(...)."

        old_mult, new_mult = self._emp_update_weights()
        if new_mult <= 0.0:
            return

        device = self.graph_cache.deg_clean.device
        row_k, col_k, gamma_k, row_a, col_a, gamma_a, gamma_self = self._gcn_sample_coeffs(
            edge_index_keep=edge_index_keep,
            edge_index_add=edge_index_add,
            device=device,
            dtype=torch.float32,
        )

        if old_mult <= 1e-12:
            self.emp_clean_edge_scale.fill_(1.0)
            self.emp_clean_edge_sq_scale.fill_(1.0)
            if self.emp_clean_edge_raw.numel():
                self.emp_clean_edge_raw.zero_()
            if self.emp_clean_edge_sq_raw.numel():
                self.emp_clean_edge_sq_raw.zero_()
            if row_k.numel():
                keys = _directed_edge_keys(row_k.detach().cpu(), col_k.detach().cpu(), self.graph_cache.num_nodes).tolist()
                idx = [self._clean_edge_lookup[int(key)] for key in keys]
                idx_t = torch.tensor(idx, dtype=torch.long, device=self.emp_clean_edge_raw.device)
                gamma_k_detached = gamma_k.detach().to(self.emp_clean_edge_raw.device, dtype=torch.float32)
                self.emp_clean_edge_raw.index_copy_(0, idx_t, gamma_k_detached)
                self.emp_clean_edge_sq_raw.index_copy_(0, idx_t.to(self.emp_clean_edge_sq_raw.device), gamma_k_detached.square().to(self.emp_clean_edge_sq_raw.device))

            self.emp_self_scale.fill_(1.0)
            self.emp_self_sq_scale.fill_(1.0)
            gamma_self_detached = gamma_self.detach().to(self.emp_self_raw.device, dtype=torch.float32)
            self.emp_self_raw = gamma_self_detached
            self.emp_self_sq_raw = gamma_self_detached.square().to(self.emp_self_sq_raw.device)

            self._reset_emp_nonedge_sparse()
            if row_a.numel():
                self.emp_nonedge_row = row_a.detach().to(self.emp_nonedge_row.device, dtype=torch.long)
                self.emp_nonedge_col = col_a.detach().to(self.emp_nonedge_col.device, dtype=torch.long)
                gamma_a_detached = gamma_a.detach().to(self.emp_nonedge_raw.device, dtype=torch.float32)
                self.emp_nonedge_raw = gamma_a_detached
                self.emp_nonedge_scale.fill_(1.0)
                self.emp_nonedge_sq_raw = gamma_a_detached.square().to(self.emp_nonedge_sq_raw.device)
                self.emp_nonedge_sq_scale.fill_(1.0)
                self._rebuild_emp_nonedge_lookup()
            self.emp_num_updates.add_(1)
            return

        self.emp_clean_edge_scale.mul_(old_mult)
        self.emp_clean_edge_sq_scale.mul_(old_mult)
        if row_k.numel():
            keys = _directed_edge_keys(row_k.detach().cpu(), col_k.detach().cpu(), self.graph_cache.num_nodes).tolist()
            idx = [self._clean_edge_lookup[int(key)] for key in keys]
            idx_t = torch.tensor(idx, dtype=torch.long, device=self.emp_clean_edge_raw.device)
            gamma_k_detached = gamma_k.detach().to(self.emp_clean_edge_raw.device, dtype=torch.float32)
            upd_t = (new_mult * gamma_k_detached) / self.emp_clean_edge_scale
            self.emp_clean_edge_raw.index_add_(0, idx_t, upd_t)
            sq_upd_t = (new_mult * gamma_k_detached.square().to(self.emp_clean_edge_sq_raw.device)) / self.emp_clean_edge_sq_scale
            self.emp_clean_edge_sq_raw.index_add_(0, idx_t.to(self.emp_clean_edge_sq_raw.device), sq_upd_t)

        self.emp_self_scale.mul_(old_mult)
        self.emp_self_sq_scale.mul_(old_mult)
        gamma_self_detached = gamma_self.detach().to(self.emp_self_raw.device, dtype=torch.float32)
        upd_self = (new_mult * gamma_self_detached) / self.emp_self_scale
        if self.emp_self_raw.numel() != upd_self.numel():
            self.emp_self_raw = torch.zeros_like(upd_self)
        self.emp_self_raw.add_(upd_self)
        sq_upd_self = (new_mult * gamma_self_detached.square().to(self.emp_self_sq_raw.device)) / self.emp_self_sq_scale
        if self.emp_self_sq_raw.numel() != sq_upd_self.numel():
            self.emp_self_sq_raw = torch.zeros_like(sq_upd_self)
        self.emp_self_sq_raw.add_(sq_upd_self)

        self.emp_nonedge_scale.mul_(old_mult)
        self.emp_nonedge_sq_scale.mul_(old_mult)
        self._append_nonedge_raw(row_a=row_a, col_a=col_a, gamma_a=gamma_a, new_mult=new_mult)
        self.emp_num_updates.add_(1)

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

            # RADE-OFS inference correction.
            if use_ofs and (not self.training) and (q > 0.0):
                if self.ep_expectation_mode == "empirical_ema":
                    # Exact sparse empirical expectation over observed clean non-edges.
                    out = out + self._apply_emp_nonedge_expectation(x)
                else:
                    # Analytic decoupled Case-B approximation.
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
        use_empirical_ep = self.ep_expectation_mode == "empirical_ema"
        has_empirical_ep = use_empirical_ep and self.emp_num_updates.item() > 0

        if has_empirical_ep:
            E_Gamma_k = self._get_clean_edge_emp(row_k, col_k, dtype=x.dtype, device=x.device)
            missing = E_Gamma_k <= 0.0
            if bool(missing.any()):
                E_Gamma_k[missing] = Gamma_k.detach().to(dtype=x.dtype, device=x.device)[missing]
        elif use_empirical_ep:
            E_Gamma_k = Gamma_k.detach().to(dtype=x.dtype, device=x.device)
        else:
            tA = _delta_E_inv_sqrt_caseA(deg_clean, comp_size, p=p, q=q).to(x.device, dtype=x.dtype)
            E_Gamma_k = (1.0 - p) * tA[row_k] * tA[col_k]

        corr_k = alpha_k / E_Gamma_k.clamp(min=self.ep_emp_eps)
        out_k = _scatter_add_rows((Gamma_k * corr_k).unsqueeze(1) * x[row_k], col_k, N)

        # Sampled non-edge contributions.
        if row_a.numel():
            sum_a = _scatter_add_rows(Gamma_a.unsqueeze(1) * x[row_a], col_a, N)

            if use_ofs:
                # RADE-OFS training:
                # keep non-edge contributions uncentered so their effect is carried to inference.
                out_a = sum_a
            else:
                if use_empirical_ep:
                    # Subtract the exact sparse empirical expectation of non-edge messages.
                    out_a = sum_a - self._apply_emp_nonedge_expectation(x)
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
            if has_empirical_ep:
                E_Gamma_self = (self.emp_self_scale * self.emp_self_raw).to(x.device, dtype=x.dtype)
            elif use_empirical_ep:
                E_Gamma_self = Gamma_self.detach().to(dtype=x.dtype, device=x.device)
            else:
                E_Gamma_self = _delta_E_inv_self(deg_clean, comp_size, p=p, q=q).to(x.device, dtype=x.dtype)
            corr_self = alpha_self / E_Gamma_self.clamp(min=self.ep_emp_eps)
            out_self = (Gamma_self * corr_self).unsqueeze(1) * x
        else:
            out_self = Gamma_self.unsqueeze(1) * x

        out = out_k + out_a + out_self
        if self.bias is not None:
            out = out + self.bias
        return out


class RADEGATConv(nn.Module):
    """
    Full-batch GAT attention aggregation with RADE expectation preservation.

    This implementation follows the attention-normalization delta approximation:
        E[alpha_ij]  ~= pi_ij w_ij (D_ij^-1 + sigma_ij^2 D_ij^-3)
        E[alpha_ij^2] ~= pi_ij w_ij^2 (D_ij^-2 + 3 sigma_ij^2 D_ij^-4)
    where D_ij = w_ij + mu_ij and mu/sigma exclude j from i's random
    attention denominator. Self-loops are excluded, matching the provided
    k != i formulation.

    moment_mode='exact' computes the denominator and non-edge sums with the
    chunked all-pairs path. moment_mode='sampled' keeps clean-neighbor
    contributions exact and estimates only dense clean-complement terms by
    deterministic Monte Carlo samples. The scalable sampler draws
    non-neighbors with replacement; use 'exact' as the reference path.
    moment_mode='bernoulli' estimates each candidate-pair denominator by
    sampling Bernoulli perturbed neighborhoods under the current p,q.

    The forward operator supports multiple heads. The GradNorm variance proxy is
    intentionally restricted to heads=1 until a multi-head logit variance
    approximation is derived and validated.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        bias: bool = True,
        rade_variant: str = "rade-of",
        ep_correction: bool = True,
        ep_expectation_mode: str = "analytic",
        ep_emp_eps: float = 1e-12,
        moment_chunk_size: int = 1024,
        moment_mode: str = "exact",
        moment_samples: int = 256,
        nonedge_samples: int = 256,
        bernoulli_samples: int = 64,
        moment_seed: int = 0,
        ep_corr_clip: float = 0.0,
    ):
        super().__init__()
        self.heads = int(heads)
        if self.heads <= 0:
            raise ValueError(f"heads must be positive. Got {heads}")
        self.concat = bool(concat)
        if self.concat:
            if int(out_channels) % self.heads != 0:
                raise ValueError(
                    "For concat=True, out_channels must be divisible by heads. "
                    f"Got out_channels={out_channels}, heads={heads}."
                )
            self.out_per_head = int(out_channels) // self.heads
            self.out_channels = int(out_channels)
        else:
            self.out_per_head = int(out_channels)
            self.out_channels = int(out_channels)

        self.lin = nn.Linear(in_channels, self.heads * self.out_per_head, bias=False)
        self.att_src = nn.Parameter(torch.empty(self.heads, self.out_per_head))
        self.att_dst = nn.Parameter(torch.empty(self.heads, self.out_per_head))
        self.bias = nn.Parameter(torch.zeros(self.out_channels)) if bias else None
        self.negative_slope = float(negative_slope)
        self.rade_variant = _validate_rade_variant(rade_variant)
        self.ep_correction = bool(ep_correction)
        self.ep_expectation_mode = _validate_ep_expectation_mode(ep_expectation_mode)
        if self.ep_expectation_mode != "analytic":
            raise ValueError("RADEGATConv currently supports only ep_expectation_mode='analytic'.")
        self.ep_emp_eps = float(ep_emp_eps)
        self.moment_chunk_size = max(1, int(moment_chunk_size))
        self.moment_mode = str(moment_mode).lower().strip()
        if self.moment_mode not in {"exact", "sampled", "bernoulli"}:
            raise ValueError(
                f"moment_mode must be 'exact', 'sampled', or 'bernoulli'. Got {moment_mode}."
            )
        self.moment_samples = max(1, int(moment_samples))
        self.nonedge_samples = max(1, int(nonedge_samples))
        self.bernoulli_samples = max(1, int(bernoulli_samples))
        self.moment_seed = int(moment_seed)
        self.ep_corr_clip = max(0.0, float(ep_corr_clip))

        self.graph_cache: Optional[GraphCache] = None
        self._last_h_for_moments: Optional[torch.Tensor] = None

    def set_graph(self, graph_cache: GraphCache) -> None:
        self.graph_cache = graph_cache

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x).view(x.size(0), self.heads, self.out_per_head)

    def _finalize_heads(self, out: torch.Tensor) -> torch.Tensor:
        if self.concat:
            out = out.reshape(out.size(0), self.heads * self.out_per_head)
        else:
            out = out.mean(dim=1)
        if self.bias is not None:
            out = out + self.bias
        return out

    def _node_attention_scores(
        self,
        h: torch.Tensor,
        *,
        detach_params: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Standard GAT decomposes the pair logit into source and destination
        # terms: a_src^T h_j + a_dst^T h_i. Precomputing both lets the moment
        # code build all candidate weights w_ij = exp(leaky_relu(...)) cheaply.
        att_src = self.att_src.detach() if detach_params else self.att_src
        att_dst = self.att_dst.detach() if detach_params else self.att_dst
        src = (h * att_src.unsqueeze(0)).sum(dim=-1)
        dst = (h * att_dst.unsqueeze(0)).sum(dim=-1)
        return src, dst

    def _edge_logits(
        self,
        h: torch.Tensor,
        row: torch.Tensor,
        col: torch.Tensor,
        *,
        detach_params: bool = False,
    ) -> torch.Tensor:
        score_src, score_dst = self._node_attention_scores(h, detach_params=detach_params)
        return torch.nn.functional.leaky_relu(
            score_src[row] + score_dst[col],
            negative_slope=self.negative_slope,
        )

    def _edge_softmax(self, logits: torch.Tensor, col: torch.Tensor, num_nodes: int) -> torch.Tensor:
        if logits.numel() == 0:
            return logits
        return softmax(logits, col, num_nodes=num_nodes)

    def _attention_logit_shift(
        self,
        score_src: torch.Tensor,
        score_dst: torch.Tensor,
    ) -> torch.Tensor:
        N = score_src.size(0)
        if N <= 1:
            return torch.zeros((N, self.heads), dtype=score_src.dtype, device=score_src.device)

        shift = torch.full(
            (N, self.heads),
            -torch.inf,
            dtype=score_src.dtype,
            device=score_src.device,
        )
        for start, end in self._chunk_ranges(N):
            src_idx = torch.arange(start, end, device=score_src.device, dtype=torch.long)
            logits = torch.nn.functional.leaky_relu(
                score_src[src_idx].unsqueeze(0) + score_dst.unsqueeze(1),
                negative_slope=self.negative_slope,
            )
            logits[src_idx, src_idx - start] = -torch.inf
            shift = torch.maximum(shift, logits.max(dim=1).values)

        return torch.where(torch.isfinite(shift), shift, torch.zeros_like(shift)).detach()

    def _scaled_attention_weights(
        self,
        logits: torch.Tensor,
        logit_shift: torch.Tensor,
        col: torch.Tensor,
    ) -> torch.Tensor:
        return (logits - logit_shift[col]).exp()

    def _scaled_chunk_attention_weights(
        self,
        logits: torch.Tensor,
        logit_shift: torch.Tensor,
    ) -> torch.Tensor:
        return (logits - logit_shift.unsqueeze(1)).exp()

    def _clean_edge_alpha(
        self,
        h: torch.Tensor,
        row: torch.Tensor,
        col: torch.Tensor,
        *,
        detach_params: bool = False,
    ) -> torch.Tensor:
        logits = self._edge_logits(h, row, col, detach_params=detach_params)
        return self._edge_softmax(logits, col, h.size(0))

    def _aggregate(self, h: torch.Tensor, row: torch.Tensor, col: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        msg = alpha.unsqueeze(-1) * h[row]
        return _scatter_add_heads(msg, col, h.size(0))

    def _chunk_ranges(self, num_nodes: int):
        step = max(1, int(self.moment_chunk_size))
        for start in range(0, num_nodes, step):
            end = min(num_nodes, start + step)
            yield start, end

    def _clean_edge_keys_sorted(self, device: torch.device) -> torch.Tensor:
        assert self.graph_cache is not None, "Call model.set_graph(...) once after loading data."
        edge_row = self.graph_cache.edge_index_clean_dir[0].to(device)
        edge_col = self.graph_cache.edge_index_clean_dir[1].to(device)
        keys = edge_row * int(self.graph_cache.num_nodes) + edge_col
        return torch.sort(keys).values

    @staticmethod
    def _edge_membership(keys: torch.Tensor, edge_keys_sorted: torch.Tensor) -> torch.Tensor:
        if keys.numel() == 0 or edge_keys_sorted.numel() == 0:
            return torch.zeros_like(keys, dtype=torch.bool)
        pos = torch.searchsorted(edge_keys_sorted, keys)
        valid = pos < edge_keys_sorted.numel()
        pos_clamped = pos.clamp(max=max(int(edge_keys_sorted.numel()) - 1, 0))
        return valid & (edge_keys_sorted[pos_clamped] == keys)

    def _sample_nonedge_sources_with_replacement(
        self,
        num_nodes: int,
        samples: int,
        device: torch.device,
        *,
        seed_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Draw per-target clean non-neighbor sources for sampled GAT moments.

        Returns:
            src_out: [N, S] sampled source nodes j for each target i.
            valid_out: [N, S] mask, because targets with |C(i)| < S have fewer
                usable clean non-neighbors than the shared sampled width.
            scale: [N] Horvitz-Thompson-style multiplier |C(i)| / sample_count_i
                used to lift sampled complement sums back to full-complement sums.

        The sampler rejects self-pairs and clean graph edges, so every accepted
        pair (j, i) is a clean non-edge candidate. A fixed CPU generator keeps
        the sampled estimator deterministic for a given moment_seed.
        """
        assert self.graph_cache is not None, "Call model.set_graph(...) once after loading data."
        N: int = int(num_nodes)
        comp: torch.Tensor = self.graph_cache.comp_size.to(device=device, dtype=torch.long)
        max_comp: int = int(comp.max().item()) if comp.numel() > 0 else 0
        sample_cols: int = min(max(1, int(samples)), max_comp)
        if N <= 1 or max_comp <= 0:
            src_empty = torch.empty((N, 0), dtype=torch.long, device=device)
            valid_empty = torch.empty((N, 0), dtype=torch.bool, device=device)
            scale_empty = torch.zeros((N,), dtype=torch.float32, device=device)
            return src_empty, valid_empty, scale_empty

        target_counts: torch.Tensor = torch.clamp(comp, max=sample_cols)
        src_out: torch.Tensor = torch.zeros((N, sample_cols), dtype=torch.long, device=device)
        valid_out: torch.Tensor = torch.zeros((N, sample_cols), dtype=torch.bool, device=device)
        counts: torch.Tensor = torch.zeros((N,), dtype=torch.long, device=device)
        edge_keys_sorted: torch.Tensor = self._clean_edge_keys_sorted(device)
        row_ids: torch.Tensor = torch.arange(N, device=device, dtype=torch.long).unsqueeze(1)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(self.moment_seed) + int(seed_offset))

        # Draw candidates in increasingly large batches. This keeps dense
        # all-pairs materialization out of the sampled path while still filling
        # high-degree targets that reject many candidate neighbors.
        draw: int = max(sample_cols * 2, 64)
        max_attempts: int = 32
        for attempt in range(max_attempts):
            active = counts < target_counts
            if not bool(active.any()):
                break
            candidate_count = N - 1
            raw = torch.randint(
                low=0,
                high=candidate_count,
                size=(N, draw),
                generator=gen,
                device=torch.device("cpu"),
                dtype=torch.long,
            ).to(device)
            target = torch.arange(N, device=device, dtype=torch.long).unsqueeze(1)
            cand = raw + (raw >= target).to(torch.long)
            keys = cand * N + target.expand_as(cand)
            accepted = (~self._edge_membership(keys.reshape(-1), edge_keys_sorted).view_as(cand)) & active.unsqueeze(1)
            rank = accepted.to(torch.long).cumsum(dim=1) - 1 + counts.unsqueeze(1)
            keep = accepted & (rank < target_counts.unsqueeze(1)) & (rank < sample_cols)
            if bool(keep.any()):
                # rank gives the destination column inside src_out for each
                # accepted source; counts tracks how many samples each target
                # still needs after this candidate batch.
                dst_pos = row_ids.expand_as(cand)[keep]
                rank_pos = rank[keep]
                src_out[dst_pos, rank_pos] = cand[keep]
                valid_out[dst_pos, rank_pos] = True
                counts = counts + keep.sum(dim=1).to(torch.long)
            draw = min(max(draw * 2, 64), max(sample_cols * 16, 1024))

        scale: torch.Tensor = torch.zeros((N,), dtype=torch.float32, device=device)
        nonzero = counts > 0
        scale[nonzero] = comp.to(torch.float32)[nonzero] / counts.to(torch.float32)[nonzero]
        return src_out, valid_out, scale

    def _attention_totals_exact(
        self,
        h: torch.Tensor,
        p: torch.Tensor | float,
        q: torch.Tensor | float,
        *,
        detach_params: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.graph_cache is not None, "Call model.set_graph(...) once after loading data."
        N = h.size(0)
        dtype = h.dtype
        device = h.device
        p_t = torch.as_tensor(p, dtype=dtype, device=device)
        q_t = torch.as_tensor(q, dtype=dtype, device=device)
        keep_t = 1.0 - p_t

        score_src, score_dst = self._node_attention_scores(h, detach_params=detach_params)
        logit_shift = self._attention_logit_shift(score_src, score_dst)
        total_mu = torch.zeros((N, self.heads), dtype=dtype, device=device)
        total_sigma = torch.zeros_like(total_mu)

        edge_row = self.graph_cache.edge_index_clean_dir[0].to(device)
        edge_col = self.graph_cache.edge_index_clean_dir[1].to(device)

        for start, end in self._chunk_ranges(N):
            # This chunk contains candidate sources j. For every target i and
            # source j in the chunk, compute w_ij and the Bernoulli probability
            # that A'_ij is present under the current p,q.
            src_idx = torch.arange(start, end, device=device)
            logits = torch.nn.functional.leaky_relu(
                score_src[src_idx].unsqueeze(0) + score_dst.unsqueeze(1),
                negative_slope=self.negative_slope,
            )
            wij = self._scaled_chunk_attention_weights(logits, logit_shift)

            pi = torch.full((N, end - start), 0.0, dtype=dtype, device=device) + q_t
            in_chunk = (edge_row >= start) & (edge_row < end)
            if bool(in_chunk.any()):
                # Clean neighbors are retained with probability 1-p; clean
                # non-neighbors keep the default add probability q.
                pi[edge_col[in_chunk], edge_row[in_chunk] - start] = keep_t

            diag_rows = src_idx[(src_idx >= 0) & (src_idx < N)]
            if diag_rows.numel() > 0:
                # GAT moments here intentionally exclude self-loops.
                pi[diag_rows, diag_rows - start] = 0.0

            pi_h = pi.unsqueeze(-1)
            # total_mu/total_sigma are moments of the full random denominator
            # sum_k A'_ik w_ik. Candidate-specific moments subtract j later.
            total_mu = total_mu + (pi_h * wij).sum(dim=1)
            total_sigma = total_sigma + (pi_h * (1.0 - pi_h) * wij.square()).sum(dim=1)

        return total_mu, total_sigma, score_src, score_dst, logit_shift

    def _attention_totals_sampled(
        self,
        h: torch.Tensor,
        p: torch.Tensor | float,
        q: torch.Tensor | float,
        *,
        detach_params: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Estimate per-target random denominator moments for sampled GAT.

        total_mu[i] and total_sigma[i] approximate the mean and variance of
        the perturbed attention denominator for target i:

            sum_k A'_ik w_ik

        Clean-neighbor terms are summed exactly because they are sparse. Dense
        clean-complement terms are estimated from sampled non-neighbors and
        scaled by |C(i)| / sample_count_i. Later, pair-specific code subtracts
        the numerator candidate j from these totals to get S_ij.
        """
        assert self.graph_cache is not None, "Call model.set_graph(...) once after loading data."
        N: int = h.size(0)
        if N <= 1:
            score_src, score_dst = self._node_attention_scores(h, detach_params=detach_params)
            logit_shift = self._attention_logit_shift(score_src, score_dst)
            zeros = torch.zeros((N, self.heads), dtype=h.dtype, device=h.device)
            return zeros, zeros, score_src, score_dst, logit_shift
        max_comp: int = int(self.graph_cache.comp_size.max().item()) if self.graph_cache.comp_size.numel() > 0 else 0
        if self.moment_samples >= max_comp:
            # If the requested sample budget covers the largest complement,
            # the exact chunked path is both clearer and no more expensive.
            return self._attention_totals_exact(h, p=p, q=q, detach_params=detach_params)

        dtype: torch.dtype = h.dtype
        device: torch.device = h.device
        p_t: torch.Tensor = torch.as_tensor(p, dtype=dtype, device=device)
        q_t: torch.Tensor = torch.as_tensor(q, dtype=dtype, device=device)
        keep_t: torch.Tensor = 1.0 - p_t

        score_src, score_dst = self._node_attention_scores(h, detach_params=detach_params)
        logit_shift = self._attention_logit_shift(score_src, score_dst)

        edge_row = self.graph_cache.edge_index_clean_dir[0].to(device)
        edge_col = self.graph_cache.edge_index_clean_dir[1].to(device)
        # In sampled-index mode, clean-edge denominator contributions are still
        # exact because they are sparse and cheap to aggregate.
        edge_logits = torch.nn.functional.leaky_relu(
            score_src[edge_row] + score_dst[edge_col],
            negative_slope=self.negative_slope,
        )
        edge_w = self._scaled_attention_weights(edge_logits, logit_shift, edge_col)
        total_mu = _scatter_add_rows(keep_t.expand_as(edge_w) * edge_w, edge_col, N)
        total_sigma = _scatter_add_rows((keep_t * p_t).expand_as(edge_w) * edge_w.square(), edge_col, N)

        # Sample only clean non-neighbors. valid masks padded sample slots, and
        # scale lifts each target's sample sum back to its full complement size.
        src_idx, valid, scale = self._sample_nonedge_sources_with_replacement(
            N,
            self.moment_samples,
            device,
            seed_offset=17,
        )
        if src_idx.numel() == 0:
            return total_mu, total_sigma, score_src, score_dst, logit_shift
        logits = torch.nn.functional.leaky_relu(
            score_src[src_idx] + score_dst.unsqueeze(1),
            negative_slope=self.negative_slope,
        )
        wij = self._scaled_chunk_attention_weights(logits, logit_shift)
        pi = (torch.zeros_like(src_idx, dtype=dtype) + q_t) * valid.to(dtype)
        pi_h = pi.unsqueeze(-1)
        scale_h = scale.to(dtype).view(N, 1)
        # Non-neighbor terms are estimated from sampled clean complement
        # sources and scaled back to the full complement size.
        total_mu = total_mu + scale_h * (pi_h * wij).sum(dim=1)
        total_sigma = total_sigma + scale_h * (pi_h * (1.0 - pi_h) * wij.square()).sum(dim=1)
        return total_mu, total_sigma, score_src, score_dst, logit_shift

    def _attention_totals(
        self,
        h: torch.Tensor,
        p: torch.Tensor | float,
        q: torch.Tensor | float,
        *,
        detach_params: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.moment_mode == "bernoulli":
            # Bernoulli mode does not use global analytic denominator totals.
            # It estimates S_ij directly for each candidate pair, so only the
            # reusable attention node scores are needed here.
            score_src, score_dst = self._node_attention_scores(h, detach_params=detach_params)
            logit_shift = self._attention_logit_shift(score_src, score_dst)
            zeros = torch.zeros((h.size(0), self.heads), dtype=h.dtype, device=h.device)
            return zeros, zeros, score_src, score_dst, logit_shift
        if self.moment_mode == "sampled":
            return self._attention_totals_sampled(h, p=p, q=q, detach_params=detach_params)
        return self._attention_totals_exact(h, p=p, q=q, detach_params=detach_params)

    def _pair_moments_bernoulli(
        self,
        *,
        row: torch.Tensor,
        col: torch.Tensor,
        pi: torch.Tensor | float,
        p: torch.Tensor | float,
        q: torch.Tensor | float,
        score_src: torch.Tensor,
        score_dst: torch.Tensor,
        logit_shift: torch.Tensor | None = None,
        seed_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Estimate candidate-pair GAT attention moments by direct Bernoulli masks.

        For each candidate source row[e] = j and target col[e] = i, we condition
        on A'_ij = 1 and sample only the competing denominator terms
        S_ij = sum_{k != i,j} A'_ik w_ik. Clean neighbors k are kept with
        probability 1-p, clean non-neighbors are added with probability q.
        """
        assert self.graph_cache is not None, "Call model.set_graph(...) once after loading data."
        if row.numel() == 0:
            empty = torch.empty((0, self.heads), dtype=score_src.dtype, device=score_src.device)
            return empty, empty, empty

        N = int(self.graph_cache.num_nodes)
        dtype = score_src.dtype
        device = score_src.device
        p_t = torch.as_tensor(p, dtype=dtype, device=device)
        q_t = torch.as_tensor(q, dtype=dtype, device=device)
        keep_t = 1.0 - p_t
        pi_t = torch.as_tensor(pi, dtype=dtype, device=device)
        edge_keys_sorted = self._clean_edge_keys_sorted(device)
        all_sources = torch.arange(N, device=device, dtype=torch.long)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(self.moment_seed) + int(seed_offset))

        out_e = torch.empty((row.numel(), self.heads), dtype=dtype, device=device)
        out_e2 = torch.empty_like(out_e)
        out_var = torch.empty_like(out_e)
        pair_chunk = max(1, min(int(self.moment_chunk_size), 512))
        R = max(1, int(self.bernoulli_samples))
        # The random masks are fixed samples for the current forward/objective
        # evaluation. Detaching probabilities prevents autograd from attempting
        # to differentiate through the discrete sampling step.
        q_det = q_t.detach()
        keep_det = keep_t.detach()
        if logit_shift is None:
            logit_shift = self._attention_logit_shift(score_src, score_dst)

        for pair_start in range(0, row.numel(), pair_chunk):
            # Work on a manageable block of candidate pairs. samples[r, e, h]
            # accumulates one Monte Carlo draw of S_ij for pair e and head h.
            pair_end = min(row.numel(), pair_start + pair_chunk)
            row_c = row[pair_start:pair_end].to(device=device, dtype=torch.long)
            col_c = col[pair_start:pair_end].to(device=device, dtype=torch.long)
            C = row_c.numel()
            samples = torch.zeros((R, C, self.heads), dtype=dtype, device=device)

            for src_start, src_end in self._chunk_ranges(N):
                # This source chunk enumerates competing sources k. For every
                # candidate pair (i,j), we compute w_ik for these k values.
                src_idx = all_sources[src_start:src_end]
                logits = torch.nn.functional.leaky_relu(
                    score_src[src_idx].unsqueeze(0) + score_dst[col_c].unsqueeze(1),
                    negative_slope=self.negative_slope,
                )
                wij = self._scaled_chunk_attention_weights(logits, logit_shift[col_c])

                keys = src_idx.unsqueeze(0) * N + col_c.unsqueeze(1)
                clean = self._edge_membership(keys.reshape(-1), edge_keys_sorted).view(C, -1)
                # Choose the Bernoulli rate for A'_ik from the clean graph:
                # 1-p for clean neighbors of i, q for clean non-neighbors.
                probs = torch.where(
                    clean,
                    keep_det.expand(clean.shape),
                    q_det.expand(clean.shape),
                )
                # Exclude k=i self-loops and k=j because A'_ij is the
                # conditioned numerator term, not part of S_ij.
                valid = (src_idx.unsqueeze(0) != col_c.unsqueeze(1)) & (
                    src_idx.unsqueeze(0) != row_c.unsqueeze(1)
                )
                probs = probs * valid.to(dtype)

                probs_cpu = probs.detach().to(device="cpu", dtype=torch.float32)
                for r in range(R):
                    uniforms = torch.rand(
                        probs_cpu.shape,
                        generator=gen,
                        device=torch.device("cpu"),
                        dtype=torch.float32,
                    ).to(device=device)
                    mask = uniforms < probs_cpu.to(device=device)
                    samples[r] = samples[r] + (mask.to(dtype).unsqueeze(-1) * wij).sum(dim=1)

            # Estimate mu_hat_ij and sigma2_hat_ij from R sampled denominators.
            # unbiased=False is deliberate: R=1 should produce variance 0, and
            # the delta approximation uses the Monte Carlo population estimate.
            mu = samples.mean(dim=0)
            sigma2 = samples.var(dim=0, unbiased=False).clamp_min(0.0)
            logits_ij = torch.nn.functional.leaky_relu(
                score_src[row_c] + score_dst[col_c],
                negative_slope=self.negative_slope,
            )
            wij = self._scaled_attention_weights(logits_ij, logit_shift, col_c)
            if pi_t.ndim == 0:
                pi_h = pi_t.expand_as(wij)
            else:
                pi_h = pi_t[pair_start:pair_end].unsqueeze(-1).expand_as(wij)
            denom = (wij + mu).clamp_min(self.ep_emp_eps)
            # Reuse the same second-order delta formulas as exact/sampled mode,
            # now with Bernoulli-mask estimates for the competing denominator.
            e_alpha = pi_h * wij * (denom.reciprocal() + sigma2 * denom.pow(-3.0))
            e2_alpha = pi_h * wij.square() * (denom.pow(-2.0) + 3.0 * sigma2 * denom.pow(-4.0))
            var_alpha = (e2_alpha - e_alpha.square()).clamp_min(0.0)
            out_e[pair_start:pair_end] = e_alpha
            out_e2[pair_start:pair_end] = e2_alpha
            out_var[pair_start:pair_end] = var_alpha

        return out_e, out_e2, out_var

    def _pair_moments_from_totals(
        self,
        *,
        row: torch.Tensor,
        col: torch.Tensor,
        pi: torch.Tensor | float,
        p: torch.Tensor | float | None = None,
        q: torch.Tensor | float | None = None,
        total_mu: torch.Tensor,
        total_sigma: torch.Tensor,
        score_src: torch.Tensor,
        score_dst: torch.Tensor,
        logit_shift: torch.Tensor | None = None,
        seed_offset: int = 53,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if row.numel() == 0:
            empty = torch.empty((0, self.heads), dtype=score_src.dtype, device=score_src.device)
            return empty, empty, empty
        if self.moment_mode == "bernoulli":
            # Bernoulli mode bypasses analytic totals and estimates the
            # candidate denominator directly under the current rates.
            if p is None or q is None:
                raise ValueError("Bernoulli GAT moments require p and q.")
            return self._pair_moments_bernoulli(
                row=row,
                col=col,
                pi=pi,
                p=p,
                q=q,
                score_src=score_src,
                score_dst=score_dst,
                logit_shift=logit_shift,
                seed_offset=seed_offset,
            )

        pi_t = torch.as_tensor(pi, dtype=score_src.dtype, device=score_src.device)
        if logit_shift is None:
            logit_shift = self._attention_logit_shift(score_src, score_dst)
        logits = torch.nn.functional.leaky_relu(
            score_src[row] + score_dst[col],
            negative_slope=self.negative_slope,
        )
        wij = self._scaled_attention_weights(logits, logit_shift, col)
        pi_h = pi_t.expand_as(wij) if pi_t.ndim == 0 else pi_t.unsqueeze(-1).expand_as(wij)
        # For exact/sampled modes, total_mu/total_sigma include j. Subtract the
        # candidate contribution so mu/sigma describe S_ij = sum_{k != i,j}.
        mu = total_mu[col] - pi_h * wij
        mu = mu.clamp_min(0.0)
        sigma = total_sigma[col] - pi_h * (1.0 - pi_h) * wij.square().clamp_min(0.0)
        sigma = sigma.clamp_min(0.0)
        denom = (wij + mu).clamp(min=self.ep_emp_eps)
        e_alpha = pi_h * wij * (denom.reciprocal() + sigma * denom.pow(-3.0))
        e2_alpha = pi_h * wij.square() * (denom.pow(-2.0) + 3.0 * sigma * denom.pow(-4.0))
        var_alpha = (e2_alpha - e_alpha.square()).clamp_min(0.0)
        return e_alpha, e2_alpha, var_alpha

    def _nonedge_expectation_sum_exact(
        self,
        h: torch.Tensor,
        p: float,
        q: float,
        *,
        centered: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.graph_cache is not None, "Call model.set_graph(...) once after loading data."
        N = h.size(0)
        total_mu, total_sigma, score_src, score_dst, logit_shift = self._attention_totals(h, p=p, q=q)
        dtype = h.dtype
        device = h.device
        num = torch.zeros((N, self.heads, self.out_per_head), dtype=dtype, device=device)
        den = torch.zeros((N, self.heads), dtype=dtype, device=device)

        edge_row = self.graph_cache.edge_index_clean_dir[0].to(device)
        edge_col = self.graph_cache.edge_index_clean_dir[1].to(device)

        for start, end in self._chunk_ranges(N):
            src_idx = torch.arange(start, end, device=device)
            logits = torch.nn.functional.leaky_relu(
                score_src[src_idx].unsqueeze(0) + score_dst.unsqueeze(1),
                negative_slope=self.negative_slope,
            )
            wij = self._scaled_chunk_attention_weights(logits, logit_shift)
            pi_h = wij.new_full(wij.shape, float(q))

            mu = total_mu.unsqueeze(1) - pi_h * wij
            mu = mu.clamp_min(0.0)
            sigma = total_sigma.unsqueeze(1) - pi_h * (1.0 - pi_h) * wij.square()
            sigma = sigma.clamp_min(0.0)
            denom = (wij + mu).clamp(min=self.ep_emp_eps)
            e_alpha = pi_h * wij * (denom.reciprocal() + sigma * denom.pow(-3.0))

            mask = torch.ones((N, end - start), dtype=torch.bool, device=device)
            in_chunk = (edge_row >= start) & (edge_row < end)
            if bool(in_chunk.any()):
                mask[edge_col[in_chunk], edge_row[in_chunk] - start] = False
            mask[src_idx, src_idx - start] = False
            e_alpha = e_alpha * mask.unsqueeze(-1).to(dtype)

            den = den + e_alpha.sum(dim=1)
            num = num + torch.einsum("ibh,bhd->ihd", e_alpha, h[src_idx])

        mean = num / den.clamp(min=self.ep_emp_eps).unsqueeze(-1)
        return (mean if centered else num), den

    def _nonedge_expectation_sum_sampled(
        self,
        h: torch.Tensor,
        p: float,
        q: float,
        *,
        centered: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate the expected added non-edge message for each target node.

        centered=False returns the full expected non-edge message sum used by
        RADE-OFS inference. centered=True returns the expected non-edge mean
        mu_i used by RADE-OF to center sampled added messages during training.
        """
        assert self.graph_cache is not None, "Call model.set_graph(...) once after loading data."
        N: int = h.size(0)
        if N <= 1:
            num = torch.zeros((N, self.heads, self.out_per_head), dtype=h.dtype, device=h.device)
            den = torch.zeros((N, self.heads), dtype=h.dtype, device=h.device)
            return (num if not centered else num), den
        max_comp: int = int(self.graph_cache.comp_size.max().item()) if self.graph_cache.comp_size.numel() > 0 else 0
        if self.nonedge_samples >= max_comp:
            return self._nonedge_expectation_sum_exact(h, p=p, q=q, centered=centered)

        total_mu, total_sigma, score_src, score_dst, logit_shift = self._attention_totals(h, p=p, q=q)
        dtype: torch.dtype = h.dtype
        device: torch.device = h.device
        src_idx, valid, scale = self._sample_nonedge_sources_with_replacement(
            N,
            self.nonedge_samples,
            device,
            seed_offset=31,
        )
        if src_idx.numel() == 0:
            num = torch.zeros((N, self.heads, self.out_per_head), dtype=dtype, device=device)
            den = torch.zeros((N, self.heads), dtype=dtype, device=device)
            return (num if centered else num), den

        logits = torch.nn.functional.leaky_relu(
            score_src[src_idx] + score_dst.unsqueeze(1),
            negative_slope=self.negative_slope,
        )
        wij = self._scaled_chunk_attention_weights(logits, logit_shift)
        pi_h = wij.new_full(wij.shape, float(q)) * valid.unsqueeze(-1).to(dtype)

        mu = total_mu.unsqueeze(1) - pi_h * wij
        mu = mu.clamp_min(0.0)
        sigma = total_sigma.unsqueeze(1) - pi_h * (1.0 - pi_h) * wij.square()
        sigma = sigma.clamp_min(0.0)
        denom = (wij + mu).clamp(min=self.ep_emp_eps)
        e_alpha = pi_h * wij * (denom.reciprocal() + sigma * denom.pow(-3.0))

        # num/den are Monte Carlo estimates of:
        #   sum_{j in C(i)} E[alpha'_ij] h_j
        #   sum_{j in C(i)} E[alpha'_ij]
        # after multiplying by each target's complement-size scale.
        num = torch.einsum("ish,ishd->ihd", e_alpha, h[src_idx])
        den = e_alpha.sum(dim=1)
        scale_h = scale.to(dtype).view(N, 1)
        num = scale_h.unsqueeze(-1) * num
        den = scale_h * den
        mean = num / den.clamp(min=self.ep_emp_eps).unsqueeze(-1)
        return (mean if centered else num), den

    def _nonedge_expectation_sum_bernoulli(
        self,
        h: torch.Tensor,
        p: torch.Tensor | float,
        q: torch.Tensor | float,
        *,
        centered: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.graph_cache is not None, "Call model.set_graph(...) once after loading data."
        N = h.size(0)
        dtype = h.dtype
        device = h.device
        num = torch.zeros((N, self.heads, self.out_per_head), dtype=dtype, device=device)
        den = torch.zeros((N, self.heads), dtype=dtype, device=device)
        if N <= 1:
            return (num if centered else num), den

        score_src, score_dst = self._node_attention_scores(h)
        edge_row = self.graph_cache.edge_index_clean_dir[0].to(device)
        edge_col = self.graph_cache.edge_index_clean_dir[1].to(device)
        targets = torch.arange(N, device=device, dtype=torch.long)
        q_t = torch.as_tensor(q, dtype=dtype, device=device)

        for start, end in self._chunk_ranges(N):
            # Build all clean non-edge candidate pairs (source j, target i) in
            # this source chunk. These are the possible added edges.
            src_idx = torch.arange(start, end, device=device, dtype=torch.long)
            mask = torch.ones((N, end - start), dtype=torch.bool, device=device)
            in_chunk = (edge_row >= start) & (edge_row < end)
            if bool(in_chunk.any()):
                mask[edge_col[in_chunk], edge_row[in_chunk] - start] = False
            mask[src_idx, src_idx - start] = False
            if not bool(mask.any()):
                continue

            row_grid = src_idx.unsqueeze(0).expand(N, -1)
            col_grid = targets.unsqueeze(1).expand(-1, end - start)
            row_flat = row_grid[mask]
            col_flat = col_grid[mask]
            # Each non-edge expectation also uses pair-conditioned Bernoulli
            # denominator samples with pi=q for the added edge itself.
            e_alpha, _, _ = self._pair_moments_bernoulli(
                row=row_flat,
                col=col_flat,
                pi=q_t,
                p=p,
                q=q,
                score_src=score_src,
                score_dst=score_dst,
                logit_shift=None,
                seed_offset=71 + start,
            )
            num = num + _scatter_add_heads(e_alpha.unsqueeze(-1) * h[row_flat], col_flat, N)
            den = den + _scatter_add_rows(e_alpha, col_flat, N)

        mean = num / den.clamp(min=self.ep_emp_eps).unsqueeze(-1)
        return (mean if centered else num), den

    def _nonedge_expectation_sum(
        self,
        h: torch.Tensor,
        p: float,
        q: float,
        *,
        centered: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.moment_mode == "bernoulli":
            return self._nonedge_expectation_sum_bernoulli(h, p=p, q=q, centered=centered)
        if self.moment_mode == "sampled":
            return self._nonedge_expectation_sum_sampled(h, p=p, q=q, centered=centered)
        return self._nonedge_expectation_sum_exact(h, p=p, q=q, centered=centered)

    def _nonedge_variance_sum_sampled(
        self,
        m: torch.Tensor,
        m_sq: torch.Tensor,
        *,
        total_mu: torch.Tensor,
        total_sigma: torch.Tensor,
        score_src: torch.Tensor,
        score_dst: torch.Tensor,
        logit_shift: torch.Tensor,
        q_t: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """
        Sampled add-side variance proxy for the GAT GradNorm objective.

        This mirrors _nonedge_expectation_sum_sampled, but accumulates
        Var(alpha'_ij) times either centered message squares (RADE-OF) or raw
        message squares (RADE-OFS). It is used only when the complement is too
        large for the exact all-pairs variance path.
        """
        assert self.graph_cache is not None, "Call model.set_graph(...) once after loading data."
        N: int = m.size(0)
        if N <= 1:
            return torch.zeros_like(m_sq)

        device: torch.device = m.device
        dtype: torch.dtype = m.dtype
        src_idx, valid, scale = self._sample_nonedge_sources_with_replacement(
            N,
            self.nonedge_samples,
            device,
            seed_offset=43,
        )
        if src_idx.numel() == 0:
            return torch.zeros_like(m_sq)

        logits = torch.nn.functional.leaky_relu(
            score_src[src_idx].squeeze(-1) + score_dst.squeeze(-1).unsqueeze(1),
            negative_slope=self.negative_slope,
        )
        wij = self._scaled_chunk_attention_weights(logits.unsqueeze(-1), logit_shift).squeeze(-1)
        pi = q_t.expand_as(wij) * valid.to(dtype)

        mu_d = total_mu.squeeze(-1).unsqueeze(1) - pi * wij
        mu_d = mu_d.clamp_min(0.0)
        sigma_d = total_sigma.squeeze(-1).unsqueeze(1) - pi * (1.0 - pi) * wij.square()
        sigma_d = sigma_d.clamp_min(0.0)
        denom_d = (wij + mu_d).clamp(min=self.ep_emp_eps)
        e_add = pi * wij * (denom_d.reciprocal() + sigma_d * denom_d.pow(-3.0))
        e2_add = pi * wij.square() * (denom_d.pow(-2.0) + 3.0 * sigma_d * denom_d.pow(-4.0))
        var_add = (e2_add - e_add.square()).clamp_min(0.0)

        if self.rade_variant == "rade-of":
            # For RADE-OF, the sampled add message is centered by the estimated
            # added-neighborhood mean for the same target node.
            scale_t = scale.to(dtype).view(N, 1)
            den = scale_t.squeeze(1) * e_add.sum(dim=1)
            num = scale_t * (e_add.unsqueeze(-1) * m[src_idx]).sum(dim=1)
            mu_msg = num / den.clamp(min=float(eps)).unsqueeze(1)
            centered_sq = (m[src_idx] - mu_msg.unsqueeze(1)).square()
            return scale_t * (var_add.unsqueeze(-1) * centered_sq).sum(dim=1)

        # RADE-OFS keeps added non-edge messages uncentered, so the variance
        # contribution uses m_j^2 directly.
        return scale.to(dtype).view(N, 1) * (var_add.unsqueeze(-1) * m_sq[src_idx]).sum(dim=1)

    def _nonedge_variance_sum_bernoulli(
        self,
        m: torch.Tensor,
        m_sq: torch.Tensor,
        *,
        score_src: torch.Tensor,
        score_dst: torch.Tensor,
        p: torch.Tensor | float,
        q_t: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        assert self.graph_cache is not None, "Call model.set_graph(...) once after loading data."
        N = m.size(0)
        dtype = m.dtype
        device = m.device
        if N <= 1:
            return torch.zeros_like(m_sq)

        edge_row = self.graph_cache.edge_index_clean_dir[0].to(device)
        edge_col = self.graph_cache.edge_index_clean_dir[1].to(device)
        targets = torch.arange(N, device=device, dtype=torch.long)

        def iter_nonedge_pairs():
            # Yield all clean non-edge pairs in source chunks so the dense
            # complement is never materialized as one giant tensor.
            for start, end in self._chunk_ranges(N):
                src_idx = torch.arange(start, end, device=device, dtype=torch.long)
                mask = torch.ones((N, end - start), dtype=torch.bool, device=device)
                in_chunk = (edge_row >= start) & (edge_row < end)
                if bool(in_chunk.any()):
                    mask[edge_col[in_chunk], edge_row[in_chunk] - start] = False
                mask[src_idx, src_idx - start] = False
                if not bool(mask.any()):
                    continue
                row_grid = src_idx.unsqueeze(0).expand(N, -1)
                col_grid = targets.unsqueeze(1).expand(-1, end - start)
                yield start, row_grid[mask], col_grid[mask]

        if self.rade_variant == "rade-of":
            # RADE-OF centers added non-edge messages. First compute the
            # Bernoulli-moment weighted non-edge mean message for each target.
            num = torch.zeros_like(m)
            den = torch.zeros((N,), dtype=dtype, device=device)
            for start, row_flat, col_flat in iter_nonedge_pairs():
                e_alpha, _, _ = self._pair_moments_bernoulli(
                    row=row_flat,
                    col=col_flat,
                    pi=q_t,
                    p=p,
                    q=q_t,
                    score_src=score_src,
                    score_dst=score_dst,
                    seed_offset=83 + start,
                )
                e_alpha = e_alpha.squeeze(-1)
                num = num + _scatter_add_rows(e_alpha.unsqueeze(1) * m[row_flat], col_flat, N)
                den = den + _scatter_add_scalar(e_alpha, col_flat, N)
            mu_msg = num / den.clamp(min=float(eps)).unsqueeze(1)

            add_var = torch.zeros_like(m_sq)
            # Then accumulate Var(alpha_ij) times the centered message square.
            for start, row_flat, col_flat in iter_nonedge_pairs():
                _, _, var_alpha = self._pair_moments_bernoulli(
                    row=row_flat,
                    col=col_flat,
                    pi=q_t,
                    p=p,
                    q=q_t,
                    score_src=score_src,
                    score_dst=score_dst,
                    seed_offset=83 + start,
                )
                centered_sq = (m[row_flat] - mu_msg[col_flat]).square()
                add_var = add_var + _scatter_add_rows(
                    var_alpha.squeeze(-1).unsqueeze(1) * centered_sq,
                    col_flat,
                    N,
                )
            return add_var

        add_var = torch.zeros_like(m_sq)
        # RADE-OFS keeps added non-edge messages uncentered.
        for start, row_flat, col_flat in iter_nonedge_pairs():
            _, _, var_alpha = self._pair_moments_bernoulli(
                row=row_flat,
                col=col_flat,
                pi=q_t,
                p=p,
                q=q_t,
                score_src=score_src,
                score_dst=score_dst,
                seed_offset=83 + start,
            )
            add_var = add_var + _scatter_add_rows(
                var_alpha.squeeze(-1).unsqueeze(1) * m_sq[row_flat],
                col_flat,
                N,
            )
        return add_var

    def gat_variance_proxy(
        self,
        m: torch.Tensor,
        m_sq: torch.Tensor,
        *,
        p: torch.Tensor | float,
        q: torch.Tensor | float,
        eps: float,
    ) -> torch.Tensor:
        if self._last_h_for_moments is None:
            return torch.zeros_like(m_sq)
        if self.heads != 1:
            raise ValueError("GAT GradNorm variance currently supports heads=1.")
        assert self.graph_cache is not None, "Call model.set_graph(...) once after loading data."

        h = self._last_h_for_moments.to(device=m.device, dtype=m.dtype)
        # GradNorm uses the last forward-pass hidden states but detaches GAT
        # attention parameters, so this remains a variance proxy for the
        # classifier-side messages.
        total_mu, total_sigma, score_src, score_dst, logit_shift = self._attention_totals(
            h,
            p=p,
            q=q,
            detach_params=True,
        )
        row, col = self.graph_cache.edge_index_clean_dir[0].to(m.device), self.graph_cache.edge_index_clean_dir[1].to(m.device)

        # Clean-edge part: compare random expected attention to clean attention
        # and scale message variance by Var(alpha_ij) / E[alpha_ij]^2.
        clean_alpha = self._clean_edge_alpha(h, row, col, detach_params=True).squeeze(-1)
        e_alpha, _, var_alpha = self._pair_moments_from_totals(
            row=row,
            col=col,
            pi=(1.0 - torch.as_tensor(p, dtype=m.dtype, device=m.device)),
            p=p,
            q=q,
            total_mu=total_mu,
            total_sigma=total_sigma,
            score_src=score_src,
            score_dst=score_dst,
            logit_shift=logit_shift,
            seed_offset=101,
        )
        e_alpha = e_alpha.squeeze(-1)
        var_alpha = var_alpha.squeeze(-1)
        factor = clean_alpha.square() * (var_alpha / (e_alpha.square() + float(eps)))
        neigh_var = _scatter_add_rows(factor.unsqueeze(1) * m_sq[row], col, m.size(0))

        add_var = torch.zeros_like(neigh_var)
        q_t = torch.as_tensor(q, dtype=m.dtype, device=m.device)
        if float(q_t.detach()) <= 0.0 and not q_t.requires_grad:
            return neigh_var

        N = m.size(0)
        max_comp = int(self.graph_cache.comp_size.max().item()) if self.graph_cache.comp_size.numel() > 0 else 0
        if self.moment_mode == "bernoulli":
            # Added non-edge GradNorm variance uses the same Bernoulli
            # pair-conditioned moments as the forward correction.
            add_var = self._nonedge_variance_sum_bernoulli(
                m,
                m_sq,
                score_src=score_src,
                score_dst=score_dst,
                p=p,
                q_t=q_t,
                eps=float(eps),
            )
            return neigh_var + add_var
        if self.moment_mode == "sampled" and self.nonedge_samples < max_comp:
            add_var = self._nonedge_variance_sum_sampled(
                m,
                m_sq,
                total_mu=total_mu,
                total_sigma=total_sigma,
                score_src=score_src,
                score_dst=score_dst,
                logit_shift=logit_shift,
                q_t=q_t,
                eps=float(eps),
            )
            return neigh_var + add_var

        edge_row = self.graph_cache.edge_index_clean_dir[0].to(m.device)
        edge_col = self.graph_cache.edge_index_clean_dir[1].to(m.device)

        mu_msg = None
        if self.rade_variant == "rade-of":
            num = torch.zeros_like(m)
            den = torch.zeros((N,), dtype=m.dtype, device=m.device)
            for start, end in self._chunk_ranges(N):
                src_idx = torch.arange(start, end, device=m.device)
                logits = torch.nn.functional.leaky_relu(
                    score_src[src_idx].unsqueeze(0) + score_dst.unsqueeze(1),
                    negative_slope=self.negative_slope,
                )
                wij = self._scaled_chunk_attention_weights(logits, logit_shift)
                pi_h = q_t.expand_as(wij)
                mu_d = total_mu.unsqueeze(1) - pi_h * wij
                mu_d = mu_d.clamp_min(0.0)
                sigma_d = total_sigma.unsqueeze(1) - pi_h * (1.0 - pi_h) * wij.square()
                sigma_d = sigma_d.clamp_min(0.0)
                denom_d = (wij + mu_d).clamp(min=self.ep_emp_eps)
                e_add = pi_h * wij * (denom_d.reciprocal() + sigma_d * denom_d.pow(-3.0))
                e_add = e_add.squeeze(-1)

                mask = torch.ones((N, end - start), dtype=torch.bool, device=m.device)
                in_chunk = (edge_row >= start) & (edge_row < end)
                if bool(in_chunk.any()):
                    mask[edge_col[in_chunk], edge_row[in_chunk] - start] = False
                mask[src_idx, src_idx - start] = False
                e_add = e_add * mask.to(m.dtype)

                den = den + e_add.sum(dim=1)
                num = num + e_add @ m[src_idx]
            mu_msg = num / den.clamp(min=float(eps)).unsqueeze(1)

        for start, end in self._chunk_ranges(N):
            src_idx = torch.arange(start, end, device=m.device)
            logits = torch.nn.functional.leaky_relu(
                score_src[src_idx].unsqueeze(0) + score_dst.unsqueeze(1),
                negative_slope=self.negative_slope,
            )
            wij = self._scaled_chunk_attention_weights(logits, logit_shift)
            pi_h = q_t.expand_as(wij)
            mu_d = total_mu.unsqueeze(1) - pi_h * wij
            mu_d = mu_d.clamp_min(0.0)
            sigma_d = total_sigma.unsqueeze(1) - pi_h * (1.0 - pi_h) * wij.square()
            sigma_d = sigma_d.clamp_min(0.0)
            denom_d = (wij + mu_d).clamp(min=self.ep_emp_eps)
            e_add = pi_h * wij * (denom_d.reciprocal() + sigma_d * denom_d.pow(-3.0))
            e2_add = pi_h * wij.square() * (denom_d.pow(-2.0) + 3.0 * sigma_d * denom_d.pow(-4.0))
            var_add = (e2_add - e_add.square()).clamp_min(0.0).squeeze(-1)

            mask = torch.ones((N, end - start), dtype=torch.bool, device=m.device)
            in_chunk = (edge_row >= start) & (edge_row < end)
            if bool(in_chunk.any()):
                mask[edge_col[in_chunk], edge_row[in_chunk] - start] = False
            mask[src_idx, src_idx - start] = False
            var_add = var_add * mask.to(m.dtype)

            if self.rade_variant == "rade-of":
                centered_sq = (m[src_idx].unsqueeze(0) - mu_msg.unsqueeze(1)).square()
                add_var = add_var + (var_add.unsqueeze(-1) * centered_sq).sum(dim=1)
            else:
                add_var = add_var + var_add @ m_sq[src_idx]

        return neigh_var + add_var

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
        h = self._project(x)
        self._last_h_for_moments = h.detach()
        N = h.size(0)
        use_ofs = self.rade_variant == "rade-ofs"

        if edge_index_keep is None and edge_index_add is None:
            # Clean graph path: use ordinary GAT attention over clean edges.
            # At inference RADE-OFS may add the expected non-edge contribution.
            row, col = edge_index_clean[0], edge_index_clean[1]
            mask = row != col
            row, col = row[mask], col[mask]
            alpha = self._clean_edge_alpha(h, row, col)
            out = self._aggregate(h, row, col, alpha)

            if use_ofs and (not self.training) and q > 0.0:
                nonedge_sum, _ = self._nonedge_expectation_sum(h, p=p, q=q, centered=False)
                out = out + nonedge_sum

            return self._finalize_heads(out)

        if edge_index_keep is None:
            row_k, col_k = edge_index_clean[0], edge_index_clean[1]
            mask_k = row_k != col_k
            row_k, col_k = row_k[mask_k], col_k[mask_k]
        else:
            row_k, col_k = edge_index_keep[0], edge_index_keep[1]
        if edge_index_add is not None and edge_index_add.numel() > 0:
            row_a, col_a = edge_index_add[0], edge_index_add[1]
        else:
            row_a = row_k.new_empty((0,))
            col_a = row_k.new_empty((0,))

        # Build the realized perturbed graph for this training step:
        # retained clean edges plus sampled clean non-edges. GAT normalizes
        # attention over exactly these realized incoming edges per target.
        row_all = torch.cat([row_k, row_a], dim=0)
        col_all = torch.cat([col_k, col_a], dim=0)
        alpha_all = self._edge_softmax(self._edge_logits(h, row_all, col_all), col_all, N)
        alpha_k = alpha_all[: row_k.numel()]
        alpha_a = alpha_all[row_k.numel():]

        if not self.ep_correction:
            # No expectation-preservation correction: aggregate the realized
            # perturbed graph directly.
            out = self._aggregate(h, row_all, col_all, alpha_all)
            return self._finalize_heads(out)

        # EP correction rescales retained clean-edge messages so their expected
        # attention matches clean attention under the current p,q moment model.
        # In moment_mode='sampled', _attention_totals keeps sparse clean-edge
        # terms exact and estimates only dense non-edge denominator terms.
        total_mu, total_sigma, score_src, score_dst, logit_shift = self._attention_totals(h, p=p, q=q)
        clean_alpha_k = self._clean_edge_alpha(h, row_k, col_k)
        e_keep, _, _ = self._pair_moments_from_totals(
            row=row_k,
            col=col_k,
            pi=1.0 - float(p),
            p=p,
            q=q,
            total_mu=total_mu,
            total_sigma=total_sigma,
            score_src=score_src,
            score_dst=score_dst,
            logit_shift=logit_shift,
            seed_offset=107,
        )
        corr_k = clean_alpha_k / e_keep.clamp(min=self.ep_emp_eps)
        if self.ep_corr_clip > 0.0:
            corr_k = corr_k.clamp(max=float(self.ep_corr_clip))
        out_k = self._aggregate(h, row_k, col_k, alpha_k * corr_k)

        if row_a.numel() > 0:
            if use_ofs:
                # RADE-OFS intentionally keeps added non-edge messages.
                out_a = self._aggregate(h, row_a, col_a, alpha_a)
            else:
                # RADE-OF centers added non-edge messages by subtracting the
                # expected added-neighborhood message for each target. In the
                # sampled GAT mode, that expectation is itself estimated from
                # sampled clean non-neighbors and scaled to the full complement.
                mu_h, _ = self._nonedge_expectation_sum(h, p=p, q=q, centered=True)
                msg_a = alpha_a.unsqueeze(-1) * (h[row_a] - mu_h[col_a])
                out_a = _scatter_add_heads(msg_a, col_a, N)
        else:
            out_a = torch.zeros_like(out_k)

        return self._finalize_heads(out_k + out_a)


class RADEGINConv(nn.Module):
    """
    GIN sum aggregation with paper-aligned RADE variants.

    Arguments:
      - rade_variant='rade-of'   -> overfitting-oriented variant
      - rade_variant='rade-ofs'  -> joint overfitting + over-squashing variant
      - ep_correction=True       -> use expectation-preserving aggregation correction
      - ep_correction=False      -> same sampled keep/add operator path, but without
                                    expectation-preserving correction
      - ep_expectation_mode='analytic'      -> original paper expectation
      - ep_expectation_mode='empirical_ema' -> lagged online EMA ablation
    """

    def __init__(
        self,
        mlp: nn.Module,
        eps: float = 0.0,
        train_eps: bool = False,
        rade_variant: str = "rade-of",
        ep_correction: bool = True,
        ep_expectation_mode: str = "analytic",
        ep_emp_average_mode: str = "ema",
        ep_emp_beta: float = 0.1,
        ep_emp_eps: float = 1e-12,
    ):
        super().__init__()
        self.mlp = mlp
        self.initial_eps = float(eps)
        self.rade_variant = _validate_rade_variant(rade_variant)
        self.ep_correction = bool(ep_correction)
        self.ep_expectation_mode = _validate_ep_expectation_mode(ep_expectation_mode)
        self.ep_emp_average_mode = _validate_ep_emp_average_mode(ep_emp_average_mode)
        self.ep_emp_beta = float(ep_emp_beta)
        self.ep_emp_eps = float(ep_emp_eps)

        if train_eps:
            self.eps = nn.Parameter(torch.tensor([self.initial_eps], dtype=torch.float32))
        else:
            self.register_buffer("eps", torch.tensor([self.initial_eps], dtype=torch.float32))

        self.graph_cache: Optional[GraphCache] = None

        self.register_buffer("emp_keep_prob", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("emp_add_prob", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("emp_num_updates", torch.tensor(0, dtype=torch.long))

    def set_graph(self, graph_cache: GraphCache) -> None:
        self.graph_cache = graph_cache

    def reset_parameters(self) -> None:
        reset(self.mlp)
        self.eps.data.fill_(self.initial_eps)

    def reset_ep_stats(self, p: float, q: float) -> None:
        self.emp_keep_prob.fill_(max(self.ep_emp_eps, 1.0 - float(p)))
        self.emp_add_prob.fill_(max(0.0, float(q)))
        self.emp_num_updates.zero_()

    def _emp_update_weights(self) -> tuple[float, float]:
        if int(self.emp_num_updates.item()) <= 0:
            return 0.0, 1.0
        if self.ep_emp_average_mode == "ema":
            beta = float(self.ep_emp_beta)
            if beta <= 0.0:
                return 1.0, 0.0
            if beta >= 1.0:
                return 0.0, 1.0
            return 1.0 - beta, beta

        t_prev = int(self.emp_num_updates.item())
        t_new = t_prev + 1
        return float(t_prev) / float(t_new), 1.0 / float(t_new)

    def update_ep_stats(
        self,
        *,
        edge_index_keep: Optional[torch.Tensor],
        edge_index_add: Optional[torch.Tensor],
        p: float,
        q: float,
    ) -> None:
        if self.ep_expectation_mode != "empirical_ema":
            return
        assert self.graph_cache is not None, "Call model.set_graph(...) before update_ep_stats(...)."

        device = self.emp_keep_prob.device
        old_mult, new_mult = self._emp_update_weights()
        if new_mult <= 0.0:
            return

        num_clean_dir = int(self.graph_cache.edge_index_clean_dir.size(1))
        num_nonedge_dir = int(round(float(self.graph_cache.comp_size.sum().item())))

        keep_obs = 0.0
        if edge_index_keep is not None and edge_index_keep.numel() > 0 and num_clean_dir > 0:
            keep_obs = float(edge_index_keep.size(1)) / float(num_clean_dir)

        add_obs = 0.0
        if edge_index_add is not None and edge_index_add.numel() > 0 and num_nonedge_dir > 0:
            add_obs = float(edge_index_add.size(1)) / float(num_nonedge_dir)

        keep_obs_t = torch.tensor(max(self.ep_emp_eps, keep_obs), dtype=torch.float32, device=device)
        add_obs_t = torch.tensor(max(0.0, add_obs), dtype=torch.float32, device=device)

        self.emp_keep_prob.mul_(old_mult).add_(new_mult * keep_obs_t)
        self.emp_add_prob.mul_(old_mult).add_(new_mult * add_obs_t)
        self.emp_num_updates.add_(1)

    def _get_keep_prob(self, p: float) -> float:
        if self.ep_expectation_mode == "empirical_ema":
            return float(self.emp_keep_prob.clamp(min=self.ep_emp_eps).item())
        return max(self.ep_emp_eps, 1.0 - float(p))

    def _get_add_prob(self, q: float) -> float:
        if self.ep_expectation_mode == "empirical_ema":
            return max(0.0, float(self.emp_add_prob.item()))
        return max(0.0, float(q))

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
                coeff_add = self._get_add_prob(q)
                comp_sum = self.graph_cache.complement_sum_unweighted(x)
                neigh_sum = neigh_sum + coeff_add * comp_sum

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

        # Retained clean edges: expectation-preserving rescaling by 1 / E[A'_ij].
        keep_prob = self._get_keep_prob(p)
        alpha = 1.0 / max(self.ep_emp_eps, keep_prob)
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
