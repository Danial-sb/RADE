from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn.inits import reset

from ..full_batch.rade_convs import RADEGATConv as _RADEGATConvBase


def _scatter_add_rows(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros((dim_size, values.size(1)), device=values.device, dtype=values.dtype)
    out.index_add_(0, index, values)
    return out


def _scatter_add_scalar(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros((dim_size,), device=values.device, dtype=values.dtype)
    out.index_add_(0, index, values)
    return out


def _validate_rade_variant(rade_variant: str) -> str:
    rade_variant = str(rade_variant).lower().strip()
    if rade_variant not in {"rade-of", "rade-ofs"}:
        raise ValueError(f"rade_variant must be one of {{'rade-of', 'rade-ofs'}}. Got {rade_variant}")
    return rade_variant


def _validate_ep_expectation_mode(ep_expectation_mode: str) -> str:
    ep_expectation_mode = str(ep_expectation_mode).lower().strip()
    if ep_expectation_mode not in {"analytic", "empirical_ema"}:
        raise ValueError(
            f"ep_expectation_mode must be one of {{'analytic', 'empirical_ema'}}. Got {ep_expectation_mode}"
        )
    return ep_expectation_mode


def _validate_ep_emp_average_mode(ep_emp_average_mode: str) -> str:
    ep_emp_average_mode = str(ep_emp_average_mode).lower().strip()
    if ep_emp_average_mode not in {"ema", "running_mean"}:
        raise ValueError(
            f"ep_emp_average_mode must be one of {{'ema', 'running_mean'}}. Got {ep_emp_average_mode}"
        )
    return ep_emp_average_mode


@dataclass
class BatchGraphCache:
    edge_index_clean_dir: torch.Tensor
    num_nodes: int
    deg_clean: torch.Tensor
    comp_size: torch.Tensor
    global_n_id: torch.Tensor
    global_num_nodes_total: int

    @staticmethod
    def from_edge_index(
        edge_index_clean: torch.Tensor,
        num_nodes: int,
        *,
        node_ids: Optional[torch.Tensor] = None,
        global_num_nodes_total: Optional[int] = None,
    ) -> "BatchGraphCache":
        row, col = edge_index_clean[0], edge_index_clean[1]
        mask = row != col
        row, col = row[mask], col[mask]
        edge_index_clean_dir = torch.stack([row, col], dim=0)

        ones = torch.ones(col.numel(), dtype=torch.float32, device=col.device)
        deg = torch.zeros((num_nodes,), dtype=torch.float32, device=col.device)
        deg.index_add_(0, col, ones)

        comp = (num_nodes - 1.0 - deg).clamp(min=0.0)
        if node_ids is None:
            global_n_id = torch.arange(num_nodes, device=col.device, dtype=torch.long)
        else:
            global_n_id = node_ids.to(device=col.device, dtype=torch.long)

        if global_num_nodes_total is None:
            total_nodes = int(global_n_id.max().item()) + 1 if global_n_id.numel() > 0 else int(num_nodes)
        else:
            total_nodes = int(global_num_nodes_total)

        return BatchGraphCache(
            edge_index_clean_dir=edge_index_clean_dir,
            num_nodes=num_nodes,
            deg_clean=deg,
            comp_size=comp,
            global_n_id=global_n_id,
            global_num_nodes_total=total_nodes,
        )

    def complement_mean_unweighted(self, x: torch.Tensor) -> torch.Tensor:
        n = self.num_nodes
        row, col = self.edge_index_clean_dir.to(x.device)
        neigh_sum = _scatter_add_rows(x[row], col, n)
        total_sum = x.sum(dim=0, keepdim=True)
        comp_sum = total_sum - x - neigh_sum
        denom = self.comp_size.to(x.device).clamp(min=1.0).unsqueeze(1)
        return comp_sum / denom

    def complement_sum_unweighted(self, x: torch.Tensor) -> torch.Tensor:
        n = self.num_nodes
        row, col = self.edge_index_clean_dir.to(x.device)
        neigh_sum = _scatter_add_rows(x[row], col, n)
        total_sum = x.sum(dim=0, keepdim=True)
        return total_sum - x - neigh_sum

    def complement_sum_weighted(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        n = self.num_nodes
        row, col = self.edge_index_clean_dir.to(x.device)
        wx = w.unsqueeze(1) * x
        total_wx = wx.sum(dim=0, keepdim=True)
        neigh_wx = _scatter_add_rows(wx[row], col, n)
        return total_wx - wx - neigh_wx

    def complement_mean_weighted(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        n = self.num_nodes
        row, col = self.edge_index_clean_dir.to(x.device)

        wx = w.unsqueeze(1) * x
        total_wx = wx.sum(dim=0, keepdim=True)
        total_w = w.sum()
        neigh_wx = _scatter_add_rows(wx[row], col, n)
        neigh_w = _scatter_add_scalar(w[row], col, n)

        num = total_wx - wx - neigh_wx
        den = (total_w - w - neigh_w).clamp(min=1e-12).unsqueeze(1)
        return num / den


class RADEGATConvMB(_RADEGATConvBase):
    """
    Mini-batch/local-subgraph GAT RADE operator.

    The mini-batch cache exposes the same clean directed-edge and node-count
    contract as the full-batch cache, so the full-batch attention moment
    implementation applies unchanged over the sampled local node set.
    """

    def set_graph(self, graph_cache: BatchGraphCache) -> None:
        self.graph_cache = graph_cache


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
    a = (1.0 - p) * d_minus + q * comp_size + 2.0
    b = d_minus * (1.0 - p) * p + comp_size * q * (1.0 - q)
    return a.pow(-0.5) + (3.0 / 8.0) * b * a.pow(-2.5)


def _delta_E_inv_sqrt_caseB(deg_clean: torch.Tensor, comp_size: torch.Tensor, p: float, q: float) -> torch.Tensor:
    c_minus = (comp_size - 1.0).clamp(min=0.0)
    a = (1.0 - p) * deg_clean + q * c_minus + 2.0
    b = deg_clean * (1.0 - p) * p + c_minus * q * (1.0 - q)
    return a.pow(-0.5) + (3.0 / 8.0) * b * a.pow(-2.5)


def _delta_E_inv_self(deg_clean: torch.Tensor, comp_size: torch.Tensor, p: float, q: float) -> torch.Tensor:
    mu = 1.0 + (1.0 - p) * deg_clean + q * comp_size
    var = deg_clean * (1.0 - p) * p + comp_size * q * (1.0 - q)
    return mu.pow(-1.0) + var * mu.pow(-3.0)


class RADEGCNConvMB(nn.Module):
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

        self.graph_cache: Optional[BatchGraphCache] = None

        self.register_buffer("emp_clean_edge_src", torch.empty(0, dtype=torch.long))
        self.register_buffer("emp_clean_edge_dst", torch.empty(0, dtype=torch.long))
        self.register_buffer("emp_clean_edge_raw", torch.empty(0, dtype=torch.float32))
        self.register_buffer("emp_clean_edge_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("emp_clean_edge_sq_raw", torch.empty(0, dtype=torch.float32))
        self.register_buffer("emp_clean_edge_sq_scale", torch.tensor(1.0, dtype=torch.float32))

        self.register_buffer("emp_self_raw", torch.empty(0, dtype=torch.float32))
        self.register_buffer("emp_self_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("emp_self_sq_raw", torch.empty(0, dtype=torch.float32))
        self.register_buffer("emp_self_sq_scale", torch.tensor(1.0, dtype=torch.float32))

        self.register_buffer("emp_nonedge_src", torch.empty(0, dtype=torch.long))
        self.register_buffer("emp_nonedge_dst", torch.empty(0, dtype=torch.long))
        self.register_buffer("emp_nonedge_raw", torch.empty(0, dtype=torch.float32))
        self.register_buffer("emp_nonedge_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("emp_nonedge_sq_raw", torch.empty(0, dtype=torch.float32))
        self.register_buffer("emp_nonedge_sq_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("emp_num_updates", torch.tensor(0, dtype=torch.long))

        self._emp_clean_edge_lookup: dict[int, int] = {}
        self._emp_nonedge_lookup: dict[int, int] = {}

    def _edge_key(self, src: int, dst: int) -> int:
        assert self.graph_cache is not None, "set_graph(...) must be called before computing empirical keys."
        return int(src) * int(self.graph_cache.global_num_nodes_total) + int(dst)

    def _rebuild_sparse_lookup(self) -> None:
        if self.graph_cache is None:
            self._emp_clean_edge_lookup = {}
            self._emp_nonedge_lookup = {}
            return

        total = int(self.graph_cache.global_num_nodes_total)
        self._emp_clean_edge_lookup = {
            int(src) * total + int(dst): idx
            for idx, (src, dst) in enumerate(
                zip(self.emp_clean_edge_src.detach().cpu().tolist(), self.emp_clean_edge_dst.detach().cpu().tolist())
            )
        }
        self._emp_nonedge_lookup = {
            int(src) * total + int(dst): idx
            for idx, (src, dst) in enumerate(
                zip(self.emp_nonedge_src.detach().cpu().tolist(), self.emp_nonedge_dst.detach().cpu().tolist())
            )
        }

    def _ensure_self_storage(self) -> None:
        assert self.graph_cache is not None, "set_graph(...) must be called before empirical EP state updates."
        total = int(self.graph_cache.global_num_nodes_total)
        device = self.emp_self_scale.device
        if self.emp_self_raw.numel() == total and self.emp_self_sq_raw.numel() == total:
            return
        if self.emp_self_raw.numel() == 0:
            self.emp_self_raw = torch.zeros((total,), dtype=torch.float32, device=device)
        if self.emp_self_sq_raw.numel() == 0:
            self.emp_self_sq_raw = torch.zeros((total,), dtype=torch.float32, device=device)
        if self.emp_self_raw.numel() < total:
            pad = torch.zeros((total - self.emp_self_raw.numel(),), dtype=torch.float32, device=device)
            self.emp_self_raw = torch.cat([self.emp_self_raw, pad], dim=0)
        elif self.emp_self_raw.numel() > total:
            self.emp_self_raw = self.emp_self_raw[:total]
        if self.emp_self_sq_raw.numel() < total:
            pad = torch.zeros((total - self.emp_self_sq_raw.numel(),), dtype=torch.float32, device=device)
            self.emp_self_sq_raw = torch.cat([self.emp_self_sq_raw, pad], dim=0)
        elif self.emp_self_sq_raw.numel() > total:
            self.emp_self_sq_raw = self.emp_self_sq_raw[:total]

    def set_graph(self, graph_cache: BatchGraphCache) -> None:
        self.graph_cache = graph_cache
        self._ensure_self_storage()
        self._rebuild_sparse_lookup()

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

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
        if self.ep_expectation_mode != "empirical_ema":
            return
        self.emp_clean_edge_src = torch.empty(0, dtype=torch.long, device=self.emp_clean_edge_src.device)
        self.emp_clean_edge_dst = torch.empty(0, dtype=torch.long, device=self.emp_clean_edge_dst.device)
        self.emp_clean_edge_raw = torch.empty(0, dtype=torch.float32, device=self.emp_clean_edge_raw.device)
        self.emp_clean_edge_scale.fill_(1.0)
        self.emp_clean_edge_sq_raw = torch.empty(0, dtype=torch.float32, device=self.emp_clean_edge_sq_raw.device)
        self.emp_clean_edge_sq_scale.fill_(1.0)

        if self.graph_cache is not None:
            self._ensure_self_storage()
            self.emp_self_raw.zero_()
            self.emp_self_sq_raw.zero_()
        else:
            self.emp_self_raw = torch.empty(0, dtype=torch.float32, device=self.emp_self_raw.device)
            self.emp_self_sq_raw = torch.empty(0, dtype=torch.float32, device=self.emp_self_sq_raw.device)
        self.emp_self_scale.fill_(1.0)
        self.emp_self_sq_scale.fill_(1.0)

        self.emp_nonedge_src = torch.empty(0, dtype=torch.long, device=self.emp_nonedge_src.device)
        self.emp_nonedge_dst = torch.empty(0, dtype=torch.long, device=self.emp_nonedge_dst.device)
        self.emp_nonedge_raw = torch.empty(0, dtype=torch.float32, device=self.emp_nonedge_raw.device)
        self.emp_nonedge_scale.fill_(1.0)
        self.emp_nonedge_sq_raw = torch.empty(0, dtype=torch.float32, device=self.emp_nonedge_sq_raw.device)
        self.emp_nonedge_sq_scale.fill_(1.0)
        self.emp_num_updates.zero_()
        self._rebuild_sparse_lookup()

    def _append_clean_edge_raw(self, src: torch.Tensor, dst: torch.Tensor, values: torch.Tensor, *, new_mult: float) -> None:
        if src.numel() == 0:
            return
        scale = float(self.emp_clean_edge_scale.item())
        if scale <= 0.0:
            raise RuntimeError("emp_clean_edge_scale must be positive when beta < 1.")

        src_list = src.detach().cpu().tolist()
        dst_list = dst.detach().cpu().tolist()
        values_detached = values.detach().to(dtype=torch.float32)
        val_list = (float(new_mult) * values_detached / scale).cpu().tolist()
        sq_scale = float(self.emp_clean_edge_sq_scale.item())
        if sq_scale <= 0.0:
            raise RuntimeError("emp_clean_edge_sq_scale must be positive when beta < 1.")
        sq_val_list = (float(new_mult) * values_detached.square() / sq_scale).cpu().tolist()

        new_src = []
        new_dst = []
        new_vals = []
        new_sq_vals = []
        existing_idx = []
        existing_vals = []
        existing_sq_vals = []

        for src_i, dst_i, val_i, sq_val_i in zip(src_list, dst_list, val_list, sq_val_list):
            key = self._edge_key(src_i, dst_i)
            idx = self._emp_clean_edge_lookup.get(key)
            if idx is None:
                idx = int(self.emp_clean_edge_raw.numel()) + len(new_src)
                self._emp_clean_edge_lookup[key] = idx
                new_src.append(int(src_i))
                new_dst.append(int(dst_i))
                new_vals.append(float(val_i))
                new_sq_vals.append(float(sq_val_i))
            else:
                existing_idx.append(int(idx))
                existing_vals.append(float(val_i))
                existing_sq_vals.append(float(sq_val_i))

        if existing_idx:
            idx_t = torch.tensor(existing_idx, dtype=torch.long, device=self.emp_clean_edge_raw.device)
            val_t = torch.tensor(existing_vals, dtype=torch.float32, device=self.emp_clean_edge_raw.device)
            self.emp_clean_edge_raw.index_add_(0, idx_t, val_t)
            sq_val_t = torch.tensor(existing_sq_vals, dtype=torch.float32, device=self.emp_clean_edge_sq_raw.device)
            self.emp_clean_edge_sq_raw.index_add_(0, idx_t.to(self.emp_clean_edge_sq_raw.device), sq_val_t)

        if new_src:
            device = self.emp_clean_edge_raw.device
            self.emp_clean_edge_src = torch.cat(
                [self.emp_clean_edge_src, torch.tensor(new_src, dtype=torch.long, device=device)],
                dim=0,
            )
            self.emp_clean_edge_dst = torch.cat(
                [self.emp_clean_edge_dst, torch.tensor(new_dst, dtype=torch.long, device=device)],
                dim=0,
            )
            self.emp_clean_edge_raw = torch.cat(
                [self.emp_clean_edge_raw, torch.tensor(new_vals, dtype=torch.float32, device=device)],
                dim=0,
            )
            self.emp_clean_edge_sq_raw = torch.cat(
                [self.emp_clean_edge_sq_raw, torch.tensor(new_sq_vals, dtype=torch.float32, device=self.emp_clean_edge_sq_raw.device)],
                dim=0,
            )

    def _append_nonedge_raw(self, src: torch.Tensor, dst: torch.Tensor, values: torch.Tensor, *, new_mult: float) -> None:
        if src.numel() == 0:
            return
        scale = float(self.emp_nonedge_scale.item())
        if scale <= 0.0:
            raise RuntimeError("emp_nonedge_scale must be positive when beta < 1.")

        src_list = src.detach().cpu().tolist()
        dst_list = dst.detach().cpu().tolist()
        values_detached = values.detach().to(dtype=torch.float32)
        val_list = (float(new_mult) * values_detached / scale).cpu().tolist()
        sq_scale = float(self.emp_nonedge_sq_scale.item())
        if sq_scale <= 0.0:
            raise RuntimeError("emp_nonedge_sq_scale must be positive when beta < 1.")
        sq_val_list = (float(new_mult) * values_detached.square() / sq_scale).cpu().tolist()

        new_src = []
        new_dst = []
        new_vals = []
        new_sq_vals = []
        existing_idx = []
        existing_vals = []
        existing_sq_vals = []

        for src_i, dst_i, val_i, sq_val_i in zip(src_list, dst_list, val_list, sq_val_list):
            key = self._edge_key(src_i, dst_i)
            idx = self._emp_nonedge_lookup.get(key)
            if idx is None:
                idx = int(self.emp_nonedge_raw.numel()) + len(new_src)
                self._emp_nonedge_lookup[key] = idx
                new_src.append(int(src_i))
                new_dst.append(int(dst_i))
                new_vals.append(float(val_i))
                new_sq_vals.append(float(sq_val_i))
            else:
                existing_idx.append(int(idx))
                existing_vals.append(float(val_i))
                existing_sq_vals.append(float(sq_val_i))

        if existing_idx:
            idx_t = torch.tensor(existing_idx, dtype=torch.long, device=self.emp_nonedge_raw.device)
            val_t = torch.tensor(existing_vals, dtype=torch.float32, device=self.emp_nonedge_raw.device)
            self.emp_nonedge_raw.index_add_(0, idx_t, val_t)
            sq_val_t = torch.tensor(existing_sq_vals, dtype=torch.float32, device=self.emp_nonedge_sq_raw.device)
            self.emp_nonedge_sq_raw.index_add_(0, idx_t.to(self.emp_nonedge_sq_raw.device), sq_val_t)

        if new_src:
            device = self.emp_nonedge_raw.device
            self.emp_nonedge_src = torch.cat(
                [self.emp_nonedge_src, torch.tensor(new_src, dtype=torch.long, device=device)],
                dim=0,
            )
            self.emp_nonedge_dst = torch.cat(
                [self.emp_nonedge_dst, torch.tensor(new_dst, dtype=torch.long, device=device)],
                dim=0,
            )
            self.emp_nonedge_raw = torch.cat(
                [self.emp_nonedge_raw, torch.tensor(new_vals, dtype=torch.float32, device=device)],
                dim=0,
            )
            self.emp_nonedge_sq_raw = torch.cat(
                [self.emp_nonedge_sq_raw, torch.tensor(new_sq_vals, dtype=torch.float32, device=self.emp_nonedge_sq_raw.device)],
                dim=0,
            )

    def _gcn_sample_coeffs(
        self,
        edge_index_keep: Optional[torch.Tensor],
        edge_index_add: Optional[torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.graph_cache is not None, "set_graph(...) must be called before update_ep_stats(...)."

        n = int(self.graph_cache.num_nodes)
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

        ar = torch.arange(n, device=device)
        row_all = torch.cat([row_k, row_a, ar], dim=0)
        col_all = torch.cat([col_k, col_a, ar], dim=0)

        w_all = torch.ones(row_all.numel(), device=device, dtype=dtype)
        deg_aug = _scatter_add_scalar(w_all, col_all, n).clamp(min=1.0)
        inv_sqrt_aug = deg_aug.pow(-0.5)

        gamma_k = inv_sqrt_aug[row_k] * inv_sqrt_aug[col_k]
        gamma_a = (
            inv_sqrt_aug[row_a] * inv_sqrt_aug[col_a]
            if row_a.numel()
            else torch.empty(0, device=device, dtype=dtype)
        )
        gamma_self = inv_sqrt_aug * inv_sqrt_aug
        return row_k, col_k, gamma_k, row_a, col_a, gamma_a, gamma_self

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
        assert self.graph_cache is not None, "set_graph(...) must be called before update_ep_stats(...)."

        self._ensure_self_storage()
        old_mult, new_mult = self._emp_update_weights()
        if new_mult <= 0.0:
            return

        device = self.graph_cache.deg_clean.device
        edge_index_keep_stats = edge_index_keep
        if edge_index_keep_stats is None and edge_index_add is not None:
            edge_index_keep_stats = self.graph_cache.edge_index_clean_dir
        row_k, col_k, gamma_k, row_a, col_a, gamma_a, gamma_self = self._gcn_sample_coeffs(
            edge_index_keep=edge_index_keep_stats,
            edge_index_add=edge_index_add,
            device=device,
            dtype=torch.float32,
        )
        global_ids = self.graph_cache.global_n_id.to(device)

        src_k = global_ids[row_k] if row_k.numel() else row_k
        dst_k = global_ids[col_k] if col_k.numel() else col_k
        src_a = global_ids[row_a] if row_a.numel() else row_a
        dst_a = global_ids[col_a] if col_a.numel() else col_a

        if old_mult <= 1e-12:
            self.emp_clean_edge_src = torch.empty(0, dtype=torch.long, device=self.emp_clean_edge_src.device)
            self.emp_clean_edge_dst = torch.empty(0, dtype=torch.long, device=self.emp_clean_edge_dst.device)
            self.emp_clean_edge_raw = torch.empty(0, dtype=torch.float32, device=self.emp_clean_edge_raw.device)
            self.emp_clean_edge_scale.fill_(1.0)
            self.emp_clean_edge_sq_raw = torch.empty(0, dtype=torch.float32, device=self.emp_clean_edge_sq_raw.device)
            self.emp_clean_edge_sq_scale.fill_(1.0)
            self._emp_clean_edge_lookup = {}
            self._append_clean_edge_raw(src_k, dst_k, gamma_k, new_mult=1.0)

            self.emp_self_scale.fill_(1.0)
            self.emp_self_sq_scale.fill_(1.0)
            self.emp_self_raw.zero_()
            self.emp_self_sq_raw.zero_()
            if global_ids.numel() > 0:
                gamma_self_detached = gamma_self.detach().to(self.emp_self_raw.device, dtype=torch.float32)
                self.emp_self_raw.index_copy_(
                    0,
                    global_ids.to(self.emp_self_raw.device, dtype=torch.long),
                    gamma_self_detached,
                )
                self.emp_self_sq_raw.index_copy_(
                    0,
                    global_ids.to(self.emp_self_sq_raw.device, dtype=torch.long),
                    gamma_self_detached.square().to(self.emp_self_sq_raw.device),
                )

            self.emp_nonedge_src = torch.empty(0, dtype=torch.long, device=self.emp_nonedge_src.device)
            self.emp_nonedge_dst = torch.empty(0, dtype=torch.long, device=self.emp_nonedge_dst.device)
            self.emp_nonedge_raw = torch.empty(0, dtype=torch.float32, device=self.emp_nonedge_raw.device)
            self.emp_nonedge_scale.fill_(1.0)
            self.emp_nonedge_sq_raw = torch.empty(0, dtype=torch.float32, device=self.emp_nonedge_sq_raw.device)
            self.emp_nonedge_sq_scale.fill_(1.0)
            self._emp_nonedge_lookup = {}
            self._append_nonedge_raw(src_a, dst_a, gamma_a, new_mult=1.0)
            self.emp_num_updates.add_(1)
            return

        self.emp_clean_edge_scale.mul_(old_mult)
        self.emp_clean_edge_sq_scale.mul_(old_mult)
        self._append_clean_edge_raw(src_k, dst_k, gamma_k, new_mult=new_mult)

        self.emp_self_scale.mul_(old_mult)
        self.emp_self_sq_scale.mul_(old_mult)
        gamma_self_detached = gamma_self.detach().to(self.emp_self_raw.device, dtype=torch.float32)
        upd_self = (
            new_mult * gamma_self_detached
        ) / self.emp_self_scale
        self.emp_self_raw.index_add_(
            0,
            global_ids.to(self.emp_self_raw.device, dtype=torch.long),
            upd_self,
        )
        upd_self_sq = (
            new_mult * gamma_self_detached.square().to(self.emp_self_sq_raw.device)
        ) / self.emp_self_sq_scale
        self.emp_self_sq_raw.index_add_(
            0,
            global_ids.to(self.emp_self_sq_raw.device, dtype=torch.long),
            upd_self_sq,
        )

        self.emp_nonedge_scale.mul_(old_mult)
        self.emp_nonedge_sq_scale.mul_(old_mult)
        self._append_nonedge_raw(src_a, dst_a, gamma_a, new_mult=new_mult)
        self.emp_num_updates.add_(1)

    def _get_clean_edge_emp(
        self,
        row: torch.Tensor,
        col: torch.Tensor,
        *,
        fallback: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if row.numel() == 0:
            return torch.empty(0, dtype=dtype, device=device)
        assert self.graph_cache is not None, "set_graph(...) must be called before forward(...)."

        global_ids = self.graph_cache.global_n_id.to(row.device)
        src = global_ids[row].detach().cpu().tolist()
        dst = global_ids[col].detach().cpu().tolist()
        vals = fallback.detach().cpu().tolist()

        out = []
        scale = float(self.emp_clean_edge_scale.item())
        for src_i, dst_i, fallback_i in zip(src, dst, vals):
            idx = self._emp_clean_edge_lookup.get(self._edge_key(src_i, dst_i))
            if idx is None:
                out.append(float(fallback_i))
            else:
                out.append(scale * float(self.emp_clean_edge_raw[idx].item()))

        return torch.tensor(out, dtype=dtype, device=device)

    def _get_self_emp(
        self,
        *,
        fallback: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        assert self.graph_cache is not None, "set_graph(...) must be called before forward(...)."
        self._ensure_self_storage()
        global_ids = self.graph_cache.global_n_id.to(self.emp_self_raw.device, dtype=torch.long)
        if global_ids.numel() == 0:
            return fallback.to(device=device, dtype=dtype)

        scale = float(self.emp_self_scale.item())
        empirical = (scale * self.emp_self_raw.index_select(0, global_ids)).to(device=device, dtype=dtype)
        if self.emp_num_updates.item() <= 0:
            return fallback.to(device=device, dtype=dtype)

        fallback = fallback.to(device=device, dtype=dtype)
        missing = empirical <= 0.0
        empirical[missing] = fallback[missing]
        return empirical

    def _apply_emp_nonedge_expectation(self, x: torch.Tensor) -> torch.Tensor:
        if self.emp_nonedge_raw.numel() == 0:
            return torch.zeros((x.size(0), x.size(1)), device=x.device, dtype=x.dtype)
        assert self.graph_cache is not None, "set_graph(...) must be called before forward(...)."

        out = torch.zeros((x.size(0), x.size(1)), device=x.device, dtype=x.dtype)
        local_pos = {
            int(global_id): idx for idx, global_id in enumerate(self.graph_cache.global_n_id.detach().cpu().tolist())
        }
        scale = float(self.emp_nonedge_scale.item())

        src_list = self.emp_nonedge_src.detach().cpu().tolist()
        dst_list = self.emp_nonedge_dst.detach().cpu().tolist()
        val_list = (scale * self.emp_nonedge_raw).detach().cpu().tolist()

        for src_g, dst_g, val in zip(src_list, dst_list, val_list):
            src_l = local_pos.get(int(src_g))
            dst_l = local_pos.get(int(dst_g))
            if src_l is None or dst_l is None:
                continue
            out[dst_l] = out[dst_l] + float(val) * x[src_l]

        return out

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
        assert self.graph_cache is not None, "set_graph(...) must be called before empirical variance lookup."

        device = m_sq.device
        dtype = m_sq.dtype
        n = int(self.graph_cache.num_nodes)
        global_ids = self.graph_cache.global_n_id.detach().cpu().tolist()
        local_pos = {int(global_id): idx for idx, global_id in enumerate(global_ids)}
        total = int(self.graph_cache.global_num_nodes_total)

        row, col = self.graph_cache.edge_index_clean_dir
        row_cpu = row.detach().cpu().tolist()
        col_cpu = col.detach().cpu().tolist()
        found = []
        target_pos = []
        for pos, (src_l, dst_l) in enumerate(zip(row_cpu, col_cpu)):
            src_g = int(global_ids[int(src_l)])
            dst_g = int(global_ids[int(dst_l)])
            idx = self._emp_clean_edge_lookup.get(src_g * total + dst_g)
            if idx is not None:
                found.append(int(idx))
                target_pos.append(int(pos))

        e = torch.zeros(row.numel(), device=device, dtype=dtype)
        e2 = torch.zeros_like(e)
        if found:
            idx_t = torch.tensor(found, dtype=torch.long, device=device)
            pos_t = torch.tensor(target_pos, dtype=torch.long, device=device)
            e.index_copy_(
                0,
                pos_t,
                self.emp_clean_edge_scale.to(device=device, dtype=dtype)
                * self.emp_clean_edge_raw.to(device=device, dtype=dtype).index_select(0, idx_t),
            )
            e2.index_copy_(
                0,
                pos_t,
                self.emp_clean_edge_sq_scale.to(device=device, dtype=dtype)
                * self.emp_clean_edge_sq_raw.to(device=device, dtype=dtype).index_select(0, idx_t),
            )

        var_g = (e2 - e * e).clamp_min(0.0)

        deg = self.graph_cache.deg_clean.to(device=device, dtype=dtype)
        d_hat = (deg + 1.0).clamp_min(1.0)
        row_d = row.to(device)
        col_d = col.to(device)
        alpha_sq = 1.0 / (d_hat[row_d] * d_hat[col_d])
        factor = alpha_sq * (var_g / (e * e + float(eps)))
        neigh_var = _scatter_add_rows(factor.unsqueeze(1) * m_sq[row_d], col_d, n)

        add_var = torch.zeros_like(neigh_var)
        if self.emp_nonedge_raw.numel() == 0:
            return neigh_var

        src_list = self.emp_nonedge_src.detach().cpu().tolist()
        dst_list = self.emp_nonedge_dst.detach().cpu().tolist()
        local_src = []
        local_dst = []
        sparse_idx = []
        for idx, (src_g, dst_g) in enumerate(zip(src_list, dst_list)):
            src_l = local_pos.get(int(src_g))
            dst_l = local_pos.get(int(dst_g))
            if src_l is None or dst_l is None:
                continue
            local_src.append(src_l)
            local_dst.append(dst_l)
            sparse_idx.append(idx)
        if not sparse_idx:
            return neigh_var

        sparse_idx_t = torch.tensor(sparse_idx, dtype=torch.long, device=device)
        row_a = torch.tensor(local_src, dtype=torch.long, device=device)
        col_a = torch.tensor(local_dst, dtype=torch.long, device=device)
        e_a = self.emp_nonedge_scale.to(device=device, dtype=dtype) * self.emp_nonedge_raw.to(device=device, dtype=dtype).index_select(0, sparse_idx_t)
        e2_a = self.emp_nonedge_sq_scale.to(device=device, dtype=dtype) * self.emp_nonedge_sq_raw.to(device=device, dtype=dtype).index_select(0, sparse_idx_t)
        var_a = (e2_a - e_a * e_a).clamp_min(0.0)
        add_var = _scatter_add_rows(var_a.unsqueeze(1) * m_sq[row_a], col_a, n)
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
        assert self.graph_cache is not None, "Mini-batch RADE requires set_graph(...) for each batch."

        x = self.lin(x)
        n = x.size(0)
        use_ofs = self.rade_variant == "rade-ofs"

        if edge_index_keep is None and edge_index_add is None:
            row, col = edge_index_clean[0], edge_index_clean[1]
            mask = row != col
            row, col = row[mask], col[mask]
            ar = torch.arange(n, device=x.device)
            row = torch.cat([row, ar], dim=0)
            col = torch.cat([col, ar], dim=0)

            w = torch.ones(row.numel(), device=x.device, dtype=x.dtype)
            deg = _scatter_add_scalar(w, col, n).clamp(min=1.0)
            inv_sqrt = deg.pow(-0.5)
            gamma = inv_sqrt[row] * inv_sqrt[col]
            out = _scatter_add_rows(gamma.unsqueeze(1) * x[row], col, n)

            if use_ofs and (not self.training) and (q > 0.0):
                if self.ep_expectation_mode == "empirical_ema":
                    out = out + self._apply_emp_nonedge_expectation(x)
                else:
                    deg_clean = self.graph_cache.deg_clean.to(x.device)
                    comp_size = self.graph_cache.comp_size.to(x.device)
                    t_b = _delta_E_inv_sqrt_caseB(deg_clean, comp_size, p=p, q=q).to(x.device, dtype=x.dtype)
                    comp_wx = self.graph_cache.complement_sum_weighted(x, w=t_b)
                    out = out + (q * t_b).unsqueeze(1) * comp_wx

            if self.bias is not None:
                out = out + self.bias
            return out

        if edge_index_keep is None:
            row_k, col_k = edge_index_clean[0], edge_index_clean[1]
            mask = row_k != col_k
            row_k, col_k = row_k[mask], col_k[mask]
        else:
            row_k, col_k = edge_index_keep[0], edge_index_keep[1]
        if edge_index_add is not None and edge_index_add.numel() > 0:
            row_a, col_a = edge_index_add[0], edge_index_add[1]
        else:
            row_a = row_k.new_empty((0,))
            col_a = row_k.new_empty((0,))

        ar = torch.arange(n, device=x.device)
        row_all = torch.cat([row_k, row_a, ar], dim=0)
        col_all = torch.cat([col_k, col_a, ar], dim=0)

        w_all = torch.ones(row_all.numel(), device=x.device, dtype=x.dtype)
        deg_aug = _scatter_add_scalar(w_all, col_all, n).clamp(min=1.0)
        inv_sqrt_aug = deg_aug.pow(-0.5)

        gamma_k = inv_sqrt_aug[row_k] * inv_sqrt_aug[col_k]
        gamma_a = inv_sqrt_aug[row_a] * inv_sqrt_aug[col_a] if row_a.numel() else None
        gamma_self = inv_sqrt_aug * inv_sqrt_aug

        if not self.ep_correction:
            out_k = _scatter_add_rows(gamma_k.unsqueeze(1) * x[row_k], col_k, n)
            out_a = (
                _scatter_add_rows(gamma_a.unsqueeze(1) * x[row_a], col_a, n)
                if row_a.numel()
                else torch.zeros_like(out_k)
            )
            out_self = gamma_self.unsqueeze(1) * x
            out = out_k + out_a + out_self
            if self.bias is not None:
                out = out + self.bias
            return out

        deg_clean = self.graph_cache.deg_clean.to(x.device)
        comp_size = self.graph_cache.comp_size.to(x.device)
        deg_hat = (deg_clean + 1.0).clamp(min=1.0)
        inv_sqrt_hat = deg_hat.pow(-0.5)

        alpha_k = inv_sqrt_hat[row_k] * inv_sqrt_hat[col_k]
        alpha_self = inv_sqrt_hat * inv_sqrt_hat

        if self.ep_expectation_mode == "empirical_ema":
            e_gamma_k = self._get_clean_edge_emp(
                row_k,
                col_k,
                fallback=gamma_k.detach().to(x.device, dtype=x.dtype),
                dtype=x.dtype,
                device=x.device,
            )
        else:
            t_a = _delta_E_inv_sqrt_caseA(deg_clean, comp_size, p=p, q=q).to(x.device, dtype=x.dtype)
            analytic_edge = (1.0 - p) * t_a[row_k] * t_a[col_k]
            e_gamma_k = analytic_edge

        corr_k = alpha_k / e_gamma_k.clamp(min=self.ep_emp_eps)
        out_k = _scatter_add_rows((gamma_k * corr_k).unsqueeze(1) * x[row_k], col_k, n)

        if row_a.numel():
            sum_a = _scatter_add_rows(gamma_a.unsqueeze(1) * x[row_a], col_a, n)
            if use_ofs:
                out_a = sum_a
            else:
                if self.ep_expectation_mode == "empirical_ema":
                    out_a = sum_a - self._apply_emp_nonedge_expectation(x)
                else:
                    t_b = _delta_E_inv_sqrt_caseB(deg_clean, comp_size, p=p, q=q).to(x.device, dtype=x.dtype)
                    mu = self.graph_cache.complement_mean_weighted(x, w=t_b)
                    sum_g = _scatter_add_scalar(gamma_a, col_a, n).unsqueeze(1)
                    out_a = sum_a - mu * sum_g
        else:
            out_a = torch.zeros_like(out_k)

        if self.correct_self_loop:
            if self.ep_expectation_mode == "empirical_ema":
                e_gamma_self = self._get_self_emp(
                    fallback=gamma_self.detach().to(x.device, dtype=x.dtype),
                    dtype=x.dtype,
                    device=x.device,
                )
            else:
                analytic_self = _delta_E_inv_self(deg_clean, comp_size, p=p, q=q).to(x.device, dtype=x.dtype)
                e_gamma_self = analytic_self
            corr_self = alpha_self / e_gamma_self.clamp(min=self.ep_emp_eps)
            out_self = (gamma_self * corr_self).unsqueeze(1) * x
        else:
            out_self = gamma_self.unsqueeze(1) * x

        out = out_k + out_a + out_self
        if self.bias is not None:
            out = out + self.bias
        return out


class RADEGINConvMB(nn.Module):
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

        self.graph_cache: Optional[BatchGraphCache] = None
        self.register_buffer("emp_keep_prob", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("emp_add_prob", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("emp_num_updates", torch.tensor(0, dtype=torch.long))

    def set_graph(self, graph_cache: BatchGraphCache) -> None:
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
        assert self.graph_cache is not None, "set_graph(...) must be called before update_ep_stats(...)."

        old_mult, new_mult = self._emp_update_weights()
        if new_mult <= 0.0:
            return

        num_clean_dir = int(self.graph_cache.edge_index_clean_dir.size(1))
        num_nonedge_dir = int(round(float(self.graph_cache.comp_size.sum().item())))

        keep_obs = 0.0
        if edge_index_keep is None and edge_index_add is not None and num_clean_dir > 0:
            keep_obs = 1.0
        elif edge_index_keep is not None and edge_index_keep.numel() > 0 and num_clean_dir > 0:
            keep_obs = float(edge_index_keep.size(1)) / float(num_clean_dir)

        add_obs = 0.0
        if edge_index_add is not None and edge_index_add.numel() > 0 and num_nonedge_dir > 0:
            add_obs = float(edge_index_add.size(1)) / float(num_nonedge_dir)

        device = self.emp_keep_prob.device
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
        assert self.graph_cache is not None, "Mini-batch RADE requires set_graph(...) for each batch."
        n = x.size(0)
        use_ofs = self.rade_variant == "rade-ofs"

        if edge_index_keep is None and edge_index_add is None:
            row, col = edge_index_clean[0], edge_index_clean[1]
            mask = row != col
            row, col = row[mask], col[mask]
            neigh_sum = _scatter_add_rows(x[row], col, n)

            if use_ofs and (not self.training) and (q > 0.0):
                coeff_add = self._get_add_prob(q)
                comp_sum = self.graph_cache.complement_sum_unweighted(x)
                neigh_sum = neigh_sum + coeff_add * comp_sum

            out = (1.0 + self.eps.to(x.device)) * x + neigh_sum
            return self.mlp(out)

        if edge_index_keep is None:
            row_k, col_k = edge_index_clean[0], edge_index_clean[1]
            mask = row_k != col_k
            row_k, col_k = row_k[mask], col_k[mask]
        else:
            row_k, col_k = edge_index_keep[0], edge_index_keep[1]
        if edge_index_add is not None and edge_index_add.numel() > 0:
            row_a, col_a = edge_index_add[0], edge_index_add[1]
        else:
            row_a = row_k.new_empty((0,))
            col_a = col_k.new_empty((0,))

        if not self.ep_correction:
            keep_sum = _scatter_add_rows(x[row_k], col_k, n)
            add_sum = (
                _scatter_add_rows(x[row_a], col_a, n)
                if row_a.numel() > 0
                else torch.zeros_like(keep_sum)
            )
            out = (1.0 + self.eps.to(x.device)) * x + keep_sum + add_sum
            return self.mlp(out)

        keep_prob = self._get_keep_prob(p)
        keep_sum = _scatter_add_rows(x[row_k], col_k, n) * (1.0 / max(self.ep_emp_eps, keep_prob))

        if row_a.numel() > 0:
            add_sum_raw = _scatter_add_rows(x[row_a], col_a, n)
            if use_ofs:
                add_sum = add_sum_raw
            else:
                mu = self.graph_cache.complement_mean_unweighted(x)
                add_deg = _scatter_add_scalar(
                    torch.ones(row_a.numel(), device=x.device, dtype=x.dtype),
                    col_a,
                    n,
                ).unsqueeze(1)
                add_sum = add_sum_raw - mu * add_deg
        else:
            add_sum = torch.zeros_like(keep_sum)

        out = (1.0 + self.eps.to(x.device)) * x + keep_sum + add_sum
        return self.mlp(out)
