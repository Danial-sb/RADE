from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn.inits import reset

try:
    from RADE_Node_Classification.full_batch.rade_convs import RADEGATConv as _RADEGATConvBase
except Exception:
    import importlib.util
    import sys
    from pathlib import Path

    _RADE_GAT_PATH = Path(__file__).resolve().parents[1] / "RADE_Node_Classification" / "full_batch" / "rade_convs.py"
    _RADE_GAT_SPEC = importlib.util.spec_from_file_location("rade_full_batch_gat_base", _RADE_GAT_PATH)
    if _RADE_GAT_SPEC is None or _RADE_GAT_SPEC.loader is None:
        raise ImportError(f"Could not load full-batch RADEGATConv from {_RADE_GAT_PATH}")
    _RADE_GAT_MODULE = importlib.util.module_from_spec(_RADE_GAT_SPEC)
    sys.modules[_RADE_GAT_SPEC.name] = _RADE_GAT_MODULE
    _RADE_GAT_SPEC.loader.exec_module(_RADE_GAT_MODULE)
    _RADEGATConvBase = _RADE_GAT_MODULE.RADEGATConv


def _scatter_add_rows(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros((dim_size, values.size(1)), device=values.device, dtype=values.dtype)
    out.index_add_(0, index, values)
    return out


def _scatter_add_scalar(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros((dim_size,), device=values.device, dtype=values.dtype)
    out.index_add_(0, index, values)
    return out


def _scatter_add_heads(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    if values.numel() == 0:
        return torch.zeros(
            (dim_size, values.size(1), values.size(2)),
            device=values.device,
            dtype=values.dtype,
        )
    flat = values.reshape(values.size(0), values.size(1) * values.size(2))
    out = _scatter_add_rows(flat, index, dim_size)
    return out.view(dim_size, values.size(1), values.size(2))


def _scatter_add_graph_rows(values: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    out = torch.zeros((num_graphs, values.size(1)), device=values.device, dtype=values.dtype)
    out.index_add_(0, batch, values)
    return out


def _scatter_add_graph_scalar(values: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    out = torch.zeros((num_graphs,), device=values.device, dtype=values.dtype)
    out.index_add_(0, batch, values)
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
class BatchedGraphCache:
    edge_index_clean_dir: torch.Tensor
    batch: torch.Tensor
    num_nodes: int
    num_graphs: int
    deg_clean: torch.Tensor
    comp_size: torch.Tensor
    n_per_node: torch.Tensor
    global_n_id: torch.Tensor
    global_num_nodes_total: int

    @staticmethod
    def from_edge_index(
        edge_index_clean: torch.Tensor,
        batch: torch.Tensor,
        *,
        num_graphs: Optional[int] = None,
        node_ids: Optional[torch.Tensor] = None,
        global_num_nodes_total: Optional[int] = None,
    ) -> "BatchedGraphCache":
        batch = batch.to(torch.long)
        num_nodes = int(batch.numel())
        num_graphs = int(batch.max().item() + 1) if num_graphs is None and batch.numel() > 0 else int(num_graphs or 0)

        row, col = edge_index_clean[0], edge_index_clean[1]
        mask = row != col
        row, col = row[mask], col[mask]
        edge_index_clean_dir = torch.stack([row, col], dim=0)

        ones_e = torch.ones(col.numel(), dtype=torch.float32, device=col.device)
        deg = torch.zeros((num_nodes,), dtype=torch.float32, device=col.device)
        deg.index_add_(0, col, ones_e)

        ones_n = torch.ones((num_nodes,), dtype=torch.float32, device=batch.device)
        n_g = torch.zeros((num_graphs,), dtype=torch.float32, device=batch.device)
        n_g.index_add_(0, batch, ones_n)
        n_per_node = n_g[batch]
        comp = (n_per_node - 1.0 - deg).clamp(min=0.0)

        if node_ids is None:
            global_n_id = torch.arange(num_nodes, device=batch.device, dtype=torch.long)
        else:
            global_n_id = node_ids.to(device=batch.device, dtype=torch.long)
        if global_num_nodes_total is None:
            total_nodes = int(global_n_id.max().item()) + 1 if global_n_id.numel() > 0 else int(num_nodes)
        else:
            total_nodes = int(global_num_nodes_total)

        return BatchedGraphCache(
            edge_index_clean_dir=edge_index_clean_dir,
            batch=batch,
            num_nodes=num_nodes,
            num_graphs=num_graphs,
            deg_clean=deg,
            comp_size=comp,
            n_per_node=n_per_node,
            global_n_id=global_n_id,
            global_num_nodes_total=total_nodes,
        )

    def complement_sum_unweighted(self, x: torch.Tensor) -> torch.Tensor:
        row, col = self.edge_index_clean_dir.to(x.device)
        neigh_sum = _scatter_add_rows(x[row], col, self.num_nodes)
        total_g = _scatter_add_graph_rows(x, self.batch.to(x.device), self.num_graphs)
        total_i = total_g[self.batch.to(x.device)]
        return total_i - x - neigh_sum

    def complement_mean_unweighted(self, x: torch.Tensor) -> torch.Tensor:
        comp_sum = self.complement_sum_unweighted(x)
        denom = self.comp_size.to(x.device).clamp(min=1.0).unsqueeze(1)
        return comp_sum / denom

    def complement_sum_weighted(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        row, col = self.edge_index_clean_dir.to(x.device)
        wx = w.unsqueeze(1) * x
        total_wx = _scatter_add_graph_rows(wx, self.batch.to(x.device), self.num_graphs)[self.batch.to(x.device)]
        neigh_wx = _scatter_add_rows(wx[row], col, self.num_nodes)
        return total_wx - wx - neigh_wx

    def complement_mean_weighted(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        row, col = self.edge_index_clean_dir.to(x.device)
        wx = w.unsqueeze(1) * x
        total_wx = _scatter_add_graph_rows(wx, self.batch.to(x.device), self.num_graphs)[self.batch.to(x.device)]
        total_w = _scatter_add_graph_scalar(w, self.batch.to(x.device), self.num_graphs)[self.batch.to(x.device)]
        neigh_wx = _scatter_add_rows(wx[row], col, self.num_nodes)
        neigh_w = _scatter_add_scalar(w[row], col, self.num_nodes)
        num = total_wx - wx - neigh_wx
        den = (total_w - w - neigh_w).clamp(min=1e-12).unsqueeze(1)
        return num / den


class RADEGATConvGC(_RADEGATConvBase):
    """
    Graph-classification GAT RADE operator.

    Attention moments are computed over each graph's local node set in the
    packed batch. Cross-graph pairs have pi=0 and are excluded from non-edge
    centering and RADE-OFS inference corrections. The sampled mode keeps
    clean-neighbor terms exact and samples graph-local non-neighbors with
    replacement.
    """

    def __init__(self, *args, ep_corr_clip: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ep_corr_clip = max(0.0, float(ep_corr_clip))

    def set_graph(self, graph_cache: BatchedGraphCache) -> None:
        self.graph_cache = graph_cache

    def _same_graph_chunk(self, src_idx: torch.Tensor, device: torch.device) -> torch.Tensor:
        assert self.graph_cache is not None
        batch = self.graph_cache.batch.to(device)
        return batch.unsqueeze(1).eq(batch[src_idx].unsqueeze(0))

    def _attention_logit_shift(
        self,
        score_src: torch.Tensor,
        score_dst: torch.Tensor,
    ) -> torch.Tensor:
        assert self.graph_cache is not None, "Call model.set_graph(...) before graph-classification GAT forward."
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
            local_pair = self._same_graph_chunk(src_idx, score_src.device)
            logits = logits.masked_fill(~local_pair.unsqueeze(-1), -torch.inf)
            logits[src_idx, src_idx - start] = -torch.inf
            shift = torch.maximum(shift, logits.max(dim=1).values)

        return torch.where(torch.isfinite(shift), shift, torch.zeros_like(shift)).detach()

    def _sample_nonedge_sources_with_replacement(
        self,
        num_nodes: int,
        samples: int,
        device: torch.device,
        *,
        seed_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.graph_cache is not None, "Call model.set_graph(...) before graph-classification GAT forward."
        N = int(num_nodes)
        comp = self.graph_cache.comp_size.detach().to(dtype=torch.long, device=torch.device("cpu"))
        max_comp = int(comp.max().item()) if comp.numel() > 0 else 0
        sample_cols = min(max(1, int(samples)), max_comp)
        if N <= 1 or max_comp <= 0:
            src_empty = torch.empty((N, 0), dtype=torch.long, device=device)
            valid_empty = torch.empty((N, 0), dtype=torch.bool, device=device)
            scale_empty = torch.zeros((N,), dtype=torch.float32, device=device)
            return src_empty, valid_empty, scale_empty

        batch_cpu = self.graph_cache.batch.detach().to(dtype=torch.long, device=torch.device("cpu"))
        graph_nodes = [
            torch.nonzero(batch_cpu == graph_idx, as_tuple=False).view(-1)
            for graph_idx in range(int(self.graph_cache.num_graphs))
        ]
        node_pos = {}
        for nodes in graph_nodes:
            for pos, node in enumerate(nodes.tolist()):
                node_pos[int(node)] = int(pos)

        clean_sources = [set() for _ in range(N)]
        edge_row = self.graph_cache.edge_index_clean_dir[0].detach().to(dtype=torch.long, device=torch.device("cpu"))
        edge_col = self.graph_cache.edge_index_clean_dir[1].detach().to(dtype=torch.long, device=torch.device("cpu"))
        for src, dst in zip(edge_row.tolist(), edge_col.tolist()):
            clean_sources[int(dst)].add(int(src))

        target_counts = torch.clamp(comp, max=sample_cols)
        src_out = torch.zeros((N, sample_cols), dtype=torch.long)
        valid_out = torch.zeros((N, sample_cols), dtype=torch.bool)
        counts = torch.zeros((N,), dtype=torch.long)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(self.moment_seed) + int(seed_offset))

        for dst in range(N):
            target = int(target_counts[dst].item())
            if target <= 0:
                continue
            nodes = graph_nodes[int(batch_cpu[dst].item())]
            candidate_count = int(nodes.numel()) - 1
            if candidate_count <= 0:
                continue
            dst_pos = int(node_pos[dst])
            draw = max(sample_cols * 2, 64)
            for _ in range(32):
                if int(counts[dst].item()) >= target:
                    break
                raw = torch.randint(
                    low=0,
                    high=candidate_count,
                    size=(draw,),
                    generator=gen,
                    device=torch.device("cpu"),
                    dtype=torch.long,
                )
                cand = nodes[raw + (raw >= dst_pos).to(torch.long)]
                for src in cand.tolist():
                    if int(src) in clean_sources[dst]:
                        continue
                    pos = int(counts[dst].item())
                    if pos >= target or pos >= sample_cols:
                        break
                    src_out[dst, pos] = int(src)
                    valid_out[dst, pos] = True
                    counts[dst] += 1
                draw = min(max(draw * 2, 64), max(sample_cols * 16, 1024))

        scale = torch.zeros((N,), dtype=torch.float32)
        nonzero = counts > 0
        scale[nonzero] = comp.to(torch.float32)[nonzero] / counts.to(torch.float32)[nonzero]
        return src_out.to(device), valid_out.to(device), scale.to(device)

    def _attention_totals_exact(
        self,
        h: torch.Tensor,
        p: torch.Tensor | float,
        q: torch.Tensor | float,
        *,
        detach_params: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.graph_cache is not None, "Call model.set_graph(...) before graph-classification GAT forward."
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
            src_idx = torch.arange(start, end, device=device)
            logits = torch.nn.functional.leaky_relu(
                score_src[src_idx].unsqueeze(0) + score_dst.unsqueeze(1),
                negative_slope=self.negative_slope,
            )
            same_graph = self._same_graph_chunk(src_idx, device)
            logits = logits.masked_fill(~same_graph.unsqueeze(-1), -torch.inf)
            logits[src_idx, src_idx - start] = -torch.inf
            wij = self._scaled_chunk_attention_weights(logits, logit_shift)

            pi = torch.where(
                same_graph,
                torch.zeros((N, end - start), dtype=dtype, device=device) + q_t,
                torch.zeros((N, end - start), dtype=dtype, device=device),
            )
            in_chunk = (edge_row >= start) & (edge_row < end)
            if bool(in_chunk.any()):
                pi[edge_col[in_chunk], edge_row[in_chunk] - start] = keep_t
            pi[src_idx, src_idx - start] = 0.0

            pi_h = pi.unsqueeze(-1)
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
        assert self.graph_cache is not None, "Call model.set_graph(...) before graph-classification GAT forward."
        N = h.size(0)
        if N <= 1:
            score_src, score_dst = self._node_attention_scores(h, detach_params=detach_params)
            logit_shift = self._attention_logit_shift(score_src, score_dst)
            zeros = torch.zeros((N, self.heads), dtype=h.dtype, device=h.device)
            return zeros, zeros, score_src, score_dst, logit_shift
        max_comp = int(self.graph_cache.comp_size.max().item()) if self.graph_cache.comp_size.numel() > 0 else 0
        if self.moment_samples >= max_comp:
            return self._attention_totals_exact(h, p=p, q=q, detach_params=detach_params)

        dtype = h.dtype
        device = h.device
        p_t = torch.as_tensor(p, dtype=dtype, device=device)
        q_t = torch.as_tensor(q, dtype=dtype, device=device)
        keep_t = 1.0 - p_t

        score_src, score_dst = self._node_attention_scores(h, detach_params=detach_params)
        logit_shift = self._attention_logit_shift(score_src, score_dst)

        edge_row = self.graph_cache.edge_index_clean_dir[0].to(device)
        edge_col = self.graph_cache.edge_index_clean_dir[1].to(device)
        edge_logits = torch.nn.functional.leaky_relu(
            score_src[edge_row] + score_dst[edge_col],
            negative_slope=self.negative_slope,
        )
        edge_w = self._scaled_attention_weights(edge_logits, logit_shift, edge_col)
        total_mu = _scatter_add_rows(keep_t.expand_as(edge_w) * edge_w, edge_col, N)
        total_sigma = _scatter_add_rows((keep_t * p_t).expand_as(edge_w) * edge_w.square(), edge_col, N)

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
        if self.moment_mode == "sampled":
            return self._attention_totals_sampled(h, p=p, q=q, detach_params=detach_params)
        return self._attention_totals_exact(h, p=p, q=q, detach_params=detach_params)

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
        assert self.graph_cache is not None, "Call model.set_graph(...) before graph-classification GAT forward."
        h = self._project(x)
        self._last_h_for_moments = h.detach()
        N = h.size(0)
        use_ofs = self.rade_variant == "rade-ofs"

        if edge_index_keep is None and edge_index_add is None:
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

        row_all = torch.cat([row_k, row_a], dim=0)
        col_all = torch.cat([col_k, col_a], dim=0)
        alpha_all = self._edge_softmax(self._edge_logits(h, row_all, col_all), col_all, N)
        alpha_k = alpha_all[: row_k.numel()]
        alpha_a = alpha_all[row_k.numel():]

        if not self.ep_correction:
            out = self._aggregate(h, row_all, col_all, alpha_all)
            return self._finalize_heads(out)

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
                out_a = self._aggregate(h, row_a, col_a, alpha_a)
            else:
                mu_h, _ = self._nonedge_expectation_sum(h, p=p, q=q, centered=True)
                msg_a = alpha_a.unsqueeze(-1) * (h[row_a] - mu_h[col_a])
                out_a = _scatter_add_heads(msg_a, col_a, N)
        else:
            out_a = torch.zeros_like(out_k)

        return self._finalize_heads(out_k + out_a)

    def gat_denominator_cv2_proxy(
        self,
        *,
        p: torch.Tensor | float,
        q: torch.Tensor | float,
        eps: float,
    ) -> torch.Tensor:
        if self._last_h_for_moments is None:
            if torch.is_tensor(p):
                return p.new_zeros((0,))
            return torch.zeros((0,), dtype=torch.float32)
        if self.heads != 1:
            raise ValueError("GAT denominator stability proxy currently supports heads=1.")
        assert self.graph_cache is not None, "Call model.set_graph(...) before graph-classification GAT forward."

        if torch.is_tensor(p):
            device = p.device
            dtype = p.dtype
        elif torch.is_tensor(q):
            device = q.device
            dtype = q.dtype
        else:
            h_ref = self._last_h_for_moments
            device = h_ref.device
            dtype = h_ref.dtype

        h = self._last_h_for_moments.to(device=device, dtype=dtype)
        total_mu, total_sigma, _, _, _ = self._attention_totals(
            h,
            p=p,
            q=q,
            detach_params=True,
        )
        return total_sigma / total_mu.square().clamp_min(float(eps))

    def gat_correction_gain_proxy(
        self,
        *,
        p: torch.Tensor | float,
        q: torch.Tensor | float,
        eps: float,
    ) -> torch.Tensor:
        if self._last_h_for_moments is None:
            if torch.is_tensor(p):
                return p.new_zeros((0,))
            return torch.zeros((0,), dtype=torch.float32)
        if self.heads != 1:
            raise ValueError("GAT correction-gain proxy currently supports heads=1.")
        assert self.graph_cache is not None, "Call model.set_graph(...) before graph-classification GAT forward."

        if torch.is_tensor(p):
            device = p.device
            dtype = p.dtype
        elif torch.is_tensor(q):
            device = q.device
            dtype = q.dtype
        else:
            h_ref = self._last_h_for_moments
            device = h_ref.device
            dtype = h_ref.dtype

        h = self._last_h_for_moments.to(device=device, dtype=dtype)
        total_mu, total_sigma, score_src, score_dst, logit_shift = self._attention_totals(
            h,
            p=p,
            q=q,
            detach_params=True,
        )
        row = self.graph_cache.edge_index_clean_dir[0].to(device)
        col = self.graph_cache.edge_index_clean_dir[1].to(device)
        if row.numel() == 0:
            return h.new_zeros((0,))

        clean_alpha = self._clean_edge_alpha(h, row, col, detach_params=True)
        p_t = torch.as_tensor(p, dtype=dtype, device=device)
        e_keep, _, _ = self._pair_moments_from_totals(
            row=row,
            col=col,
            pi=1.0 - p_t,
            total_mu=total_mu,
            total_sigma=total_sigma,
            score_src=score_src,
            score_dst=score_dst,
            logit_shift=logit_shift,
        )
        return clean_alpha / e_keep.clamp_min(float(eps))

    def _nonedge_expectation_sum_exact(
        self,
        h: torch.Tensor,
        p: float,
        q: float,
        *,
        centered: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.graph_cache is not None, "Call model.set_graph(...) before graph-classification GAT forward."
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
            pair_mask = self._same_graph_chunk(src_idx, device)
            in_chunk = (edge_row >= start) & (edge_row < end)
            if bool(in_chunk.any()):
                pair_mask[edge_col[in_chunk], edge_row[in_chunk] - start] = False
            pair_mask[src_idx, src_idx - start] = False
            logits = logits.masked_fill(~pair_mask.unsqueeze(-1), -torch.inf)
            wij = self._scaled_chunk_attention_weights(logits, logit_shift)
            pair_mask_f = pair_mask.to(dtype)
            pi_h = wij.new_full(wij.shape, float(q)) * pair_mask_f.unsqueeze(-1)

            mu = total_mu.unsqueeze(1) - pi_h * wij
            mu = mu.clamp_min(0.0)
            sigma = total_sigma.unsqueeze(1) - pi_h * (1.0 - pi_h) * wij.square()
            sigma = sigma.clamp_min(0.0)
            denom = (wij + mu).clamp(min=self.ep_emp_eps)
            e_alpha = pi_h * wij * (denom.reciprocal() + sigma * denom.pow(-3.0))

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
        assert self.graph_cache is not None, "Call model.set_graph(...) before graph-classification GAT forward."
        N = h.size(0)
        if N <= 1:
            num = torch.zeros((N, self.heads, self.out_per_head), dtype=h.dtype, device=h.device)
            den = torch.zeros((N, self.heads), dtype=h.dtype, device=h.device)
            return (num if centered else num), den
        max_comp = int(self.graph_cache.comp_size.max().item()) if self.graph_cache.comp_size.numel() > 0 else 0
        if self.nonedge_samples >= max_comp:
            return self._nonedge_expectation_sum_exact(h, p=p, q=q, centered=centered)

        total_mu, total_sigma, score_src, score_dst, logit_shift = self._attention_totals(h, p=p, q=q)
        dtype = h.dtype
        device = h.device
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

        num = torch.einsum("ish,ishd->ihd", e_alpha, h[src_idx])
        den = e_alpha.sum(dim=1)
        scale_h = scale.to(dtype).view(N, 1)
        num = scale_h.unsqueeze(-1) * num
        den = scale_h * den
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
        assert self.graph_cache is not None, "Call model.set_graph(...) before graph-classification GAT forward."
        N = m.size(0)
        if N <= 1:
            return torch.zeros_like(m_sq)

        device = m.device
        dtype = m.dtype
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

        scale_t = scale.to(dtype).view(N, 1)
        if self.rade_variant == "rade-of":
            den = scale_t.squeeze(1) * e_add.sum(dim=1)
            num = scale_t * (e_add.unsqueeze(-1) * m[src_idx]).sum(dim=1)
            mu_msg = num / den.clamp(min=float(eps)).unsqueeze(1)
            centered_sq = (m[src_idx] - mu_msg.unsqueeze(1)).square()
            return scale_t * (var_add.unsqueeze(-1) * centered_sq).sum(dim=1)

        return scale_t * (var_add.unsqueeze(-1) * m_sq[src_idx]).sum(dim=1)

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
        assert self.graph_cache is not None, "Call model.set_graph(...) before graph-classification GAT forward."

        h = self._last_h_for_moments.to(device=m.device, dtype=m.dtype)
        total_mu, total_sigma, score_src, score_dst, logit_shift = self._attention_totals(
            h,
            p=p,
            q=q,
            detach_params=True,
        )
        row = self.graph_cache.edge_index_clean_dir[0].to(m.device)
        col = self.graph_cache.edge_index_clean_dir[1].to(m.device)

        clean_alpha = self._clean_edge_alpha(h, row, col, detach_params=True).squeeze(-1)
        e_alpha, _, var_alpha = self._pair_moments_from_totals(
            row=row,
            col=col,
            pi=(1.0 - torch.as_tensor(p, dtype=m.dtype, device=m.device)),
            total_mu=total_mu,
            total_sigma=total_sigma,
            score_src=score_src,
            score_dst=score_dst,
            logit_shift=logit_shift,
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
                pair_mask = self._same_graph_chunk(src_idx, m.device)
                in_chunk = (edge_row >= start) & (edge_row < end)
                if bool(in_chunk.any()):
                    pair_mask[edge_col[in_chunk], edge_row[in_chunk] - start] = False
                pair_mask[src_idx, src_idx - start] = False
                logits = logits.masked_fill(~pair_mask.unsqueeze(-1), -torch.inf)
                wij = self._scaled_chunk_attention_weights(logits, logit_shift)
                pi_h = q_t.expand_as(wij) * pair_mask.unsqueeze(-1).to(m.dtype)
                mu_d = total_mu.unsqueeze(1) - pi_h * wij
                mu_d = mu_d.clamp_min(0.0)
                sigma_d = total_sigma.unsqueeze(1) - pi_h * (1.0 - pi_h) * wij.square()
                sigma_d = sigma_d.clamp_min(0.0)
                denom_d = (wij + mu_d).clamp(min=self.ep_emp_eps)
                e_add = pi_h * wij * (denom_d.reciprocal() + sigma_d * denom_d.pow(-3.0))
                e_add = e_add.squeeze(-1)

                den = den + e_add.sum(dim=1)
                num = num + e_add @ m[src_idx]
            mu_msg = num / den.clamp(min=float(eps)).unsqueeze(1)

        for start, end in self._chunk_ranges(N):
            src_idx = torch.arange(start, end, device=m.device)
            logits = torch.nn.functional.leaky_relu(
                score_src[src_idx].unsqueeze(0) + score_dst.unsqueeze(1),
                negative_slope=self.negative_slope,
            )
            pair_mask = self._same_graph_chunk(src_idx, m.device)
            in_chunk = (edge_row >= start) & (edge_row < end)
            if bool(in_chunk.any()):
                pair_mask[edge_col[in_chunk], edge_row[in_chunk] - start] = False
            pair_mask[src_idx, src_idx - start] = False
            logits = logits.masked_fill(~pair_mask.unsqueeze(-1), -torch.inf)
            wij = self._scaled_chunk_attention_weights(logits, logit_shift)
            pi_h = q_t.expand_as(wij) * pair_mask.unsqueeze(-1).to(m.dtype)
            mu_d = total_mu.unsqueeze(1) - pi_h * wij
            mu_d = mu_d.clamp_min(0.0)
            sigma_d = total_sigma.unsqueeze(1) - pi_h * (1.0 - pi_h) * wij.square()
            sigma_d = sigma_d.clamp_min(0.0)
            denom_d = (wij + mu_d).clamp(min=self.ep_emp_eps)
            e_add = pi_h * wij * (denom_d.reciprocal() + sigma_d * denom_d.pow(-3.0))
            e2_add = pi_h * wij.square() * (denom_d.pow(-2.0) + 3.0 * sigma_d * denom_d.pow(-4.0))
            var_add = (e2_add - e_add.square()).clamp_min(0.0).squeeze(-1)

            if self.rade_variant == "rade-of":
                centered_sq = (m[src_idx].unsqueeze(0) - mu_msg.unsqueeze(1)).square()
                add_var = add_var + (var_add.unsqueeze(-1) * centered_sq).sum(dim=1)
            else:
                add_var = add_var + var_add @ m_sq[src_idx]

        return neigh_var + add_var


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


class RADEGCNConvGC(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
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
        self.graph_cache: Optional[BatchedGraphCache] = None

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
        assert self.graph_cache is not None
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
        assert self.graph_cache is not None
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

    def set_graph(self, graph_cache: BatchedGraphCache) -> None:
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

    def _append_sparse_raw(
        self,
        *,
        src_store: str,
        dst_store: str,
        raw_store: str,
        lookup: dict[int, int],
        scale: float,
        src: torch.Tensor,
        dst: torch.Tensor,
        values: torch.Tensor,
        new_mult: float,
    ) -> None:
        if src.numel() == 0:
            return
        src_list = src.detach().cpu().tolist()
        dst_list = dst.detach().cpu().tolist()
        values_detached = values.detach().to(dtype=torch.float32)
        val_list = (float(new_mult) * values_detached / max(scale, 1e-12)).cpu().tolist()
        sq_raw_store = raw_store.replace("_raw", "_sq_raw")
        sq_scale_store = raw_store.replace("_raw", "_sq_scale")
        sq_raw_buf = getattr(self, sq_raw_store, None)
        sq_scale = float(getattr(self, sq_scale_store).item()) if sq_raw_buf is not None else 1.0
        sq_val_list = (float(new_mult) * values_detached.square() / max(sq_scale, 1e-12)).cpu().tolist()

        src_buf = getattr(self, src_store)
        dst_buf = getattr(self, dst_store)
        raw_buf = getattr(self, raw_store)

        new_src = []
        new_dst = []
        new_vals = []
        new_sq_vals = []
        existing_idx = []
        existing_vals = []
        existing_sq_vals = []
        for src_i, dst_i, val_i, sq_val_i in zip(src_list, dst_list, val_list, sq_val_list):
            key = self._edge_key(src_i, dst_i)
            idx = lookup.get(key)
            if idx is None:
                idx = int(raw_buf.numel()) + len(new_src)
                lookup[key] = idx
                new_src.append(int(src_i))
                new_dst.append(int(dst_i))
                new_vals.append(float(val_i))
                new_sq_vals.append(float(sq_val_i))
            else:
                existing_idx.append(int(idx))
                existing_vals.append(float(val_i))
                existing_sq_vals.append(float(sq_val_i))

        if existing_idx:
            idx_t = torch.tensor(existing_idx, dtype=torch.long, device=raw_buf.device)
            val_t = torch.tensor(existing_vals, dtype=torch.float32, device=raw_buf.device)
            raw_buf.index_add_(0, idx_t, val_t)
            if sq_raw_buf is not None:
                sq_val_t = torch.tensor(existing_sq_vals, dtype=torch.float32, device=sq_raw_buf.device)
                sq_raw_buf.index_add_(0, idx_t.to(sq_raw_buf.device), sq_val_t)
        if new_src:
            device = raw_buf.device
            src_buf = torch.cat([src_buf, torch.tensor(new_src, dtype=torch.long, device=device)], dim=0)
            dst_buf = torch.cat([dst_buf, torch.tensor(new_dst, dtype=torch.long, device=device)], dim=0)
            raw_buf = torch.cat([raw_buf, torch.tensor(new_vals, dtype=torch.float32, device=device)], dim=0)
            if sq_raw_buf is not None:
                sq_raw_buf = torch.cat(
                    [sq_raw_buf, torch.tensor(new_sq_vals, dtype=torch.float32, device=sq_raw_buf.device)],
                    dim=0,
                )
            setattr(self, src_store, src_buf)
            setattr(self, dst_store, dst_buf)
        setattr(self, raw_store, raw_buf)
        if sq_raw_buf is not None:
            setattr(self, sq_raw_store, sq_raw_buf)

    def _gcn_sample_coeffs(
        self,
        edge_index_keep: Optional[torch.Tensor],
        edge_index_add: Optional[torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.graph_cache is not None
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

        deg_aug = _scatter_add_scalar(torch.ones(row_all.numel(), device=device, dtype=dtype), col_all, n).clamp(min=1.0)
        inv_sqrt_aug = deg_aug.pow(-0.5)
        gamma_k = inv_sqrt_aug[row_k] * inv_sqrt_aug[col_k]
        gamma_a = inv_sqrt_aug[row_a] * inv_sqrt_aug[col_a] if row_a.numel() else torch.empty(0, device=device, dtype=dtype)
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
        assert self.graph_cache is not None

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
            self._append_sparse_raw(
                src_store="emp_clean_edge_src",
                dst_store="emp_clean_edge_dst",
                raw_store="emp_clean_edge_raw",
                lookup=self._emp_clean_edge_lookup,
                scale=1.0,
                src=src_k,
                dst=dst_k,
                values=gamma_k,
                new_mult=1.0,
            )

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
            self._append_sparse_raw(
                src_store="emp_nonedge_src",
                dst_store="emp_nonedge_dst",
                raw_store="emp_nonedge_raw",
                lookup=self._emp_nonedge_lookup,
                scale=1.0,
                src=src_a,
                dst=dst_a,
                values=gamma_a,
                new_mult=1.0,
            )
            self.emp_num_updates.add_(1)
            return

        self.emp_clean_edge_scale.mul_(old_mult)
        self.emp_clean_edge_sq_scale.mul_(old_mult)
        self._append_sparse_raw(
            src_store="emp_clean_edge_src",
            dst_store="emp_clean_edge_dst",
            raw_store="emp_clean_edge_raw",
            lookup=self._emp_clean_edge_lookup,
            scale=float(self.emp_clean_edge_scale.item()),
            src=src_k,
            dst=dst_k,
            values=gamma_k,
            new_mult=new_mult,
        )

        self.emp_self_scale.mul_(old_mult)
        self.emp_self_sq_scale.mul_(old_mult)
        gamma_self_detached = gamma_self.detach().to(self.emp_self_raw.device, dtype=torch.float32)
        upd_self = (new_mult * gamma_self_detached) / self.emp_self_scale.clamp_min(1e-12)
        self.emp_self_raw.index_add_(0, global_ids.to(self.emp_self_raw.device, dtype=torch.long), upd_self)
        upd_self_sq = (
            new_mult * gamma_self_detached.square().to(self.emp_self_sq_raw.device)
        ) / self.emp_self_sq_scale.clamp_min(1e-12)
        self.emp_self_sq_raw.index_add_(0, global_ids.to(self.emp_self_sq_raw.device, dtype=torch.long), upd_self_sq)

        self.emp_nonedge_scale.mul_(old_mult)
        self.emp_nonedge_sq_scale.mul_(old_mult)
        self._append_sparse_raw(
            src_store="emp_nonedge_src",
            dst_store="emp_nonedge_dst",
            raw_store="emp_nonedge_raw",
            lookup=self._emp_nonedge_lookup,
            scale=float(self.emp_nonedge_scale.item()),
            src=src_a,
            dst=dst_a,
            values=gamma_a,
            new_mult=new_mult,
        )
        self.emp_num_updates.add_(1)

    def _get_sparse_emp(
        self,
        src_store: str,
        dst_store: str,
        raw_store: str,
        scale_store: str,
        lookup: dict[int, int],
        row: torch.Tensor,
        col: torch.Tensor,
        *,
        fallback: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if row.numel() == 0:
            return fallback.to(device=device, dtype=dtype)
        values = fallback.detach().to(device=device, dtype=dtype).clone()
        global_ids = self.graph_cache.global_n_id.to(device)
        row_global = global_ids[row].detach().cpu().tolist()
        col_global = global_ids[col].detach().cpu().tolist()
        found_idx = []
        target_pos = []
        for pos, (src_i, dst_i) in enumerate(zip(row_global, col_global)):
            idx = lookup.get(self._edge_key(src_i, dst_i))
            if idx is not None:
                found_idx.append(int(idx))
                target_pos.append(int(pos))
        if found_idx:
            idx_t = torch.tensor(found_idx, dtype=torch.long, device=device)
            pos_t = torch.tensor(target_pos, dtype=torch.long, device=device)
            stored = getattr(self, raw_store).to(device=device, dtype=dtype)[idx_t]
            scale = getattr(self, scale_store).to(device=device, dtype=dtype)
            values[pos_t] = stored * scale
        return values

    def _apply_emp_nonedge_expectation(self, x: torch.Tensor) -> torch.Tensor:
        if self.emp_nonedge_raw.numel() == 0:
            return torch.zeros((x.size(0), x.size(1)), device=x.device, dtype=x.dtype)
        assert self.graph_cache is not None

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
        msg: torch.Tensor,
        msg_sq: torch.Tensor,
        *,
        p: float,
        q: float,
        eps: float,
    ) -> Optional[torch.Tensor]:
        if self.ep_expectation_mode != "empirical_ema":
            return None
        assert self.graph_cache is not None

        device = msg_sq.device
        dtype = msg_sq.dtype
        num_nodes = int(self.graph_cache.num_nodes)
        global_ids = self.graph_cache.global_n_id.detach().cpu().tolist()
        local_pos = {int(global_id): idx for idx, global_id in enumerate(global_ids)}
        total = int(self.graph_cache.global_num_nodes_total)

        row, col = self.graph_cache.edge_index_clean_dir
        found = []
        target_pos = []
        for pos, (src_l, dst_l) in enumerate(zip(row.detach().cpu().tolist(), col.detach().cpu().tolist())):
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
        neigh_var = _scatter_add_rows(factor.unsqueeze(1) * msg_sq[row_d], col_d, num_nodes)

        if self.emp_nonedge_raw.numel() == 0:
            return neigh_var

        local_src = []
        local_dst = []
        sparse_idx = []
        for idx, (src_g, dst_g) in enumerate(
            zip(self.emp_nonedge_src.detach().cpu().tolist(), self.emp_nonedge_dst.detach().cpu().tolist())
        ):
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
        add_var = _scatter_add_rows(var_a.unsqueeze(1) * msg_sq[row_a], col_a, num_nodes)
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
        assert self.graph_cache is not None, "Call model.set_graph(...) per batch before forward."
        x = self.lin(x)
        num_nodes = x.size(0)

        if edge_index_keep is None and edge_index_add is None:
            row, col = edge_index_clean[0], edge_index_clean[1]
            ar = torch.arange(num_nodes, device=x.device)
            row_all = torch.cat([row, ar])
            col_all = torch.cat([col, ar])
            deg = _scatter_add_scalar(torch.ones(row_all.numel(), device=x.device, dtype=x.dtype), col_all, num_nodes)
            inv_sqrt = deg.clamp(min=1.0).pow(-0.5)
            gamma = inv_sqrt[row_all] * inv_sqrt[col_all]
            out = _scatter_add_rows(gamma.unsqueeze(1) * x[row_all], col_all, num_nodes)

            if self.rade_variant == "rade-ofs" and (not self.training) and self.ep_correction and (q > 0.0):
                if self.ep_expectation_mode == "empirical_ema":
                    out = out + self._apply_emp_nonedge_expectation(x)
                else:
                    deg_clean = self.graph_cache.deg_clean.to(x.device)
                    comp_size = self.graph_cache.comp_size.to(x.device)
                    t_b = _delta_E_inv_sqrt_caseB(deg_clean, comp_size, p=p, q=q).to(x.device, dtype=x.dtype)
                    comp_sum = self.graph_cache.complement_sum_weighted(x, w=t_b)
                    out = out + (q * t_b).unsqueeze(1) * comp_sum

            if self.bias is not None:
                out = out + self.bias
            return out

        if edge_index_keep is None:
            row_k, col_k = self.graph_cache.edge_index_clean_dir.to(x.device)
        else:
            row_k, col_k = edge_index_keep[0], edge_index_keep[1]
        if edge_index_add is not None and edge_index_add.numel() > 0:
            row_a, col_a = edge_index_add[0], edge_index_add[1]
        else:
            row_a = row_k.new_empty((0,))
            col_a = col_k.new_empty((0,))

        ar = torch.arange(num_nodes, device=x.device)
        row_all = torch.cat([row_k, row_a, ar])
        col_all = torch.cat([col_k, col_a, ar])
        deg_aug = _scatter_add_scalar(torch.ones(row_all.numel(), device=x.device, dtype=x.dtype), col_all, num_nodes).clamp(min=1.0)
        inv_sqrt_aug = deg_aug.pow(-0.5)

        gamma_k = inv_sqrt_aug[row_k] * inv_sqrt_aug[col_k]
        gamma_a = inv_sqrt_aug[row_a] * inv_sqrt_aug[col_a] if row_a.numel() else None
        gamma_self = inv_sqrt_aug * inv_sqrt_aug

        deg_clean = self.graph_cache.deg_clean.to(x.device)
        comp_size = self.graph_cache.comp_size.to(x.device)
        deg_hat = (deg_clean + 1.0).clamp(min=1.0)
        inv_sqrt_hat = deg_hat.pow(-0.5)
        alpha_k = inv_sqrt_hat[row_k] * inv_sqrt_hat[col_k]
        alpha_self = inv_sqrt_hat * inv_sqrt_hat

        if self.ep_correction:
            if self.ep_expectation_mode == "empirical_ema":
                e_gamma_k = self._get_sparse_emp(
                    "emp_clean_edge_src",
                    "emp_clean_edge_dst",
                    "emp_clean_edge_raw",
                    "emp_clean_edge_scale",
                    self._emp_clean_edge_lookup,
                    row_k,
                    col_k,
                    fallback=gamma_k.detach().to(x.device, dtype=x.dtype),
                    dtype=x.dtype,
                    device=x.device,
                )
                if self.emp_num_updates.item() > 0:
                    e_self = self.emp_self_raw.to(x.device, dtype=x.dtype)[self.graph_cache.global_n_id.to(x.device)] * self.emp_self_scale.to(x.device, dtype=x.dtype)
                    missing_self = e_self <= 0.0
                    if bool(missing_self.any()):
                        e_self[missing_self] = gamma_self.detach().to(x.device, dtype=x.dtype)[missing_self]
                else:
                    e_self = gamma_self.detach().to(x.device, dtype=x.dtype)
            else:
                t_a = _delta_E_inv_sqrt_caseA(deg_clean, comp_size, p=p, q=q)
                e_gamma_k = ((1.0 - p) * t_a[row_k] * t_a[col_k]).to(x.dtype)
                e_self = _delta_E_inv_self(deg_clean, comp_size, p=p, q=q).to(x.device, dtype=x.dtype)

            corr_k = alpha_k / e_gamma_k.clamp(min=self.ep_emp_eps)
            out_k = _scatter_add_rows((gamma_k * corr_k).unsqueeze(1) * x[row_k], col_k, num_nodes)
            corr_self = alpha_self / e_self.clamp(min=self.ep_emp_eps)
            out_self = (gamma_self * corr_self).unsqueeze(1) * x if self.correct_self_loop else gamma_self.unsqueeze(1) * x
        else:
            out_k = _scatter_add_rows(gamma_k.unsqueeze(1) * x[row_k], col_k, num_nodes)
            out_self = gamma_self.unsqueeze(1) * x

        if row_a.numel():
            sum_a = _scatter_add_rows(gamma_a.unsqueeze(1) * x[row_a], col_a, num_nodes)
            if self.rade_variant == "rade-ofs":
                out_a = sum_a
            elif self.ep_correction and self.ep_expectation_mode == "empirical_ema":
                out_a = sum_a - self._apply_emp_nonedge_expectation(x)
            else:
                t_b = _delta_E_inv_sqrt_caseB(deg_clean, comp_size, p=p, q=q).to(x.device, dtype=x.dtype)
                mu = self.graph_cache.complement_mean_weighted(x, w=t_b)
                sum_g = _scatter_add_scalar(gamma_a, col_a, num_nodes).unsqueeze(1)
                out_a = sum_a - mu * sum_g
        else:
            out_a = torch.zeros_like(out_k)

        out = out_k + out_a + out_self
        if self.bias is not None:
            out = out + self.bias
        return out

class RADEGINConvGC(nn.Module):
    def __init__(
        self,
        mlp: nn.Module,
        *,
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
        self.graph_cache: Optional[BatchedGraphCache] = None

        self.register_buffer("emp_keep_prob", torch.empty(0, dtype=torch.float32))
        self.register_buffer("emp_add_prob", torch.empty(0, dtype=torch.float32))
        self.register_buffer("emp_num_updates", torch.tensor(0, dtype=torch.long))

    def set_graph(self, graph_cache: BatchedGraphCache) -> None:
        self.graph_cache = graph_cache

    def reset_parameters(self) -> None:
        reset(self.mlp)
        self.eps.data.fill_(self.initial_eps)

    def _ensure_emp_prob_storage(self) -> None:
        assert self.graph_cache is not None
        total = int(self.graph_cache.global_num_nodes_total)
        device = self.eps.device
        if self.emp_keep_prob.numel() != total:
            self.emp_keep_prob = torch.zeros((total,), dtype=torch.float32, device=device)
        if self.emp_add_prob.numel() != total:
            self.emp_add_prob = torch.zeros((total,), dtype=torch.float32, device=device)

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
        if self.graph_cache is not None:
            self._ensure_emp_prob_storage()
            self.emp_keep_prob.zero_()
            self.emp_add_prob.zero_()
        else:
            self.emp_keep_prob = torch.empty(0, dtype=torch.float32, device=self.eps.device)
            self.emp_add_prob = torch.empty(0, dtype=torch.float32, device=self.eps.device)
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
        assert self.graph_cache is not None
        self._ensure_emp_prob_storage()
        old_mult, new_mult = self._emp_update_weights()
        global_ids = self.graph_cache.global_n_id.to(self.emp_keep_prob.device)
        keep_prob = torch.zeros((self.graph_cache.num_nodes,), dtype=torch.float32, device=self.emp_keep_prob.device)
        add_prob = torch.zeros((self.graph_cache.num_nodes,), dtype=torch.float32, device=self.emp_add_prob.device)

        if edge_index_keep is None and edge_index_add is not None:
            keep_prob.fill_(1.0)
        elif edge_index_keep is not None and edge_index_keep.numel() > 0:
            keep_deg = _scatter_add_scalar(
                torch.ones(edge_index_keep.size(1), device=keep_prob.device, dtype=torch.float32),
                edge_index_keep[1].to(keep_prob.device),
                self.graph_cache.num_nodes,
            )
            denom = self.graph_cache.deg_clean.to(keep_prob.device).clamp(min=1.0)
            keep_prob = keep_deg / denom

        if edge_index_add is not None and edge_index_add.numel() > 0:
            add_deg = _scatter_add_scalar(
                torch.ones(edge_index_add.size(1), device=add_prob.device, dtype=torch.float32),
                edge_index_add[1].to(add_prob.device),
                self.graph_cache.num_nodes,
            )
            denom = self.graph_cache.comp_size.to(add_prob.device).clamp(min=1.0)
            add_prob = add_deg / denom

        if old_mult <= 1e-12:
            self.emp_keep_prob.zero_()
            self.emp_add_prob.zero_()
        else:
            self.emp_keep_prob.mul_(old_mult)
            self.emp_add_prob.mul_(old_mult)

        self.emp_keep_prob.index_add_(0, global_ids, new_mult * keep_prob)
        self.emp_add_prob.index_add_(0, global_ids, new_mult * add_prob)
        self.emp_num_updates.add_(1)

    def _get_keep_prob(self, p: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        assert self.graph_cache is not None
        if self.ep_expectation_mode == "empirical_ema" and self.emp_num_updates.item() > 0:
            self._ensure_emp_prob_storage()
            return self.emp_keep_prob.to(device=device, dtype=dtype)[self.graph_cache.global_n_id.to(device)].clamp(min=self.ep_emp_eps)
        return torch.full((self.graph_cache.num_nodes,), 1.0 - float(p), device=device, dtype=dtype)

    def _get_add_prob(self, q: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        assert self.graph_cache is not None
        if self.ep_expectation_mode == "empirical_ema" and self.emp_num_updates.item() > 0:
            self._ensure_emp_prob_storage()
            return self.emp_add_prob.to(device=device, dtype=dtype)[self.graph_cache.global_n_id.to(device)].clamp(min=0.0)
        return torch.full((self.graph_cache.num_nodes,), float(q), device=device, dtype=dtype)

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
        num_nodes = x.size(0)

        if edge_index_keep is None and edge_index_add is None:
            row, col = edge_index_clean[0], edge_index_clean[1]
            neigh_sum = _scatter_add_rows(x[row], col, num_nodes)
            if self.rade_variant == "rade-ofs" and (not self.training) and self.ep_correction and (q > 0.0):
                add_prob = self._get_add_prob(q=float(q), device=x.device, dtype=x.dtype)
                comp_sum = self.graph_cache.complement_sum_unweighted(x)
                neigh_sum = neigh_sum + add_prob.unsqueeze(1) * comp_sum
            out = (1.0 + self.eps.to(x.device)) * x + neigh_sum
            return self.mlp(out)

        if edge_index_keep is None:
            row_k, col_k = self.graph_cache.edge_index_clean_dir.to(x.device)
        else:
            row_k, col_k = edge_index_keep[0], edge_index_keep[1]
        if edge_index_add is not None and edge_index_add.numel() > 0:
            row_a, col_a = edge_index_add[0], edge_index_add[1]
        else:
            row_a = row_k.new_empty((0,))
            col_a = row_k.new_empty((0,))

        if self.ep_correction:
            keep_prob = self._get_keep_prob(p=float(p), device=x.device, dtype=x.dtype)
            keep_scale = keep_prob[col_k].clamp(min=self.ep_emp_eps)
            keep_sum = _scatter_add_rows((x[row_k] / keep_scale.unsqueeze(1)), col_k, num_nodes)
        else:
            alpha = 1.0 / max(1e-12, 1.0 - float(p))
            keep_sum = _scatter_add_rows(x[row_k], col_k, num_nodes) * alpha

        if row_a.numel() > 0:
            add_sum_raw = _scatter_add_rows(x[row_a], col_a, num_nodes)
            if self.rade_variant == "rade-ofs":
                add_sum = add_sum_raw
            else:
                mu = self.graph_cache.complement_mean_unweighted(x)
                if self.ep_correction:
                    add_prob = self._get_add_prob(q=float(q), device=x.device, dtype=x.dtype)
                    add_deg = _scatter_add_scalar(add_prob[row_a], col_a, num_nodes).unsqueeze(1)
                else:
                    add_deg = _scatter_add_scalar(
                        torch.ones(row_a.numel(), device=x.device, dtype=x.dtype),
                        col_a,
                        num_nodes,
                    ).unsqueeze(1)
                add_sum = add_sum_raw - mu * add_deg
        else:
            add_sum = torch.zeros_like(keep_sum)

        out = (1.0 + self.eps.to(x.device)) * x + keep_sum + add_sum
        return self.mlp(out)
