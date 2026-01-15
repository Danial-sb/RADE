# graph_classification/pq_gradnorm_gc.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from rade_convs_gc import (
    _scatter_add_rows,
    _scatter_add_scalar,
    _delta_E_inv_sqrt_caseA,
    _delta_E_inv_sqrt_caseB,
    _delta_E_inv_caseA,
    _delta_E_inv_caseB,
)


def _grad_norm(grads) -> float:
    s = 0.0
    for g in grads:
        if g is None:
            continue
        s += float((g * g).sum().item())
    return math.sqrt(max(s, 0.0))


def _dot(grads_a, grads_b) -> float:
    s = 0.0
    for ga, gb in zip(grads_a, grads_b):
        if ga is None or gb is None:
            continue
        s += float((ga * gb).sum().item())
    return s


def _freeze_batchnorm_buffers(model: nn.Module) -> List[Tuple[nn.Module, bool]]:
    """
    Put BN modules into eval() so running_mean/var buffers are not updated during tuning,
    while the overall model remains in train() (dropout stays on if you use it).
    Returns (bn_module, was_training) to restore later.
    """
    states = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            states.append((m, m.training))
            m.eval()
    return states


def _restore_batchnorm_buffers(states: List[Tuple[nn.Module, bool]]) -> None:
    for m, was_training in states:
        m.train(was_training)

def _num_non_edges_undirected_total(cache: BatchGraphCache) -> int:
    """
    Total #undirected non-edges across graphs in this disjoint-union batch.

    Then:
      M_total = sum_g [ C(n_g,2) - m_g ]
    where sum_g C(n_g,2) is computed from cache.nodes_per_graph.
    """
    n_per_g = cache.nodes_per_graph  # [num_graphs] float
    total_pairs = (n_per_g * (n_per_g - 1.0) * 0.5).sum().item()

    m_dir = int(cache.edge_index_clean_dir.size(1))
    m_undir = m_dir // 2

    M_total = int(round(total_pairs)) - int(m_undir)
    return max(1, M_total)


class BatchGraphCache:
    """
    GraphCache analogue for a PyG Batch (disjoint union).
    Crucially: complement sums are computed *within each graph* using the `batch` vector.
    """

    def __init__(self, edge_index_clean: torch.Tensor, batch: torch.Tensor, num_graphs: int):
        edge_index_clean = edge_index_clean.to(torch.long)
        row, col = edge_index_clean[0], edge_index_clean[1]

        # remove self-loops (they are handled separately in theory; here complements exclude self)
        mask = row != col
        row, col = row[mask], col[mask]
        self.edge_index_clean_dir = torch.stack([row, col], dim=0)

        self.batch = batch.to(torch.long)
        self.num_graphs = int(num_graphs)
        self.num_nodes = int(batch.numel())

        # in-degree for directed aggregation form (j -> i contributes to i)
        ones_e = torch.ones(col.numel(), device=col.device, dtype=torch.float32)
        deg = torch.zeros((self.num_nodes,), device=col.device, dtype=torch.float32)
        deg.index_add_(0, col, ones_e)
        self.deg_clean = deg  # [N]

        # nodes per graph
        ones_n = torch.ones((self.num_nodes,), device=batch.device, dtype=torch.float32)
        n_per_g = torch.zeros((self.num_graphs,), device=batch.device, dtype=torch.float32)
        n_per_g.index_add_(0, self.batch, ones_n)
        self.nodes_per_graph = n_per_g  # [B]

        # complement size within graph: |V_g| - 1 - deg(i)
        self.comp_size = (n_per_g[self.batch] - 1.0 - deg).clamp(min=0.0)  # [N]

    def _graph_sum_rows(self, x: torch.Tensor) -> torch.Tensor:
        B = self.num_graphs
        out = torch.zeros((B, x.size(1)), device=x.device, dtype=x.dtype)
        out.index_add_(0, self.batch.to(x.device), x)
        return out

    def _graph_sum_scalar(self, x: torch.Tensor) -> torch.Tensor:
        B = self.num_graphs
        out = torch.zeros((B,), device=x.device, dtype=x.dtype)
        out.index_add_(0, self.batch.to(x.device), x)
        return out

    def complement_sum_unweighted(self, x: torch.Tensor) -> torch.Tensor:
        """
        sum_{j in C(i)} x_j   within the same graph as i
        """
        N = self.num_nodes
        row, col = self.edge_index_clean_dir.to(x.device)
        neigh_sum = _scatter_add_rows(x[row], col, N)  # sum_{j in N(i)} x_j

        graph_sum = self._graph_sum_rows(x)  # [B,F]
        total_sum = graph_sum[self.batch.to(x.device)]  # [N,F]

        return total_sum - x - neigh_sum

    def complement_sum_weighted(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        sum_{j in C(i)} w_j x_j   within-graph
        """
        N = self.num_nodes
        row, col = self.edge_index_clean_dir.to(x.device)
        w = w.to(x.device)
        wx = w.unsqueeze(1) * x

        neigh_wx = _scatter_add_rows(wx[row], col, N)
        graph_wx = self._graph_sum_rows(wx)
        total_wx = graph_wx[self.batch.to(x.device)]

        return total_wx - wx - neigh_wx

    def complement_mean_weighted(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        mu_i = (sum_{j in C(i)} w_j x_j) / (sum_{j in C(i)} w_j)
        """
        N = self.num_nodes
        row, col = self.edge_index_clean_dir.to(x.device)
        w = w.to(x.device)
        wx = w.unsqueeze(1) * x

        neigh_wx = _scatter_add_rows(wx[row], col, N)
        neigh_w = _scatter_add_scalar(w[row], col, N)

        graph_wx = self._graph_sum_rows(wx)
        graph_w = self._graph_sum_scalar(w)

        total_wx = graph_wx[self.batch.to(x.device)]
        total_w = graph_w[self.batch.to(x.device)]

        num = total_wx - wx - neigh_wx
        den = (total_w - w - neigh_w).clamp(min=1e-12).unsqueeze(1)
        return num / den


@dataclass
class PQTunerGCConfig:
    p_max: float = 0.8
    q_max: float = 1e-3
    grid_size: int = 11
    eps: float = 1e-12
    seed: int = 0


class PQGradNormTunerGC:
    """
    Batch-wise PQ-GradNorm matcher for graph classification.

    - For each mini_batch: find (p_b, q_b) by matching ||∇L_data|| and ||∇R(p,q)||.
    - Caller should average batch-wise solutions across the epoch to obtain (p^{t+1}, q^{t+1}).

    Notes:
    - Assumes your GC model exposes:
        * model.pred.weight  (graph-level prediction head)
        * forward(..., return_embeddings=True) -> (logits_graph [B,C], node_emb [N,H])
    - Uses within-graph complements via BatchGraphCache (correct for disjoint union batching).
    """

    def __init__(self, *, gnn: str, variant: str, pooling: str, cfg: PQTunerGCConfig):
        self.cfg = cfg
        self.gnn = str(gnn).lower().strip()         # 'gin' or 'gcn'
        self.variant = str(variant).lower().strip() # 'rade' or 'rade-ic'
        self.pooling = str(pooling).lower().strip() # 'sum' or 'mean'

        if self.gnn not in {"gin", "gcn"}:
            raise ValueError("gnn must be 'gin' or 'gcn'")
        if self.variant not in {"rade", "rade-ic"}:
            raise ValueError("variant must be 'rade' or 'rade-ic'")
        if self.pooling not in {"sum", "mean"}:
            raise ValueError("pooling must be 'sum' or 'mean'")

    @torch.no_grad()
    def _messages_detached(self, emb: torch.Tensor, W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Node "logit contributions":
          m_i = W h_i  (per node, per class/task)
        Treated as deterministic for variance computation (detached).
        """
        h = emb.detach()
        W = W.detach()
        m = h @ W.t()  # [N,C]
        return m, m * m

    def _loss_and_w(self, logits: torch.Tensor, y: torch.Tensor, loss_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          loss_data (scalar),
          w_graph   (same shape as logits)
        """
        loss_type = str(loss_type).lower().strip()
        if loss_type == "ce":
            if y.dim() == 2 and y.size(1) == 1:
                y = y.squeeze(1)
            y = y.view(-1).to(torch.long)
            loss = F.cross_entropy(logits, y)
            prob = torch.softmax(logits, dim=-1)
            w = prob * (1.0 - prob)
            return loss, w

        if loss_type == "bce":
            if y.dim() == 1:
                y = y.view(-1, 1)
            y_f = y.to(torch.float32)
            mask = ((y == 0) | (y == 1)).to(torch.float32)  # OGB-style missing labels
            loss_el = F.binary_cross_entropy_with_logits(logits, y_f, reduction="none")
            loss = (loss_el * mask).sum() / mask.sum().clamp(min=1.0)

            z = torch.sigmoid(logits)
            w = z * (1.0 - z) * mask
            return loss, w

        raise ValueError(f"Unsupported loss_type={loss_type}. Use 'ce' or 'bce'.")

    def _pool_w2(self, batch_vec: torch.Tensor, num_graphs: int, num_nodes: int, device) -> torch.Tensor:
        """
        Squared pooling weights per node:
          sum pooling: a_i = 1
          mean pooling: a_i = 1/|V_g|
        We use a_i^2 for variance aggregation.
        """
        if self.pooling == "sum":
            return torch.ones((num_nodes,), device=device, dtype=torch.float32)

        ones = torch.ones((num_nodes,), device=device, dtype=torch.float32)
        n_per_g = torch.zeros((num_graphs,), device=device, dtype=torch.float32)
        n_per_g.index_add_(0, batch_vec, ones)
        return 1.0 / n_per_g[batch_vec].clamp(min=1.0).pow(2.0)

    def _var_gin_bases(self, cache: BatchGraphCache, m: torch.Tensor, m_sq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Per-node variance bases (detached):
          V_drop(i) = sum_{j in N(i)} m_j^2
          V_add(i)  = sum_{j in C(i)} (...)^2
        """
        row, col = cache.edge_index_clean_dir
        N = cache.num_nodes

        neigh_sum_sq = _scatter_add_rows(m_sq[row], col, N)        # [N,C]
        comp_sum_sq = cache.complement_sum_unweighted(m_sq)        # [N,C]

        if self.variant == "rade":
            comp_sum_m = cache.complement_sum_unweighted(m)        # [N,C]
            comp_size = cache.comp_size.to(m.dtype).clamp_min(1.0).unsqueeze(1)
            mu = comp_sum_m / comp_size
            add_centered = comp_sum_sq - 2.0 * mu * comp_sum_m + comp_size * (mu * mu)
            return neigh_sum_sq, add_centered

        # rade-ic: uncentered
        return neigh_sum_sq, comp_sum_sq

    @torch.no_grad()
    def _var_gcn(self, cache: BatchGraphCache, p: float, q: float, m: torch.Tensor, m_sq: torch.Tensor) -> torch.Tensor:
        """
        Per-node variance for GCN symmetric normalization via your mixed-binomial/delta approximations.
        Returned tensor is detached.
        """
        eps = self.cfg.eps
        row, col = cache.edge_index_clean_dir
        N = cache.num_nodes

        deg = cache.deg_clean.to(torch.float32)
        comp = cache.comp_size.to(torch.float32)
        d_hat = (deg + 1.0).clamp_min(1.0)

        # --- edges (Case A) ---
        tA = _delta_E_inv_sqrt_caseA(deg, comp, p=p, q=q)
        uA = _delta_E_inv_caseA(deg, comp, p=p, q=q, eps=eps)

        E = (1.0 - p) * (tA[row] * tA[col])
        E2 = (1.0 - p) * (uA[row] * uA[col])
        VarG = (E2 - E * E).clamp_min(0.0)

        alpha_sq = 1.0 / (d_hat[row] * d_hat[col])  # alpha^2
        factor = alpha_sq * (VarG / (E * E + eps))  # per directed edge
        neigh_var = _scatter_add_rows(factor.unsqueeze(1) * m_sq[row], col, N)

        # --- non-edges (Case B) ---
        tB = _delta_E_inv_sqrt_caseB(deg, comp, p=p, q=q)
        uB = _delta_E_inv_caseB(deg, comp, p=p, q=q, eps=eps)
        tB2 = tB * tB

        if self.variant == "rade":
            mu = cache.complement_mean_weighted(m, tB)

            ones = torch.ones((N, 1), device=m.device, dtype=m.dtype)

            # S1 = sum uB[j] (m_j - mu_i)^2 over C(i)
            A_u = cache.complement_sum_weighted(m_sq, uB)
            B_u = cache.complement_sum_weighted(m, uB)
            C_u = cache.complement_sum_weighted(ones, uB)
            S1 = A_u - 2.0 * mu * B_u + (mu * mu) * C_u

            # S2 = sum tB[j]^2 (m_j - mu_i)^2 over C(i)
            A_t = cache.complement_sum_weighted(m_sq, tB2)
            B_t = cache.complement_sum_weighted(m, tB2)
            C_t = cache.complement_sum_weighted(ones, tB2)
            S2 = A_t - 2.0 * mu * B_t + (mu * mu) * C_t
        else:
            # rade-ic: uncentered
            S1 = cache.complement_sum_weighted(m_sq, uB)
            S2 = cache.complement_sum_weighted(m_sq, tB2)

        add_var = (q * uB.unsqueeze(1) * S1) - ((q * q) * tB2.unsqueeze(1) * S2)
        return neigh_var + add_var

    def suggest_pq_batch(
        self,
        *,
        model: nn.Module,
        batch,
        loss_type: str,
        aug_mode: str,
        p_max: float,
        q_max: float,
        seed: int,
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Returns (p_best, q_best, info) for THIS mini_batch.

        Caller should average (p_best, q_best) across batches to obtain epoch-level (p^{t+1}, q^{t+1}).
        """
        cfg = self.cfg
        eps = cfg.eps
        aug_mode = str(aug_mode).lower().strip()

        x = batch.x
        edge_index = batch.edge_index
        batch_vec = batch.batch
        num_graphs = int(batch.num_graphs)

        # train-mode grads but do not update BN buffers
        was_training = model.training
        model.train()
        bn_states = _freeze_batchnorm_buffers(model)

        devices = []
        if x.is_cuda and x.device.index is not None:
            devices = [x.device.index]

        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.manual_seed(int(seed) + int(cfg.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed) + int(cfg.seed))

            model.zero_grad(set_to_none=True)

            edge_attr = getattr(batch, "edge_attr", None)

            # clean forward for GradNorm matching (no RADE masks here)
            logits, node_emb = model(
                x, edge_index, batch_vec, edge_attr=edge_attr, return_embeddings=True
            )

            loss_data, w_graph = self._loss_and_w(logits, batch.y, loss_type=loss_type)

            params = [p for p in model.parameters() if p.requires_grad]
            g_data = torch.autograd.grad(loss_data, params, retain_graph=True, create_graph=False, allow_unused=True)
            G_data = _grad_norm(g_data)
            logGd = math.log(G_data + eps)

            # detached message stats
            W = model.pred.weight
            m, m_sq = self._messages_detached(node_emb, W)

            cache = BatchGraphCache(edge_index, batch_vec, num_graphs)

            pool_w2 = self._pool_w2(batch_vec, num_graphs, cache.num_nodes, device=m.device).to(m.dtype).unsqueeze(1)

            # grids respecting aug_mode
            p_max_eff = p_max if aug_mode in {"both", "drop"} else 0.0
            q_max_eff = q_max if aug_mode in {"both", "add"} else 0.0

            p_grid = [0.0] if p_max_eff <= 0 else torch.linspace(
                0.0, float(p_max_eff), steps=int(cfg.grid_size), device=m.device
            ).tolist()
            # --- K_add-based q grid (prevents "q floor" / fixed added edges) ---
            if q_max_eff <= 0.0:
                q_grid = [0.0]
            else:
                M_total = _num_non_edges_undirected_total(cache)  # total undirected non-edges across graphs in batch
                K_max = int(round(float(q_max_eff) * float(M_total)))

                if K_max <= 0:
                    q_grid = [0.0]
                else:
                    K_core = [0, 1, 2, 5, 10, 20, 50]
                    K_core = [k for k in K_core if k <= K_max]

                    target = int(cfg.grid_size)
                    remaining = max(0, target - len(K_core))

                    K_lin = []
                    if remaining > 0:
                        lin = torch.linspace(0.0, float(K_max), steps=remaining + 2, device="cpu")
                        K_lin = [int(round(x)) for x in lin.tolist()[1:-1]]

                    K_list = sorted(set(K_core + K_lin))[:target]
                    q_grid = [float(k) / float(M_total) for k in K_list]

            best_obj = float("inf")
            best_p, best_q, best_Greg = 0.0, 0.0, 0.0
            B = max(1, num_graphs)

            if self.gnn == "gin":
                # detached bases
                V_drop_node, V_add_node = self._var_gin_bases(cache, m, m_sq)

                # graph-level variance ≈ sum_{i in graph} a_i^2 Var(delta_i)
                V_drop_g = torch.zeros((num_graphs, V_drop_node.size(1)), device=m.device, dtype=m.dtype)
                V_add_g = torch.zeros_like(V_drop_g)
                V_drop_g.index_add_(0, batch_vec, V_drop_node * pool_w2)
                V_add_g.index_add_(0, batch_vec, V_add_node * pool_w2)

                # base regularizers (differentiable through w_graph/logits; variance stays detached)
                R_drop = 0.5 * (w_graph * V_drop_g).sum() / B
                R_add = 0.5 * (w_graph * V_add_g).sum() / B

                g_drop = torch.autograd.grad(R_drop, params, retain_graph=True, create_graph=False, allow_unused=True)
                g_add = torch.autograd.grad(R_add, params, retain_graph=True, create_graph=False, allow_unused=True)

                a = _dot(g_drop, g_drop)
                b = _dot(g_add, g_add)
                c = _dot(g_drop, g_add)

                for p_try in p_grid:
                    s = float(p_try) / max(1.0 - float(p_try), eps)  # p/(1-p)
                    for q_try in q_grid:
                        t = float(q_try) * (1.0 - float(q_try))        # q(1-q)
                        g2 = (s * s) * a + (t * t) * b + 2.0 * s * t * c
                        Greg = math.sqrt(max(g2, 0.0))
                        obj = (math.log(Greg + eps) - logGd) ** 2
                        if obj < best_obj:
                            best_obj, best_p, best_q, best_Greg = obj, float(p_try), float(q_try), float(Greg)

            else:  # gcn
                for p_try in p_grid:
                    for q_try in q_grid:
                        V_node = self._var_gcn(cache, float(p_try), float(q_try), m, m_sq)
                        V_g = torch.zeros((num_graphs, V_node.size(1)), device=m.device, dtype=m.dtype)
                        V_g.index_add_(0, batch_vec, V_node * pool_w2)

                        R = 0.5 * (w_graph * V_g).sum() / B
                        g_reg = torch.autograd.grad(R, params, retain_graph=True, create_graph=False, allow_unused=True)
                        Greg = _grad_norm(g_reg)

                        obj = (math.log(Greg + eps) - logGd) ** 2
                        if obj < best_obj:
                            best_obj, best_p, best_q, best_Greg = obj, float(p_try), float(q_try), float(Greg)

        _restore_batchnorm_buffers(bn_states)
        if not was_training:
            model.eval()

        # enforce aug_mode
        if aug_mode == "drop":
            best_q = 0.0
        elif aug_mode == "add":
            best_p = 0.0
        elif aug_mode == "none":
            best_p, best_q = 0.0, 0.0

        info = {
            "G_data": float(G_data),
            "G_reg_best": float(best_Greg),
            "p_best": float(best_p),
            "q_best": float(best_q),
            "obj": float(best_obj),
        }
        return float(best_p), float(best_q), info
