# nc_minibatch/mb_pq_gradnorm.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple, List, Iterable, Any, Optional

import torch
import torch.nn as nn

from .mb_rade_convs import (
    BatchGraphCache,
    _scatter_add_rows,
    _delta_E_inv_sqrt_caseA,
    _delta_E_inv_sqrt_caseB,
    _delta_E_inv_caseA,
    _delta_E_inv_caseB,
)


# -------------------------
# small helpers (same logic as full_batch)
# -------------------------
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


def _pick_subset(idx: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    # k<=0 => full idx
    if k <= 0 or k >= idx.numel():
        return idx
    g = torch.Generator(device=idx.device)
    g.manual_seed(int(seed))
    perm = torch.randperm(idx.numel(), generator=g, device=idx.device)
    return idx[perm[:k]]


def _w_multiclass(logits: torch.Tensor) -> torch.Tensor:
    prob = torch.softmax(logits, dim=-1)
    return prob * (1.0 - prob)


def _freeze_batchnorm_buffers(model: nn.Module) -> List[Tuple[nn.Module, bool]]:
    """
    Put BN modules into eval() so running_mean/var buffers are not updated during tuning,
    while keeping the overall model in train() (so dropout stays on).
    Returns list of (bn_module, was_training) to restore later.
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

def _num_non_edges_undirected_from_dir_edges(num_nodes: int, edge_index_dir: torch.Tensor) -> int:
    """
    Approximate #undirected non-edges in the induced batch graph.

    If your NeighborLoader batch edge_index is NOT symmetrized, this will be off. In that case,
    either symmetrize before calling the tuner or replace this with a unique-undirected counter.
    """
    m_dir = int(edge_index_dir.size(1))
    m_undir = m_dir // 2
    n = int(num_nodes)
    return n * (n - 1) // 2 - m_undir

# -------------------------
# config
# -------------------------
@dataclass
class PQTunerMBConfig:
    """
    Mini-batch PQ GradNorm tuner config.

    Notes:
    - subset_nodes is applied over *batch train nodes* (i.e., within each induced subgraph).
    - epoch_ema is applied after epoch-level aggregation (avg over batches).
    """
    p_max: float = 0.8
    q_max: float = 1e-3
    grid_size: int = 11
    epoch_ema: float = 0.9
    eps: float = 1e-12
    subset_nodes: int = 0
    seed: int = 0

    # If set, you can tune on only the first K batches per epoch for speed (0 => all batches).
    max_batches_for_tuning: int = 0


class PQGradNormTunerMB:
    """
    Mini-batch PQ-GradNorm tuner implementing your epoch-level aggregation rule:

      For each batch b:
        (p_b, q_b) = argmin_{p,q in grid} ( log||∇ L_data(B_b)|| - log||∇ R(p,q;B_b)|| )^2
      Then:
        (p^{t+1}, q^{t+1}) = EMA( (mean_b p_b, mean_b q_b), (p^t, q^t) )

    IMPORTANT SEMANTICS:
    - This tuner is *batch-local*: it builds a GraphCache on the induced subgraph and computes
      Var(δ_i) w.r.t. the batch graph (local complements). This matches your mini_batch RADE
      semantics when mb_rade_mode="local".
    """

    def __init__(self, gnn: str, variant: str, cfg: PQTunerMBConfig):
        self.cfg = cfg
        self.gnn = str(gnn).lower().strip()         # 'gin' or 'gcn'
        self.variant = str(variant).lower().strip() # 'rade' or 'rade-ic'

    @torch.no_grad()
    def _messages_detached(self, emb: torch.Tensor, W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = emb.detach()
        W = W.detach()
        m = h @ W.t()  # [B, C]
        return m, m * m

    def _var_gin_bases(self, cache: BatchGraphCache, m: torch.Tensor, m_sq: torch.Tensor, B: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row, col = cache.edge_index_clean_dir
        neigh_sum_sq = _scatter_add_rows(m_sq[row], col, B)                # [B,C]
        comp_sum_sq = cache.complement_sum_unweighted(m_sq)                # [B,C]

        if self.variant == "rade":
            comp_sum_m = cache.complement_sum_unweighted(m)                # [B,C]
            comp_size = cache.comp_size.to(m.dtype).clamp_min(1.0).unsqueeze(1)
            mu = comp_sum_m / comp_size
            add_centered = comp_sum_sq - 2.0 * mu * comp_sum_m + comp_size * (mu * mu)
            return neigh_sum_sq, add_centered

        # rade-ic
        return neigh_sum_sq, comp_sum_sq

    @torch.no_grad()
    def _var_gcn(self, cache: BatchGraphCache, p: float, q: float, m: torch.Tensor, m_sq: torch.Tensor, B: int) -> torch.Tensor:
        eps = self.cfg.eps
        row, col = cache.edge_index_clean_dir

        deg = cache.deg_clean.to(torch.float32)
        comp = cache.comp_size.to(torch.float32)
        d_hat = (deg + 1.0).clamp_min(1.0)

        # --- edges (Case A) ---
        tA = _delta_E_inv_sqrt_caseA(deg, comp, p=p, q=q)
        uA = _delta_E_inv_caseA(deg, comp, p=p, q=q, eps=eps)

        E = (1.0 - p) * (tA[row] * tA[col])
        E2 = (1.0 - p) * (uA[row] * uA[col])
        VarG = (E2 - E * E).clamp_min(0.0)

        alpha_sq = 1.0 / (d_hat[row] * d_hat[col])          # alpha^2
        factor = alpha_sq * (VarG / (E * E + eps))          # per-directed-edge
        neigh_var = _scatter_add_rows(factor.unsqueeze(1) * m_sq[row], col, B)

        # --- non-edges (Case B) ---
        tB = _delta_E_inv_sqrt_caseB(deg, comp, p=p, q=q)
        uB = _delta_E_inv_caseB(deg, comp, p=p, q=q, eps=eps)
        tB2 = tB * tB

        if self.variant == "rade":
            # mu weights ∝ tB[j]
            mu = cache.complement_mean_weighted(m, tB)

            ones = torch.ones((B, 1), device=m.device, dtype=m.dtype)

            # S1 = sum uB[j] (m_j - mu_i)^2
            A_u = cache.complement_sum_weighted(m_sq, uB)
            B_u = cache.complement_sum_weighted(m, uB)
            C_u = cache.complement_sum_weighted(ones, uB)
            S1 = A_u - 2.0 * mu * B_u + (mu * mu) * C_u

            # S2 = sum tB[j]^2 (m_j - mu_i)^2
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

    def suggest_pq_for_batch(
        self,
        model: nn.Module,
        x_b: torch.Tensor,
        edge_index_b_clean: torch.Tensor,
        y_b: torch.Tensor,
        train_mask_b: torch.Tensor,
        criterion,
        current_p: float,
        current_q: float,
        aug_mode: str,
        epoch_seed: int,
        batch_id: int,
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Returns (p_best, q_best) for this batch (NOT EMA-smoothed), plus info.
        """
        cfg = self.cfg
        eps = cfg.eps
        # -------------------------
        # Ensure device consistency (tuner may receive CPU batches from NeighborLoader)
        # -------------------------
        dev = next(model.parameters()).device

        x_b = x_b.to(dev)
        edge_index_b_clean = edge_index_b_clean.to(dev, dtype=torch.long)

        # y must be on same device as logits for criterion(logits[idx], y[idx])
        y_b = y_b.to(dev)
        if y_b.dim() == 2 and y_b.size(1) == 1:
            y_b = y_b.squeeze(1)
        y_b = y_b.view(-1).to(torch.long)

        train_mask_b = train_mask_b.to(dev, dtype=torch.bool)


        # local batch indices of training nodes
        train_mask_b = train_mask_b.to(torch.bool)
        idx_all = torch.where(train_mask_b)[0]
        if idx_all.numel() == 0:
            return 0.0, 0.0, {"skip": 1.0}

        # optional subset for speed
        idx = _pick_subset(idx_all, cfg.subset_nodes, seed=int(epoch_seed) + int(batch_id) + int(cfg.seed))

        # We want training-mode gradients (dropout consistent), but we must NOT update BN running stats.
        was_training = model.training
        model.train()
        bn_states = _freeze_batchnorm_buffers(model)

        devices = []
        if x_b.is_cuda and x_b.device.index is not None:
            devices = [x_b.device.index]

        B = int(x_b.size(0))
        cache = BatchGraphCache.from_edge_index(edge_index_b_clean, B)

        with torch.random.fork_rng(devices=devices, enabled=True):
            # isolate RNG so tuning doesn't consume training RNG stream
            s = int(epoch_seed) + 1000003 * int(batch_id) + int(cfg.seed)
            torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)

            model.zero_grad(set_to_none=True)

            # clean forward (no RADE masks; no inference correction)
            logits, emb = model(x_b, edge_index_b_clean, p=0.0, q=0.0, return_embeddings=True)

            # data grad norm (batch-local)
            loss_data = criterion(logits[idx], y_b[idx])
            params = [p for p in model.parameters() if p.requires_grad]
            g_data = torch.autograd.grad(loss_data, params, retain_graph=True, create_graph=False, allow_unused=True)
            G_data = _grad_norm(g_data)
            logGd = math.log(G_data + eps)

            # detached message stats (Var detached)
            W = model.pred_local.weight
            m, m_sq = self._messages_detached(emb, W)

            # z(1-z)-like weights (multiclass), differentiable
            w = _w_multiclass(logits)

            # bounds / grids (respect aug_mode)
            aug_mode = str(aug_mode).lower().strip()
            p_max = cfg.p_max if aug_mode in ("both", "drop") else 0.0
            q_max = cfg.q_max if aug_mode in ("both", "add") else 0.0

            p_grid = [0.0] if p_max <= 0 else torch.linspace(
                0.0, float(p_max), steps=int(cfg.grid_size), device=x_b.device
            ).tolist()
            # --- K_add-based q grid (prevents "q floor" / fixed added edges) ---
            if q_max <= 0.0:
                q_grid = [0.0]
            else:
                # number of undirected non-edges in the induced batch graph
                M_undir = max(1, _num_non_edges_undirected_from_dir_edges(B, edge_index_b_clean))
                K_max = int(round(float(q_max) * float(M_undir)))  # max expected undirected additions at q_max

                if K_max <= 0:
                    q_grid = [0.0]
                else:
                    # Always include tiny K so q can go below q_max/(grid_size-1)
                    K_core = [0, 1, 2, 5, 10, 20, 50]
                    K_core = [k for k in K_core if k <= K_max]

                    target = int(cfg.grid_size)
                    remaining = max(0, target - len(K_core))

                    K_lin = []
                    if remaining > 0:
                        lin = torch.linspace(0.0, float(K_max), steps=remaining + 2, device="cpu")
                        K_lin = [int(round(x)) for x in lin.tolist()[1:-1]]

                    K_list = sorted(set(K_core + K_lin))[:target]
                    q_grid = [float(k) / float(M_undir) for k in K_list]

            best_obj = float("inf")
            best_p, best_q = 0.0, 0.0
            best_Greg = 0.0

            if self.gnn == "gin":
                V_drop, V_add = self._var_gin_bases(cache, m, m_sq, B)  # detached bases

                R_drop = 0.5 * (w[idx] * V_drop[idx]).sum() / max(int(idx.numel()), 1)
                R_add = 0.5 * (w[idx] * V_add[idx]).sum() / max(int(idx.numel()), 1)

                g_drop = torch.autograd.grad(R_drop, params, retain_graph=True, create_graph=False, allow_unused=True)
                g_add = torch.autograd.grad(R_add, params, retain_graph=True, create_graph=False, allow_unused=True)

                a = _dot(g_drop, g_drop)
                b = _dot(g_add, g_add)
                c = _dot(g_drop, g_add)

                for p_try in p_grid:
                    s_fac = p_try / max(1.0 - p_try, eps)   # p/(1-p)
                    for q_try in q_grid:
                        t_fac = q_try * (1.0 - q_try)       # q(1-q)
                        g2 = (s_fac * s_fac) * a + (t_fac * t_fac) * b + 2.0 * s_fac * t_fac * c
                        Greg = math.sqrt(max(g2, 0.0))
                        obj = (math.log(Greg + eps) - logGd) ** 2
                        if obj < best_obj:
                            best_obj, best_p, best_q, best_Greg = obj, float(p_try), float(q_try), float(Greg)

            elif self.gnn == "gcn":
                for p_try in p_grid:
                    for q_try in q_grid:
                        V = self._var_gcn(cache, float(p_try), float(q_try), m, m_sq, B)  # detached
                        R = 0.5 * (w[idx] * V[idx]).sum() / max(int(idx.numel()), 1)

                        g_reg = torch.autograd.grad(R, params, retain_graph=True, create_graph=False, allow_unused=True)
                        Greg = _grad_norm(g_reg)

                        obj = (math.log(Greg + eps) - logGd) ** 2
                        if obj < best_obj:
                            best_obj, best_p, best_q, best_Greg = obj, float(p_try), float(q_try), float(Greg)
            else:
                raise ValueError(f"Unsupported gnn={self.gnn}. Expected 'gin' or 'gcn'.")

        _restore_batchnorm_buffers(bn_states)
        if not was_training:
            model.eval()

        # enforce aug_mode (hard constraints)
        if aug_mode == "drop":
            best_q = 0.0
        elif aug_mode == "add":
            best_p = 0.0
        elif aug_mode == "none":
            best_p, best_q = 0.0, 0.0

        # clamp
        best_p = float(max(0.0, min(best_p, cfg.p_max)))
        best_q = float(max(0.0, min(best_q, cfg.q_max)))

        info = {
            "G_data": float(G_data),
            "G_reg_best": float(best_Greg),
            "p_best": float(best_p),
            "q_best": float(best_q),
            "obj": float(best_obj),
            "batch_train_nodes": float(idx_all.numel()),
            "batch_train_used": float(idx.numel()),
        }
        return best_p, best_q, info

    def suggest_pq_for_epoch(
        self,
        model: nn.Module,
        batches: Iterable[Any],
        criterion,
        current_p: float,
        current_q: float,
        aug_mode: str,
        epoch_seed: int,
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Epoch-level aggregation:
          - compute (p_b, q_b) per batch
          - average across batches (weighted by #train nodes in that batch)
          - EMA with current_p/current_q
        """
        cfg = self.cfg
        aug_mode = str(aug_mode).lower().strip()

        was_training = model.training
        model.train()
        bn_states = _freeze_batchnorm_buffers(model)

        sum_w = 0.0
        sum_p = 0.0
        sum_q = 0.0

        Gd_list: List[float] = []
        Gr_list: List[float] = []
        obj_list: List[float] = []
        used_batches = 0
        skipped_batches = 0

        max_batches = int(cfg.max_batches_for_tuning)
        for b_id, batch in enumerate(batches):
            if max_batches > 0 and used_batches >= max_batches:
                break

            # Accept:
            #  (x, edge_index, y, train_mask) OR dict OR PyG NeighborLoader batch (Data-like).
            if isinstance(batch, (tuple, list)) and len(batch) >= 4:
                x_b, edge_index_b, y_b, train_mask_b = batch[0], batch[1], batch[2], batch[3]

            elif isinstance(batch, dict):
                x_b = batch["x"]
                edge_index_b = batch["edge_index"]
                y_b = batch["y"]
                train_mask_b = batch["train_mask"]

            elif hasattr(batch, "x") and hasattr(batch, "edge_index") and hasattr(batch, "y"):
                # NeighborLoader batch: seed nodes are the first `batch.batch_size` nodes.
                x_b = batch.x
                edge_index_b = batch.edge_index
                y_b = batch.y

                seed_n = int(getattr(batch, "batch_size", x_b.size(0)))
                train_mask_b = torch.zeros((x_b.size(0),), device=x_b.device, dtype=torch.bool)
                train_mask_b[:seed_n] = True

            else:
                raise ValueError(
                    "Each batch must be (x, edge_index, y, train_mask), a dict with keys "
                    "x/edge_index/y/train_mask, or a PyG Data-like batch with attributes x/edge_index/y "
                    "(e.g., NeighborLoader batch)."
                )


            # criterion expects y shape consistent with logits:
            # - if y_b is [B,1], squeeze to [B]
            if y_b.dim() == 2 and y_b.size(1) == 1:
                y_b_use = y_b.squeeze(1)
            else:
                y_b_use = y_b

            p_b, q_b, info_b = self.suggest_pq_for_batch(
                model=model,
                x_b=x_b,
                edge_index_b_clean=edge_index_b,
                y_b=y_b_use,
                train_mask_b=train_mask_b,
                criterion=criterion,
                current_p=current_p,
                current_q=current_q,
                aug_mode=aug_mode,
                epoch_seed=epoch_seed,
                batch_id=int(b_id),
            )

            if "skip" in info_b:
                skipped_batches += 1
                continue

            w_b = float(info_b.get("batch_train_nodes", 0.0))
            if w_b <= 0.0:
                skipped_batches += 1
                continue

            sum_w += w_b
            sum_p += w_b * float(p_b)
            sum_q += w_b * float(q_b)

            Gd_list.append(float(info_b["G_data"]))
            Gr_list.append(float(info_b["G_reg_best"]))
            obj_list.append(float(info_b["obj"]))

            used_batches += 1

        _restore_batchnorm_buffers(bn_states)
        if not was_training:
            model.eval()

        if sum_w <= 0.0 or used_batches == 0:
            # no usable batches -> keep current
            p_avg, q_avg = float(current_p), float(current_q)
        else:
            p_avg = sum_p / sum_w
            q_avg = sum_q / sum_w

        # EMA update at epoch level
        p_new = cfg.epoch_ema * float(current_p) + (1.0 - cfg.epoch_ema) * float(p_avg)
        q_new = cfg.epoch_ema * float(current_q) + (1.0 - cfg.epoch_ema) * float(q_avg)

        # enforce aug_mode
        if aug_mode == "drop":
            q_new = 0.0
        elif aug_mode == "add":
            p_new = 0.0
        elif aug_mode == "none":
            p_new, q_new = 0.0, 0.0

        # clamp
        p_new = float(max(0.0, min(p_new, cfg.p_max)))
        q_new = float(max(0.0, min(q_new, cfg.q_max)))

        def _mean(xs: List[float]) -> float:
            return float(sum(xs) / max(len(xs), 1))

        info = {
            "p_epoch_avg_best": float(p_avg),
            "q_epoch_avg_best": float(q_avg),
            "p_new": float(p_new),
            "q_new": float(q_new),
            "used_batches": float(used_batches),
            "skipped_batches": float(skipped_batches),
            "G_data_mean": _mean(Gd_list),
            "G_reg_best_mean": _mean(Gr_list),
            "obj_mean": _mean(obj_list),
        }
        return p_new, q_new, info
