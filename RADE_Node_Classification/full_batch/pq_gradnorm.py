# pq_gradnorm.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
from augmentation import BernoulliEdgeAugmentor

from rade_convs import (
    GraphCache,
    _scatter_add_rows,
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


def _pick_subset(idx: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    # k==0 means: use full idx (full_batch setting)
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
    while the overall model remains in train() (so dropout stays on if you use it).
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


@dataclass
class PQTunerConfig:
    p_max: float = 0.8
    q_max: float = 1e-3
    grid_size: int = 11
    ema: float = 0.9
    eps: float = 1e-12
    subset_nodes: int = 0   # 0 => full train_idx
    seed: int = 0
    data_anchor: str = "clean"


class PQGradNormTuner:
    """
    Epoch-wise GradNorm matcher for RADE / RADE-IC.

    - GIN: fast (two base gradients).
    - GCN: grid (each candidate does one autograd.grad on R).
    """

    def __init__(self, edge_index_clean: torch.Tensor, num_nodes: int, gnn: str, variant: str, cfg: PQTunerConfig):
        self.cfg = cfg
        self.gnn = str(gnn).lower().strip()         # 'gin' or 'gcn'
        self.variant = str(variant).lower().strip() # 'rade' or 'rade-ic'
        self.cache = GraphCache.from_edge_index(edge_index_clean, int(num_nodes))
        self.N = int(num_nodes)

        self.edge_index_clean = edge_index_clean
        self.augmentor = BernoulliEdgeAugmentor.from_edge_index(edge_index_clean, num_nodes=self.N)

        da = str(getattr(cfg, "data_anchor", "clean")).lower().strip()
        if da not in {"clean", "aug"}:
            raise ValueError(f"PQTunerConfig.data_anchor must be 'clean' or 'aug', got {da}")
        self.data_anchor = da

    def _num_non_edges_undirected(self) -> int:
        # Assumes edge_index_clean_dir contains both directions for each undirected edge.
        m_dir = int(self.cache.edge_index_clean_dir.size(1))
        m_undir = m_dir // 2
        n = int(self.N)
        return n * (n - 1) // 2 - m_undir

    @torch.no_grad()
    def _messages_detached(self, emb: torch.Tensor, W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = emb.detach()
        W = W.detach()
        m = h @ W.t()      # [N,C]
        return m, m * m

    def _var_gin_bases(self, m: torch.Tensor, m_sq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        row, col = self.cache.edge_index_clean_dir
        neigh_sum_sq = _scatter_add_rows(m_sq[row], col, self.N)  # [N,C]
        comp_sum_sq = self.cache.complement_sum_unweighted(m_sq)  # [N,C]

        if self.variant == "rade":
            comp_sum_m = self.cache.complement_sum_unweighted(m)  # [N,C]
            comp_size = self.cache.comp_size.to(m.dtype).clamp_min(1.0).unsqueeze(1)
            mu = comp_sum_m / comp_size
            add_centered = comp_sum_sq - 2.0 * mu * comp_sum_m + comp_size * (mu * mu)
            return neigh_sum_sq, add_centered

        # rade-ic
        return neigh_sum_sq, comp_sum_sq

    @torch.no_grad()
    def _var_gcn(self, p: float, q: float, m: torch.Tensor, m_sq: torch.Tensor) -> torch.Tensor:
        eps = self.cfg.eps
        row, col = self.cache.edge_index_clean_dir

        deg = self.cache.deg_clean.to(torch.float32)
        comp = self.cache.comp_size.to(torch.float32)
        d_hat = (deg + 1.0).clamp_min(1.0)

        # --- edges (Case A) ---
        tA = _delta_E_inv_sqrt_caseA(deg, comp, p=p, q=q)
        uA = _delta_E_inv_caseA(deg, comp, p=p, q=q, eps=eps)

        E = (1.0 - p) * (tA[row] * tA[col])
        E2 = (1.0 - p) * (uA[row] * uA[col])
        VarG = (E2 - E * E).clamp_min(0.0)

        alpha_sq = 1.0 / (d_hat[row] * d_hat[col])  # alpha^2
        factor = alpha_sq * (VarG / (E * E + eps))  # per-directed-edge
        neigh_var = _scatter_add_rows(factor.unsqueeze(1) * m_sq[row], col, self.N)

        # --- non-edges (Case B) ---
        tB = _delta_E_inv_sqrt_caseB(deg, comp, p=p, q=q)
        uB = _delta_E_inv_caseB(deg, comp, p=p, q=q, eps=eps)
        tB2 = tB * tB

        if self.variant == "rade":
            # mu weights ∝ tB[j]
            mu = self.cache.complement_mean_weighted(m, tB)

            ones = torch.ones((self.N, 1), device=m.device, dtype=m.dtype)

            # S1 = sum uB[j] (m_j - mu_i)^2
            A_u = self.cache.complement_sum_weighted(m_sq, uB)
            B_u = self.cache.complement_sum_weighted(m, uB)
            C_u = self.cache.complement_sum_weighted(ones, uB)
            S1 = A_u - 2.0 * mu * B_u + (mu * mu) * C_u

            # S2 = sum tB[j]^2 (m_j - mu_i)^2
            A_t = self.cache.complement_sum_weighted(m_sq, tB2)
            B_t = self.cache.complement_sum_weighted(m, tB2)
            C_t = self.cache.complement_sum_weighted(ones, tB2)
            S2 = A_t - 2.0 * mu * B_t + (mu * mu) * C_t
        else:
            # rade-ic: uncentered
            S1 = self.cache.complement_sum_weighted(m_sq, uB)
            S2 = self.cache.complement_sum_weighted(m_sq, tB2)

        add_var = (q * uB.unsqueeze(1) * S1) - ((q * q) * tB2.unsqueeze(1) * S2)

        return neigh_var + add_var

    def suggest_pq(
        self,
        model,
        data,
        train_idx: torch.Tensor,
        criterion,
        current_p: float,
        current_q: float,
        aug_mode: str,
        epoch_seed: int,
        mask_sharing: str = "shared",
        num_layers: int = 1,
    ) -> Tuple[float, float, Dict[str, float]]:
        cfg = self.cfg
        eps = cfg.eps

        # (Optional) subset for speed; set subset_nodes=0 for full train_idx
        idx = _pick_subset(train_idx, cfg.subset_nodes, seed=epoch_seed)

        # We want training-mode gradients (dropout consistent), but we must NOT update BN running stats.
        was_training = model.training
        model.train()
        bn_states = _freeze_batchnorm_buffers(model)

        # Isolate RNG usage so this tuning step does not consume global RNG state.
        # Works for CPU and CUDA.
        devices = []
        if data.x.is_cuda and data.x.device.index is not None:
            devices = [data.x.device.index]

        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.manual_seed(int(epoch_seed) + int(cfg.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(epoch_seed) + int(cfg.seed))

            # Ensure clean grads for safety (autograd.grad does not write .grad, but keep this explicit)
            model.zero_grad(set_to_none=True)

            # clean forward (no RADE masks, no inference correction)
            # -------------------------
            # 1) Anchor data gradient norm (clean or aug)
            # -------------------------
            def _sample_masks_for_anchor(p0: float, q0: float):
                mode0 = str(aug_mode).lower().strip()
                ms = str(mask_sharing).lower().strip()
                L = max(1, int(num_layers))

                if mode0 == "none" or (p0 <= 0.0 and q0 <= 0.0):
                    return None, None

                base_seed = int(epoch_seed) + 777_000  # disjoint from other RNG uses inside this fork

                if ms == "shared":
                    _, ei_keep, ei_add, _ = self.augmentor.augment_edge_indices(
                        p=float(p0),
                        q=float(q0),
                        mode=mode0,
                        seed=base_seed,
                        device=data.x.device,
                    )
                    return ei_keep, ei_add

                if ms == "layerwise":
                    keep_list, add_list = [], []
                    for layer in range(L):
                        _, ei_keep_l, ei_add_l, _ = self.augmentor.augment_edge_indices(
                            p=float(p0),
                            q=float(q0),
                            mode=mode0,
                            seed=base_seed + 1_000_000 * layer,
                            device=data.x.device,
                        )
                        keep_list.append(ei_keep_l)
                        add_list.append(ei_add_l)
                    return keep_list, add_list

                raise ValueError(f"mask_sharing must be 'shared' or 'layerwise', got {ms}")

            model.zero_grad(set_to_none=True)

            if self.data_anchor == "aug":
                # Use the SAME kind of stochastic training forward that SGD sees (one draw of masks),
                # anchored at the CURRENT (p,q), not the candidate (p_try,q_try).
                ei_keep0, ei_add0 = _sample_masks_for_anchor(float(current_p), float(current_q))

                if ei_keep0 is None and ei_add0 is None:
                    logits_anchor = model(data.x, data.edge_index, p=0.0, q=0.0)
                else:
                    logits_anchor = model(
                        data.x,
                        data.edge_index,
                        edge_index_keep=ei_keep0,
                        edge_index_add=ei_add0,
                        p=float(current_p),
                        q=float(current_q),
                    )
            else:
                # Clean anchor
                logits_anchor = model(data.x, data.edge_index, p=0.0, q=0.0)

            loss_data = criterion(logits_anchor[idx], data.y[idx])
            params = [p for p in model.parameters() if p.requires_grad]
            g_data = torch.autograd.grad(loss_data, params, retain_graph=False, create_graph=False, allow_unused=True)
            G_data = _grad_norm(g_data)
            logGd = math.log(G_data + eps)

            # -------------------------
            # 2) Snapshot for regularizer proxy (keep existing convention)
            #    Always compute from a clean forward (theoretical reference),
            #    while Var remains detached.
            # -------------------------
            model.zero_grad(set_to_none=True)
            logits, emb = model(data.x, data.edge_index, p=0.0, q=0.0, return_embeddings=True)

            W = model.pred_local.weight
            m, m_sq = self._messages_detached(emb, W)

            # z(1-z)-like weights (multiclass); differentiable; Var stays detached
            w = _w_multiclass(logits)


            # bounds / grids (respect aug_mode)
            aug_mode = str(aug_mode).lower().strip()
            p_max = cfg.p_max if aug_mode in ("both", "drop") else 0.0
            q_max = cfg.q_max if aug_mode in ("both", "add") else 0.0

            p_grid = [0.0] if p_max <= 0 else torch.linspace(
                0.0, float(p_max), steps=int(cfg.grid_size), device=data.x.device
            ).tolist()

            # --- REPLACE q_grid linspace with a K_add-based grid (prevents "q floor" / fixed added edges) ---
            if q_max <= 0.0:
                q_grid = [0.0]
            else:
                M_undir = max(1, self._num_non_edges_undirected())  # number of undirected non-edges
                K_max = int(round(float(q_max) * float(M_undir)))   # max expected undirected additions under q_max

                if K_max <= 0:
                    q_grid = [0.0]
                else:
                    # Always include very small expected-add counts so q can go below q_max/(grid_size-1)
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
                V_drop, V_add = self._var_gin_bases(m, m_sq)  # detached bases

                R_drop = 0.5 * (w[idx] * V_drop[idx]).sum() / max(int(idx.numel()), 1)
                R_add = 0.5 * (w[idx] * V_add[idx]).sum() / max(int(idx.numel()), 1)

                g_drop = torch.autograd.grad(R_drop, params, retain_graph=True, create_graph=False, allow_unused=True)
                g_add = torch.autograd.grad(R_add, params, retain_graph=True, create_graph=False, allow_unused=True)

                a = _dot(g_drop, g_drop)
                b = _dot(g_add, g_add)
                c = _dot(g_drop, g_add)

                for p_try in p_grid:
                    s = p_try / max(1.0 - p_try, eps)   # p/(1-p)
                    for q_try in q_grid:
                        t = q_try * (1.0 - q_try)       # q(1-q)
                        g2 = (s * s) * a + (t * t) * b + 2.0 * s * t * c
                        Greg = math.sqrt(max(g2, 0.0))
                        obj = (math.log(Greg + eps) - logGd) ** 2
                        if obj < best_obj:
                            best_obj, best_p, best_q, best_Greg = obj, float(p_try), float(q_try), float(Greg)

            elif self.gnn == "gcn":
                for p_try in p_grid:
                    for q_try in q_grid:
                        V = self._var_gcn(float(p_try), float(q_try), m, m_sq)  # detached
                        R = 0.5 * (w[idx] * V[idx]).sum() / max(int(idx.numel()), 1)

                        g_reg = torch.autograd.grad(R, params, retain_graph=True, create_graph=False, allow_unused=True)
                        Greg = _grad_norm(g_reg)

                        obj = (math.log(Greg + eps) - logGd) ** 2
                        if obj < best_obj:
                            best_obj, best_p, best_q, best_Greg = obj, float(p_try), float(q_try), float(Greg)
            else:
                raise ValueError(f"Unsupported gnn={self.gnn}. Expected 'gin' or 'gcn'.")

        # Restore BN module states
        _restore_batchnorm_buffers(bn_states)

        # Restore model mode
        if not was_training:
            model.eval()

        # EMA update
        p_new = cfg.ema * float(current_p) + (1.0 - cfg.ema) * float(best_p)
        q_new = cfg.ema * float(current_q) + (1.0 - cfg.ema) * float(best_q)

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

        info = {
            "G_data": float(G_data),
            "G_reg_best": float(best_Greg),
            "p_best": float(best_p),
            "q_best": float(best_q),
            "p_new": float(p_new),
            "q_new": float(q_new),
            "obj": float(best_obj),
        }
        return p_new, q_new, info
