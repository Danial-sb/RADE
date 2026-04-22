from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, Callable, Optional

import numpy as np
import torch
import torch.nn as nn

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover
    minimize = None

from augmentation import BernoulliEdgeAugmentor
from rade_convs import (
    GraphCache,
    _scatter_add_rows,
    _delta_E_inv_sqrt_caseA,
    _delta_E_inv_sqrt_caseB,
    _delta_E_inv_caseA,
    _delta_E_inv_caseB,
)


def _validate_rade_variant(rade_variant: str) -> str:
    rade_variant = str(rade_variant).lower().strip()
    if rade_variant not in {"rade-of", "rade-ofs"}:
        raise ValueError(
            f"rade_variant must be one of {{'rade-of', 'rade-ofs'}}. Got {rade_variant}"
        )
    return rade_variant


def _validate_search_method(search_method: str) -> str:
    search_method = str(search_method).lower().strip()
    if search_method not in {"grid", "powell", "newton", "adam"}:
        raise ValueError(
            f"search_method must be one of {{'grid', 'powell', 'newton', 'adam'}}. Got {search_method}"
        )
    return search_method


# Keep SEARCH_METHODS for the derivative-free / comparison family only.
# Adam is stateful and should not be mixed into the frozen-snapshot side-by-side comparison utilities.
SEARCH_METHODS: Tuple[str, ...] = ("grid", "powell", "newton")


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
    # k == 0 means: use full idx (full-batch setting)
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
    ema: float = 0.9  # used only by derivative-free search modes
    eps: float = 1e-12
    subset_nodes: int = 0   # 0 => full train_idx
    seed: int = 0
    data_anchor: str = "clean"
    search_method: str = "grid"   # {"grid", "powell", "newton", "adam"}
    compare_search_methods: bool = False
    powell_maxiter: int = 25
    powell_xtol: float = 1e-3
    powell_ftol: float = 1e-3
    newton_maxiter: int = 10
    newton_tol: float = 1e-4
    newton_damping: float = 1e-4
    # Adam controller for p,q
    adam_lr: float = 1e-3
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8


class PQGradNormTuner:
    """
    Epoch-wise GradNorm matcher for RADE-OF / RADE-OFS.

    Search options:
    - grid   : bounded finite search over candidate (p, q)
    - powell : bounded derivative-free continuous search
    - newton : bounded damped-Newton search in normalized coordinates

    Evaluation paths:
    - GIN: fast path via two base gradients
    - GCN: each candidate/objective evaluation does one autograd.grad on the detached proxy
    """

    def __init__(
        self,
        edge_index_clean: torch.Tensor,
        num_nodes: int,
        gnn: str,
        rade_variant: str,
        cfg: PQTunerConfig,
    ):
        self.cfg = cfg
        self.gnn = str(gnn).lower().strip()                # "gin" or "gcn"
        self.rade_variant = _validate_rade_variant(rade_variant)
        self.search_method = _validate_search_method(getattr(cfg, "search_method", "grid"))
        self.cache = GraphCache.from_edge_index(edge_index_clean, int(num_nodes))
        self.N = int(num_nodes)

        self.edge_index_clean = edge_index_clean
        self.augmentor = BernoulliEdgeAugmentor.from_edge_index(
            edge_index_clean, num_nodes=self.N
        )

        da = str(getattr(cfg, "data_anchor", "clean")).lower().strip()
        if da not in {"clean", "aug"}:
            raise ValueError(f"PQTunerConfig.data_anchor must be 'clean' or 'aug', got {da}")
        self.data_anchor = da
        # Stateful Adam controller for pq_search_method='adam'
        self._adam_u: Optional[nn.Parameter] = None
        self._adam_v: Optional[nn.Parameter] = None
        self._adam_opt: Optional[torch.optim.Optimizer] = None
        self._adam_device: Optional[torch.device] = None

    def _num_non_edges_undirected(self) -> int:
        # Assumes edge_index_clean_dir contains both directions for each undirected edge.
        m_dir = int(self.cache.edge_index_clean_dir.size(1))
        m_undir = m_dir // 2
        n = int(self.N)
        return n * (n - 1) // 2 - m_undir

    @torch.no_grad()
    def _messages_detached(
        self, emb: torch.Tensor, W: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = emb.detach()
        W = W.detach()
        m = h @ W.t()      # [N, C]
        return m, m * m

    def _var_gin_bases(
        self, m: torch.Tensor, m_sq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        row, col = self.cache.edge_index_clean_dir
        neigh_sum_sq = _scatter_add_rows(m_sq[row], col, self.N)  # [N, C]
        comp_sum_sq = self.cache.complement_sum_unweighted(m_sq)  # [N, C]

        if self.rade_variant == "rade-of":
            comp_sum_m = self.cache.complement_sum_unweighted(m)  # [N, C]
            comp_size = self.cache.comp_size.to(m.dtype).clamp_min(1.0).unsqueeze(1)
            mu = comp_sum_m / comp_size
            add_centered = comp_sum_sq - 2.0 * mu * comp_sum_m + comp_size * (mu * mu)
            return neigh_sum_sq, add_centered

        # RADE-OFS
        return neigh_sum_sq, comp_sum_sq

    @torch.no_grad()
    def _var_gcn(self, p: float, q: float, m: torch.Tensor, m_sq: torch.Tensor) -> torch.Tensor:
        eps = self.cfg.eps
        row, col = self.cache.edge_index_clean_dir

        deg = self.cache.deg_clean.to(torch.float32)
        comp = self.cache.comp_size.to(torch.float32)
        d_hat = (deg + 1.0).clamp_min(1.0)

        # --- retained clean edges: Case A ---
        tA = _delta_E_inv_sqrt_caseA(deg, comp, p=p, q=q)
        uA = _delta_E_inv_caseA(deg, comp, p=p, q=q, eps=eps)

        E = (1.0 - p) * (tA[row] * tA[col])
        E2 = (1.0 - p) * (uA[row] * uA[col])
        VarG = (E2 - E * E).clamp_min(0.0)

        alpha_sq = 1.0 / (d_hat[row] * d_hat[col])  # clean alpha^2
        factor = alpha_sq * (VarG / (E * E + eps))  # per-directed-edge
        neigh_var = _scatter_add_rows(factor.unsqueeze(1) * m_sq[row], col, self.N)

        # --- clean non-edges: Case B ---
        tB = _delta_E_inv_sqrt_caseB(deg, comp, p=p, q=q)
        uB = _delta_E_inv_caseB(deg, comp, p=p, q=q, eps=eps)
        tB2 = tB * tB

        if self.rade_variant == "rade-of":
            # weighted centering with weights proportional to tB[j]
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
            # RADE-OFS: uncentered non-edge contributions
            S1 = self.cache.complement_sum_weighted(m_sq, uB)
            S2 = self.cache.complement_sum_weighted(m_sq, tB2)

        add_var = (q * uB.unsqueeze(1) * S1) - ((q * q) * tB2.unsqueeze(1) * S2)

        return neigh_var + add_var

    def _var_gcn_tensor(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        m: torch.Tensor,
        m_sq: torch.Tensor,
    ) -> torch.Tensor:
        eps = self.cfg.eps
        row, col = self.cache.edge_index_clean_dir

        deg = self.cache.deg_clean.to(torch.float32)
        comp = self.cache.comp_size.to(torch.float32)
        d_hat = (deg + 1.0).clamp_min(1.0)

        tA = _delta_E_inv_sqrt_caseA(deg, comp, p=p, q=q)
        uA = _delta_E_inv_caseA(deg, comp, p=p, q=q, eps=eps)

        E = (1.0 - p) * (tA[row] * tA[col])
        E2 = (1.0 - p) * (uA[row] * uA[col])
        VarG = (E2 - E * E).clamp_min(0.0)

        alpha_sq = 1.0 / (d_hat[row] * d_hat[col])
        factor = alpha_sq * (VarG / (E * E + eps))
        neigh_var = _scatter_add_rows(factor.unsqueeze(1) * m_sq[row], col, self.N)

        tB = _delta_E_inv_sqrt_caseB(deg, comp, p=p, q=q)
        uB = _delta_E_inv_caseB(deg, comp, p=p, q=q, eps=eps)
        tB2 = tB * tB

        if self.rade_variant == "rade-of":
            mu = self.cache.complement_mean_weighted(m, tB)
            ones = torch.ones((self.N, 1), device=m.device, dtype=m.dtype)

            A_u = self.cache.complement_sum_weighted(m_sq, uB)
            B_u = self.cache.complement_sum_weighted(m, uB)
            C_u = self.cache.complement_sum_weighted(ones, uB)
            S1 = A_u - 2.0 * mu * B_u + (mu * mu) * C_u

            A_t = self.cache.complement_sum_weighted(m_sq, tB2)
            B_t = self.cache.complement_sum_weighted(m, tB2)
            C_t = self.cache.complement_sum_weighted(ones, tB2)
            S2 = A_t - 2.0 * mu * B_t + (mu * mu) * C_t
        else:
            S1 = self.cache.complement_sum_weighted(m_sq, uB)
            S2 = self.cache.complement_sum_weighted(m_sq, tB2)

        add_var = (q * uB.unsqueeze(1) * S1) - ((q * q) * tB2.unsqueeze(1) * S2)
        return neigh_var + add_var

    def _build_q_grid(self, q_max: float) -> List[float]:
        cfg = self.cfg
        if q_max <= 0.0:
            return [0.0]
        target = max(2, int(cfg.grid_size))
        q_grid = torch.linspace(0.0, float(q_max), steps=target, device="cpu").tolist()
        q_grid[0] = 0.0
        q_grid[-1] = float(q_max)
        return [float(q) for q in q_grid]

    def _make_objective(
        self,
        *,
        params,
        idx: torch.Tensor,
        w: torch.Tensor,
        m: torch.Tensor,
        m_sq: torch.Tensor,
        logGd: float,
        eps: float,
    ) -> Callable[[float, float], Tuple[float, float]]:
        if self.gnn == "gin":
            V_drop, V_add = self._var_gin_bases(m, m_sq)

            R_drop = 0.5 * (w[idx] * V_drop[idx]).sum() / max(int(idx.numel()), 1)
            R_add = 0.5 * (w[idx] * V_add[idx]).sum() / max(int(idx.numel()), 1)

            g_drop = torch.autograd.grad(
                R_drop, params, retain_graph=True, create_graph=False, allow_unused=True
            )
            g_add = torch.autograd.grad(
                R_add, params, retain_graph=True, create_graph=False, allow_unused=True
            )

            a = _dot(g_drop, g_drop)
            b = _dot(g_add, g_add)
            c = _dot(g_drop, g_add)

            def objective(p_try: float, q_try: float) -> Tuple[float, float]:
                s = p_try / max(1.0 - p_try, eps)   # p / (1-p)
                t = q_try * (1.0 - q_try)           # q(1-q)
                g2 = (s * s) * a + (t * t) * b + 2.0 * s * t * c
                Greg = math.sqrt(max(g2, 0.0))
                obj = (math.log(Greg + eps) - logGd) ** 2
                return float(obj), float(Greg)

            return objective

        if self.gnn == "gcn":
            def objective(p_try: float, q_try: float) -> Tuple[float, float]:
                V = self._var_gcn(float(p_try), float(q_try), m, m_sq)
                R = 0.5 * (w[idx] * V[idx]).sum() / max(int(idx.numel()), 1)

                g_reg = torch.autograd.grad(
                    R, params, retain_graph=True, create_graph=False, allow_unused=True
                )
                Greg = _grad_norm(g_reg)
                obj = (math.log(Greg + eps) - logGd) ** 2
                return float(obj), float(Greg)

            return objective

        raise ValueError(f"Unsupported gnn={self.gnn}. Expected 'gin' or 'gcn'.")

    def _make_objective_tensor(
        self,
        *,
        params,
        idx: torch.Tensor,
        w: torch.Tensor,
        m: torch.Tensor,
        m_sq: torch.Tensor,
        logGd: float,
        eps: float,
    ) -> Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        logGd_t = w.new_tensor(float(logGd))
        eps_t = w.new_tensor(float(eps))

        if self.gnn == "gin":
            V_drop, V_add = self._var_gin_bases(m, m_sq)

            R_drop = 0.5 * (w[idx] * V_drop[idx]).sum() / max(int(idx.numel()), 1)
            R_add = 0.5 * (w[idx] * V_add[idx]).sum() / max(int(idx.numel()), 1)

            g_drop = torch.autograd.grad(
                R_drop, params, retain_graph=True, create_graph=False, allow_unused=True
            )
            g_add = torch.autograd.grad(
                R_add, params, retain_graph=True, create_graph=False, allow_unused=True
            )

            a = w.new_tensor(_dot(g_drop, g_drop))
            b = w.new_tensor(_dot(g_add, g_add))
            c = w.new_tensor(_dot(g_drop, g_add))

            def objective(p_try: torch.Tensor, q_try: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                one = torch.ones_like(p_try)
                s = p_try / (one - p_try).clamp_min(float(eps))
                t = q_try * (one - q_try)
                g2 = (s * s) * a + (t * t) * b + 2.0 * s * t * c
                Greg = torch.sqrt(torch.clamp_min(g2, 0.0))
                obj = (torch.log(Greg + eps_t) - logGd_t) ** 2
                return obj, Greg

            return objective

        if self.gnn == "gcn":
            def objective(p_try: torch.Tensor, q_try: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                V = self._var_gcn_tensor(p_try, q_try, m, m_sq)
                R = 0.5 * (w[idx] * V[idx]).sum() / max(int(idx.numel()), 1)

                g_reg = torch.autograd.grad(
                    R, params, retain_graph=True, create_graph=True, allow_unused=True
                )
                g2 = w.new_zeros(())
                for g in g_reg:
                    if g is None:
                        continue
                    g2 = g2 + (g * g).sum()
                Greg = torch.sqrt(torch.clamp_min(g2, 0.0))
                obj = (torch.log(Greg + eps_t) - logGd_t) ** 2
                return obj, Greg

            return objective

        raise ValueError(f"Unsupported gnn={self.gnn}. Expected 'gin' or 'gcn'.")

    def _search_grid(
        self,
        objective: Callable[[float, float], Tuple[float, float]],
        *,
        p_max: float,
        q_max: float,
        device: torch.device,
    ) -> Tuple[float, float, float, float, Dict[str, float]]:
        cfg = self.cfg

        p_grid = [0.0] if p_max <= 0.0 else torch.linspace(
            0.0, float(p_max), steps=int(cfg.grid_size), device=device
        ).tolist()
        q_grid = self._build_q_grid(q_max)

        best_obj = float("inf")
        best_p, best_q = 0.0, 0.0
        best_Greg = 0.0
        n_eval = 0

        for p_try in p_grid:
            for q_try in q_grid:
                obj, Greg = objective(float(p_try), float(q_try))
                n_eval += 1
                if obj < best_obj:
                    best_obj, best_p, best_q, best_Greg = obj, float(p_try), float(q_try), float(Greg)

        meta = {
            "n_eval": float(n_eval),
            "solver_success": 1.0,
        }
        return best_p, best_q, best_Greg, best_obj, meta

    def _search_powell(
            self,
            objective: Callable[[float, float], Tuple[float, float]],
            *,
            current_p: float,
            current_q: float,
            p_max: float,
            q_max: float,
    ) -> Tuple[float, float, float, float, Dict[str, float]]:
        if minimize is None:
            raise ImportError(
                "search_method='powell' requires scipy. Install scipy or switch search_method='grid'."
            )

        active = []
        if p_max > 0.0:
            active.append("p")
        if q_max > 0.0:
            active.append("q")

        if not active:
            obj0, Greg0 = objective(0.0, 0.0)
            meta = {
                "n_eval": 1.0,
                "solver_success": 1.0,
            }
            return 0.0, 0.0, Greg0, obj0, meta

        def unpack(x: np.ndarray) -> Tuple[float, float]:
            # x lives in normalized coordinates in [0, 1]^d; map back to raw (p, q).
            i = 0
            p_try, q_try = 0.0, 0.0
            if "p" in active:
                p_try = float(x[i]) * float(p_max)
                i += 1
            if "q" in active:
                q_try = float(x[i]) * float(q_max)
                i += 1
            return p_try, q_try

        # Normalize active search variables to [0, 1] so Powell sees coordinates on comparable scales.
        # This is important because p may live on [0, 0.8] while q may live on [0, 1e-3], and
        # direct optimization in raw coordinates can make the q-dimension numerically under-resolved.
        x0 = []
        bounds = []
        if "p" in active:
            p0 = float(max(0.0, min(current_p, p_max)))
            x0.append(0.0 if p_max <= 0.0 else p0 / float(p_max))
            bounds.append((0.0, 1.0))
        if "q" in active:
            q0 = float(max(0.0, min(current_q, q_max)))
            x0.append(0.0 if q_max <= 0.0 else q0 / float(q_max))
            bounds.append((0.0, 1.0))

        best_seen = {
            "obj": float("inf"),
            "p": 0.0,
            "q": 0.0,
            "Greg": 0.0,
            "n_eval": 0,
        }

        def fun_np(x: np.ndarray) -> float:
            p_try, q_try = unpack(np.asarray(x, dtype=float))
            obj, Greg = objective(p_try, q_try)
            best_seen["n_eval"] += 1
            if obj < best_seen["obj"]:
                best_seen["obj"] = float(obj)
                best_seen["p"] = float(p_try)
                best_seen["q"] = float(q_try)
                best_seen["Greg"] = float(Greg)
            return float(obj)

        res = minimize(
            fun_np,
            x0=np.asarray(x0, dtype=float),
            method="Powell",
            bounds=bounds,
            options={
                "maxiter": int(self.cfg.powell_maxiter),
                "xtol": float(self.cfg.powell_xtol),
                "ftol": float(self.cfg.powell_ftol),
                "disp": False,
            },
        )

        p_best = float(best_seen["p"])
        q_best = float(best_seen["q"])
        Greg_best = float(best_seen["Greg"])
        obj_best = float(best_seen["obj"])

        meta = {
            "n_eval": float(best_seen["n_eval"]),
            "solver_success": 1.0 if bool(res.success) else 0.0,
        }
        return p_best, q_best, Greg_best, obj_best, meta

    def _search_newton(
        self,
        objective: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        *,
        current_p: float,
        current_q: float,
        p_max: float,
        q_max: float,
        device: torch.device,
    ) -> Tuple[float, float, float, float, Dict[str, float]]:
        active = []
        if p_max > 0.0:
            active.append("p")
        if q_max > 0.0:
            active.append("q")

        dtype = torch.float32
        zero = torch.zeros((), device=device, dtype=dtype)

        def unpack(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            i = 0
            p_try = zero
            q_try = zero
            if "p" in active:
                p_try = x[i] * float(p_max)
                i += 1
            if "q" in active:
                q_try = x[i] * float(q_max)
                i += 1
            return p_try, q_try

        if not active:
            obj0, Greg0 = objective(zero, zero)
            meta = {
                "n_eval": 1.0,
                "solver_success": 1.0,
            }
            return 0.0, 0.0, float(Greg0.detach()), float(obj0.detach()), meta

        x0 = []
        if "p" in active:
            p0 = float(max(0.0, min(current_p, p_max)))
            x0.append(0.0 if p_max <= 0.0 else p0 / float(p_max))
        if "q" in active:
            q0 = float(max(0.0, min(current_q, q_max)))
            x0.append(0.0 if q_max <= 0.0 else q0 / float(q_max))
        x_best = torch.tensor(x0, device=device, dtype=dtype)

        n_eval = 0
        solver_success = 0.0
        min_damping = max(float(self.cfg.newton_damping), 1e-8)
        damping = min_damping

        for _ in range(int(self.cfg.newton_maxiter)):
            x_curr = x_best.detach().clone().requires_grad_(True)
            p_curr, q_curr = unpack(x_curr)
            obj_curr, _ = objective(p_curr, q_curr)
            n_eval += 1

            grad = torch.autograd.grad(obj_curr, x_curr, create_graph=True)[0]
            if float(grad.detach().abs().max()) <= float(self.cfg.newton_tol):
                x_best = x_curr.detach()
                solver_success = 1.0
                break

            rows = []
            for i in range(x_curr.numel()):
                row = torch.autograd.grad(
                    grad[i],
                    x_curr,
                    retain_graph=(i + 1) < x_curr.numel(),
                )[0]
                rows.append(row)
            hess = torch.stack(rows, dim=0)
            hess = 0.5 * (hess + hess.transpose(0, 1))
            eye = torch.eye(x_curr.numel(), device=device, dtype=dtype)

            accepted = False
            trial_damping = damping
            for _ in range(8):
                try:
                    step = torch.linalg.solve(hess + trial_damping * eye, -grad)
                except RuntimeError:
                    trial_damping *= 10.0
                    continue

                if not torch.isfinite(step).all():
                    trial_damping *= 10.0
                    continue

                alpha = 1.0
                step_detached = step.detach()
                for _ in range(12):
                    x_trial = torch.clamp(x_curr.detach() + alpha * step_detached, 0.0, 1.0)
                    if torch.equal(x_trial, x_curr.detach()):
                        alpha *= 0.5
                        continue

                    x_trial_req = x_trial.clone().detach().requires_grad_(True)
                    p_trial, q_trial = unpack(x_trial_req)
                    obj_trial, _ = objective(p_trial, q_trial)
                    n_eval += 1

                    if float(obj_trial.detach()) < float(obj_curr.detach()):
                        x_best = x_trial.detach()
                        damping = max(trial_damping * 0.5, min_damping)
                        solver_success = 1.0
                        accepted = True
                        break
                    alpha *= 0.5

                if accepted:
                    break
                trial_damping *= 10.0

            if not accepted:
                x_best = x_curr.detach()
                break

        x_final = x_best.detach().clone().requires_grad_(True)
        p_final, q_final = unpack(x_final)
        obj_final, Greg_final = objective(p_final, q_final)
        n_eval += 1

        meta = {
            "n_eval": float(n_eval),
            "solver_success": float(solver_success),
        }
        return (
            float(p_final.detach()),
            float(q_final.detach()),
            float(Greg_final.detach()),
            float(obj_final.detach()),
            meta,
        )

    def _run_search(
        self,
        search_method: str,
        objective: Callable[[float, float], Tuple[float, float]],
        *,
        current_p: float,
        current_q: float,
        p_max: float,
        q_max: float,
        device: torch.device,
        objective_tensor: Optional[
            Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
    ) -> Tuple[float, float, float, float, Dict[str, float]]:
        search_method = _validate_search_method(search_method)
        if search_method == "grid":
            return self._search_grid(
                objective,
                p_max=p_max,
                q_max=q_max,
                device=device,
            )
        if search_method == "powell":
            return self._search_powell(
                objective,
                current_p=current_p,
                current_q=current_q,
                p_max=p_max,
                q_max=q_max,
            )
        if objective_tensor is None:
            raise ValueError("search_method='newton' requires a differentiable objective.")
        return self._search_newton(
            objective_tensor,
            current_p=current_p,
            current_q=current_q,
            p_max=p_max,
            q_max=q_max,
            device=device,
        )

    @staticmethod
    def _format_search_result(
        search_method: str,
        p_best: float,
        q_best: float,
        G_reg_best: float,
        obj_best: float,
        meta: Dict[str, float],
    ) -> Dict[str, float]:
        return {
            "method": str(search_method),
            "p_best": float(p_best),
            "q_best": float(q_best),
            "G_reg_best": float(G_reg_best),
            "obj": float(obj_best),
            "n_eval": float(meta.get("n_eval", 0.0)),
            "solver_success": float(meta.get("solver_success", 0.0)),
        }

    def _collect_search_results(
        self,
        *,
        objective: Callable[[float, float], Tuple[float, float]],
        objective_tensor: Optional[
            Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        ],
        current_p: float,
        current_q: float,
        p_max: float,
        q_max: float,
        device: torch.device,
        primary_method: str,
        primary_result: Dict[str, float],
    ) -> Dict[str, Dict[str, float]]:
        results: Dict[str, Dict[str, float]] = {str(primary_method): dict(primary_result)}

        for method in SEARCH_METHODS:
            if method in results:
                continue
            p_try, q_try, G_try, obj_try, meta_try = self._run_search(
                method,
                objective,
                current_p=current_p,
                current_q=current_q,
                p_max=p_max,
                q_max=q_max,
                device=device,
                objective_tensor=objective_tensor if method == "newton" else None,
            )
            results[method] = self._format_search_result(
                method,
                p_try,
                q_try,
                G_try,
                obj_try,
                meta_try,
            )

        return results

    @staticmethod
    def _make_comparison_info(search_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        methods = [method for method in SEARCH_METHODS if method in search_results]
        if not methods:
            return {}

        best_method = min(methods, key=lambda method: float(search_results[method]["obj"]))
        objs = [float(search_results[method]["obj"]) for method in methods]

        info: Dict[str, Any] = {
            "comparison_enabled": 1.0,
            "comparison_methods": ",".join(methods),
            "comparison_num_methods": float(len(methods)),
            "comparison_better_method": str(best_method),
            "comparison_best_obj": float(search_results[best_method]["obj"]),
            "comparison_obj_spread": float(max(objs) - min(objs)),
        }

        for method in methods:
            result = search_results[method]
            info[f"comparison_{method}_obj"] = float(result["obj"])
            info[f"comparison_{method}_p_best"] = float(result["p_best"])
            info[f"comparison_{method}_q_best"] = float(result["q_best"])
            info[f"comparison_{method}_G_reg_best"] = float(result["G_reg_best"])
            info[f"comparison_{method}_n_eval"] = float(result["n_eval"])
            info[f"comparison_{method}_solver_success"] = float(result["solver_success"])

        for left_idx, left_method in enumerate(methods):
            for right_method in methods[left_idx + 1:]:
                gap = float(search_results[right_method]["obj"]) - float(search_results[left_method]["obj"])
                info[f"comparison_gap_{right_method}_minus_{left_method}"] = gap

        return info

    @staticmethod
    def _sample_objective_surface(
        objective: Callable[[float, float], Tuple[float, float]],
        *,
        p_max: float,
        q_max: float,
        grid_size: int,
    ) -> Dict[str, Any]:
        p_size = max(2, int(grid_size)) if p_max > 0.0 else 1
        q_size = max(2, int(grid_size)) if q_max > 0.0 else 1

        p_values = np.linspace(0.0, float(p_max), num=p_size, dtype=np.float64)
        q_values = np.linspace(0.0, float(q_max), num=q_size, dtype=np.float64)

        obj_grid = np.zeros((q_values.shape[0], p_values.shape[0]), dtype=np.float64)
        greg_grid = np.zeros_like(obj_grid)

        for q_idx, q_try in enumerate(q_values):
            for p_idx, p_try in enumerate(p_values):
                obj_val, greg_val = objective(float(p_try), float(q_try))
                obj_grid[q_idx, p_idx] = float(obj_val)
                greg_grid[q_idx, p_idx] = float(greg_val)

        return {
            "p_values": p_values.tolist(),
            "q_values": q_values.tolist(),
            "objective": obj_grid.tolist(),
            "G_reg": greg_grid.tolist(),
        }

    @staticmethod
    def _safe_logit_from_bounded(x: float, x_max: float, eps: float = 1e-6) -> float:
        if x_max <= 0.0:
            return 0.0
        r = float(x) / float(x_max)
        r = min(max(r, eps), 1.0 - eps)
        return math.log(r / (1.0 - r))

    def _ensure_adam_state(
        self,
        *,
        current_p: float,
        current_q: float,
        device: torch.device,
    ) -> None:
        if (
            self._adam_opt is not None
            and self._adam_u is not None
            and self._adam_v is not None
            and self._adam_device == device
        ):
            return

        cfg = self.cfg
        u0 = self._safe_logit_from_bounded(current_p, cfg.p_max)
        v0 = self._safe_logit_from_bounded(current_q, cfg.q_max)

        self._adam_u = nn.Parameter(torch.tensor(u0, device=device, dtype=torch.float32))
        self._adam_v = nn.Parameter(torch.tensor(v0, device=device, dtype=torch.float32))

        self._adam_opt = torch.optim.Adam(
            [self._adam_u, self._adam_v],
            lr=float(cfg.adam_lr),
            betas=(float(cfg.adam_beta1), float(cfg.adam_beta2)),
            eps=float(cfg.adam_eps),
        )
        self._adam_device = device

    def _adam_rates_tensor(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._adam_u is None or self._adam_v is None:
            raise RuntimeError("Adam pq controller is not initialized.")
        cfg = self.cfg
        p = float(cfg.p_max) * torch.sigmoid(self._adam_u)
        q = float(cfg.q_max) * torch.sigmoid(self._adam_v)
        return p, q

    @torch.no_grad()
    def _adam_rates_float(self, aug_mode: str) -> Tuple[float, float]:
        p_t, q_t = self._adam_rates_tensor()
        p = float(p_t.item())
        q = float(q_t.item())

        aug_mode = str(aug_mode).lower().strip()
        if aug_mode == "drop":
            q = 0.0
        elif aug_mode == "add":
            p = 0.0
        elif aug_mode == "none":
            p, q = 0.0, 0.0

        p = float(max(0.0, min(p, self.cfg.p_max)))
        q = float(max(0.0, min(q, self.cfg.q_max)))
        return p, q

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
        capture_surface: bool = False,
        surface_grid_size: Optional[int] = None,
    ) -> Tuple[float, float, Dict[str, Any]]:
        cfg = self.cfg
        eps = cfg.eps

        # Optional subset for speed; subset_nodes=0 => full train_idx
        idx = _pick_subset(train_idx, cfg.subset_nodes, seed=epoch_seed)

        # We want training-mode gradients (dropout consistent), but must NOT update BN running stats.
        was_training = model.training
        model.train()
        bn_states = _freeze_batchnorm_buffers(model)

        # Isolate RNG usage so this tuning step does not consume global RNG state.
        devices = []
        if data.x.is_cuda and data.x.device.index is not None:
            devices = [data.x.device.index]

        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.manual_seed(int(epoch_seed) + int(cfg.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(epoch_seed) + int(cfg.seed))

            model.zero_grad(set_to_none=True)

            # -------------------------
            # 1) Anchor data gradient norm (clean or one stochastic RADE draw)
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
                # Use the same kind of stochastic training forward that SGD sees,
                # anchored at the current (p, q), not the candidate (p_try, q_try).
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
            g_data = torch.autograd.grad(
                loss_data, params, retain_graph=False, create_graph=False, allow_unused=True
            )
            G_data = _grad_norm(g_data)
            logGd = math.log(G_data + eps)

            # -------------------------
            # 2) Snapshot for regularizer proxy
            #    Always compute from a clean forward (theoretical reference),
            #    while the variance term stays detached.
            # -------------------------
            model.zero_grad(set_to_none=True)
            logits, emb = model(data.x, data.edge_index, p=0.0, q=0.0, return_embeddings=True)

            W = model.pred_local.weight
            m, m_sq = self._messages_detached(emb, W)

            # Multiclass analogue of z(1-z); differentiable; Var stays detached.
            w = _w_multiclass(logits)

            # Bounds (respect aug_mode)
            aug_mode = str(aug_mode).lower().strip()
            p_max = cfg.p_max if aug_mode in ("both", "drop") else 0.0
            q_max = cfg.q_max if aug_mode in ("both", "add") else 0.0

            objective = self._make_objective(
                params=params,
                idx=idx,
                w=w,
                m=m,
                m_sq=m_sq,
                logGd=logGd,
                eps=eps,
            )
            comparison_enabled = bool(getattr(cfg, "compare_search_methods", False))
            need_all_search_results = comparison_enabled or bool(capture_surface)

            objective_tensor = self._make_objective_tensor(
                params=params,
                idx=idx,
                w=w,
                m=m,
                m_sq=m_sq,
                logGd=logGd,
                eps=eps,
            )
            # -------------------------------------------------
            # Adam controller path: one online step on (u, v)
            # -------------------------------------------------
            if self.search_method == "adam":
                if bool(cfg.compare_search_methods):
                    raise ValueError("pq_search_method='adam' does not support --pq_compare_search_methods.")

                if objective_tensor is None:
                    raise RuntimeError("pq_search_method='adam' requires a differentiable objective.")

                self._ensure_adam_state(
                    current_p=float(current_p),
                    current_q=float(current_q),
                    device=data.x.device,
                )

                # Current controller rates before the Adam step
                p_before, q_before = self._adam_rates_tensor()

                aug_mode_norm = str(aug_mode).lower().strip()
                if aug_mode_norm == "drop":
                    q_before = q_before * 0.0
                elif aug_mode_norm == "add":
                    p_before = p_before * 0.0
                elif aug_mode_norm == "none":
                    p_before = p_before * 0.0
                    q_before = q_before * 0.0

                self._adam_opt.zero_grad(set_to_none=True)
                obj_t, greg_t = objective_tensor(p_before, q_before)
                obj_t.backward(retain_graph=True)
                self._adam_opt.step()

                p_new, q_new = self._adam_rates_float(aug_mode_norm)
                obj_new, greg_new = objective(float(p_new), float(q_new))

                best_p = float(p_before.detach().item())
                best_q = float(q_before.detach().item())
                best_Greg = float(greg_t.detach().item())
                best_obj = float(obj_t.detach().item())

                search_meta = {
                    "n_eval": 2.0,
                    "solver_success": 1.0,
                    "adam_lr": float(cfg.adam_lr),
                }

                comparison_info = None
                surface_data = None
                if bool(capture_surface):
                    surface_data = self._sample_objective_surface(
                        objective,
                        p_max=p_max,
                        q_max=q_max,
                        grid_size=int(surface_grid_size or max(11, int(cfg.grid_size))),
                    )
                    surface_data["search_results"] = {
                        "adam": {
                            "method": "adam",
                            "p_best": float(best_p),
                            "q_best": float(best_q),
                            "G_reg_best": float(best_Greg),
                            "obj": float(best_obj),
                            "p_new": float(p_new),
                            "q_new": float(q_new),
                            "G_reg_new": float(greg_new),
                            "obj_new": float(obj_new),
                            "n_eval": 2.0,
                            "solver_success": 1.0,
                        }
                    }
                    surface_data["selected_method"] = "adam"
                    surface_data["p_max"] = float(p_max)
                    surface_data["q_max"] = float(q_max)
                    surface_data["G_data"] = float(G_data)

                # Restore BN module states
                _restore_batchnorm_buffers(bn_states)

                # Restore model mode
                if not was_training:
                    model.eval()

                info = {
                    "G_data": float(G_data),
                    "G_reg_best": float(best_Greg),
                    "p_best": float(best_p),
                    "q_best": float(best_q),
                    "p_new": float(p_new),
                    "q_new": float(q_new),
                    "obj": float(best_obj),
                    "G_reg_new": float(greg_new),
                    "obj_new": float(obj_new),
                    "search_method": "adam",
                    **search_meta,
                }
                if surface_data is not None:
                    info["surface_data"] = surface_data
                return p_new, q_new, info

            if self.search_method == "newton" or need_all_search_results:
                objective_tensor = self._make_objective_tensor(
                    params=params,
                    idx=idx,
                    w=w,
                    m=m,
                    m_sq=m_sq,
                    logGd=logGd,
                    eps=eps,
                )

            best_p, best_q, best_Greg, best_obj, search_meta = self._run_search(
                self.search_method,
                objective,
                current_p=float(current_p),
                current_q=float(current_q),
                p_max=p_max,
                q_max=q_max,
                device=data.x.device,
                objective_tensor=objective_tensor,
            )
            primary_result = self._format_search_result(
                self.search_method,
                best_p,
                best_q,
                best_Greg,
                best_obj,
                search_meta,
            )

            search_results = None
            if need_all_search_results:
                search_results = self._collect_search_results(
                    objective=objective,
                    objective_tensor=objective_tensor,
                    current_p=float(current_p),
                    current_q=float(current_q),
                    p_max=p_max,
                    q_max=q_max,
                    device=data.x.device,
                    primary_method=self.search_method,
                    primary_result=primary_result,
                )

                selected_result = search_results[self.search_method]
                best_p = float(selected_result["p_best"])
                best_q = float(selected_result["q_best"])
                best_Greg = float(selected_result["G_reg_best"])
                best_obj = float(selected_result["obj"])
                search_meta = {
                    "n_eval": float(selected_result["n_eval"]),
                    "solver_success": float(selected_result["solver_success"]),
                }

            comparison_info = None
            if comparison_enabled:
                if search_results is None:
                    raise RuntimeError("Comparison requested but search results were not collected.")
                comparison_info = self._make_comparison_info(search_results)

            surface_data = None
            if bool(capture_surface):
                if search_results is None:
                    raise RuntimeError("Surface capture requested but search results were not collected.")
                surface_data = self._sample_objective_surface(
                    objective,
                    p_max=p_max,
                    q_max=q_max,
                    grid_size=int(surface_grid_size or max(11, int(cfg.grid_size))),
                )
                surface_data["search_results"] = {
                    method: dict(result) for method, result in search_results.items()
                }
                surface_data["selected_method"] = str(self.search_method)
                surface_data["p_max"] = float(p_max)
                surface_data["q_max"] = float(q_max)
                surface_data["G_data"] = float(G_data)

        # Restore BN module states
        _restore_batchnorm_buffers(bn_states)

        # Restore model mode
        if not was_training:
            model.eval()

        # EMA update
        p_new = cfg.ema * float(current_p) + (1.0 - cfg.ema) * float(best_p)
        q_new = cfg.ema * float(current_q) + (1.0 - cfg.ema) * float(best_q)

        # Enforce aug_mode
        if aug_mode == "drop":
            q_new = 0.0
        elif aug_mode == "add":
            p_new = 0.0
        elif aug_mode == "none":
            p_new, q_new = 0.0, 0.0

        # Clamp
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
            "search_method": self.search_method,
            **search_meta,
        }
        if comparison_info is not None:
            info.update(comparison_info)
        if surface_data is not None:
            info["surface_data"] = surface_data
        return p_new, q_new, info
