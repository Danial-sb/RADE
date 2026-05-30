from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover
    minimize = None

from .mb_augmentation import BatchBernoulliEdgeAugmentor
from .mb_rade_convs import (
    BatchGraphCache,
    _delta_E_inv_caseA,
    _delta_E_inv_caseB,
    _delta_E_inv_sqrt_caseA,
    _delta_E_inv_sqrt_caseB,
    _scatter_add_rows,
)


# Keep SEARCH_METHODS for the derivative-free / comparison family only.
# Adam is stateful and should not be mixed into the frozen-snapshot side-by-side comparison utilities.
SEARCH_METHODS: Tuple[str, ...] = ("grid", "powell", "newton")

# Match the full-batch online controller bounds. Non-Adam mini-batch
# search methods still use cfg.p_max/cfg.q_max as their finite search box.
P_PROBABILITY_UPPER: float = 1.0 - 1e-6
Q_PROBABILITY_UPPER: float = 1.0


def _validate_rade_variant(rade_variant: str) -> str:
    rade_variant = str(rade_variant).lower().strip()
    if rade_variant not in {"rade-of", "rade-ofs"}:
        raise ValueError(f"rade_variant must be one of {{'rade-of', 'rade-ofs'}}. Got {rade_variant}")
    return rade_variant


def _validate_search_method(search_method: str) -> str:
    search_method = str(search_method).lower().strip()
    if search_method not in {"grid", "powell", "newton", "adam"}:
        raise ValueError(
            f"search_method must be one of {{'grid', 'powell', 'newton', 'adam'}}. Got {search_method}"
        )
    return search_method


def _validate_penalty_type(penalty_type: str) -> str:
    penalty_type = str(penalty_type).lower().strip()
    if penalty_type not in {"quadratic", "linear", "hinge"}:
        raise ValueError(
            "densification_penalty_type must be one of {'quadratic', 'linear', 'hinge'}. "
            f"Got {penalty_type}"
        )
    return penalty_type


def _validate_batch_weighting(batch_weighting: str) -> str:
    batch_weighting = str(batch_weighting).lower().strip()
    if batch_weighting not in {"node_count", "uniform"}:
        raise ValueError(
            "batch_weighting must be one of {'node_count', 'uniform'}. "
            f"Got {batch_weighting}"
        )
    return batch_weighting


def _grad_norm(grads) -> float:
    total = 0.0
    for grad in grads:
        if grad is None:
            continue
        total += float((grad * grad).sum().item())
    return math.sqrt(max(total, 0.0))


def _dot(grads_a, grads_b) -> float:
    total = 0.0
    for grad_a, grad_b in zip(grads_a, grads_b):
        if grad_a is None or grad_b is None:
            continue
        total += float((grad_a * grad_b).sum().item())
    return total


def _pick_subset(idx: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    if k <= 0 or k >= idx.numel():
        return idx
    gen = torch.Generator(device=idx.device)
    gen.manual_seed(int(seed))
    perm = torch.randperm(idx.numel(), generator=gen, device=idx.device)
    return idx[perm[:k]]


def _w_multiclass(logits: torch.Tensor) -> torch.Tensor:
    prob = torch.softmax(logits, dim=-1)
    return prob * (1.0 - prob)


def _freeze_batchnorm_buffers(model: nn.Module) -> List[Tuple[nn.Module, bool]]:
    states = []
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            states.append((module, module.training))
            module.eval()
    return states


def _restore_batchnorm_buffers(states: List[Tuple[nn.Module, bool]]) -> None:
    for module, was_training in states:
        module.train(was_training)


@dataclass
class PQTunerMBConfig:
    p_max: float = 0.8
    q_max: float = 1e-3
    expected_add_ratio_scale: float = 0.0
    deletion_penalty_lambda: float = 0.0
    densification_penalty_lambda: float = 1.0
    densification_penalty_type: str = "quadratic"
    densification_penalty_rho0: float = 1.0
    optimize_rho: bool = False
    cap_p: bool = False
    p_cap_max: float = 0.9
    cap_rho: bool = False
    rho_max: float = 0.2
    grid_size: int = 11
    ema: float = 0.9
    eps: float = 1e-12
    subset_nodes: int = 0
    seed: int = 0
    data_anchor: str = "clean"
    search_method: str = "grid"
    epoch_objective_mode: str = "batch_solutions"
    batch_weighting: str = "node_count"
    compare_search_methods: bool = False
    powell_maxiter: int = 25
    powell_xtol: float = 1e-3
    powell_ftol: float = 1e-3
    newton_maxiter: int = 10
    newton_tol: float = 1e-4
    newton_damping: float = 1e-4
    max_batches_for_tuning: int = 0
    adam_lr: float = 1e-3
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    ep_expectation_mode: str = "analytic"


class PQGradNormTunerMB:
    def __init__(self, gnn: str, rade_variant: str, cfg: PQTunerMBConfig):
        self.cfg = cfg
        self.gnn = str(gnn).lower().strip()
        self.rade_variant = _validate_rade_variant(rade_variant)
        self.ep_expectation_mode = str(getattr(cfg, "ep_expectation_mode", "analytic")).lower().strip()
        if self.gnn not in {"gin", "gcn", "gat"}:
            raise ValueError("gnn must be 'gin', 'gcn', or 'gat'")
        if self.ep_expectation_mode not in {"analytic", "empirical_ema"}:
            raise ValueError(
                "PQTunerMBConfig.ep_expectation_mode must be one of {'analytic', 'empirical_ema'}. "
                f"Got {self.ep_expectation_mode}"
            )
        if self.gnn == "gat" and self.ep_expectation_mode != "analytic":
            raise ValueError("GAT PQ-GradNorm currently requires ep_expectation_mode='analytic'.")
        self.search_method = _validate_search_method(cfg.search_method)
        self.optimize_rho = bool(getattr(cfg, "optimize_rho", False))
        if self.optimize_rho and self.search_method != "adam":
            raise ValueError("--pq_optimize_rho is supported only with pq_search_method='adam' in mini-batch.")
        self.expected_add_ratio_scale = max(0.0, float(getattr(cfg, "expected_add_ratio_scale", 0.0)))
        self.cap_p = bool(getattr(cfg, "cap_p", False))
        p_cap_max = float(getattr(cfg, "p_cap_max", 0.9))
        if not (0.0 <= p_cap_max < 1.0):
            raise ValueError(f"p_cap_max must be in [0, 1). Got {p_cap_max}")
        self.p_probability_upper = min(P_PROBABILITY_UPPER, p_cap_max) if self.cap_p else P_PROBABILITY_UPPER

        self.rho_probability_upper = float(self.expected_add_ratio_scale) * float(Q_PROBABILITY_UPPER)
        self.cap_rho = bool(getattr(cfg, "cap_rho", False))
        rho_max = float(getattr(cfg, "rho_max", 0.2))
        if rho_max < 0.0:
            raise ValueError(f"rho_max must be >= 0. Got {rho_max}")
        self.rho_upper = min(self.rho_probability_upper, rho_max) if self.cap_rho else self.rho_probability_upper
        self.q_probability_upper = (
            min(Q_PROBABILITY_UPPER, self._q_from_rho_float(self.rho_upper))
            if self.cap_rho
            else Q_PROBABILITY_UPPER
        )
        self.search_p_max = min(float(cfg.p_max), float(self.p_probability_upper))
        self.search_q_max = min(float(cfg.q_max), float(self.q_probability_upper))
        self.penalty_type = _validate_penalty_type(getattr(cfg, "densification_penalty_type", "quadratic"))
        self.penalty_rho0 = float(getattr(cfg, "densification_penalty_rho0", 1.0))
        if self.penalty_rho0 < 0.0:
            raise ValueError(f"densification_penalty_rho0 must be >= 0. Got {self.penalty_rho0}")

        data_anchor = str(getattr(cfg, "data_anchor", "clean")).lower().strip()
        if data_anchor not in {"clean", "aug"}:
            raise ValueError(f"PQTunerMBConfig.data_anchor must be 'clean' or 'aug', got {data_anchor}")
        self.data_anchor = data_anchor

        epoch_objective_mode = str(getattr(cfg, "epoch_objective_mode", "batch_solutions")).lower().strip()
        if epoch_objective_mode not in {"batch_solutions", "joint_objective"}:
            raise ValueError(
                "PQTunerMBConfig.epoch_objective_mode must be 'batch_solutions' or 'joint_objective', "
                f"got {epoch_objective_mode}"
            )
        self.epoch_objective_mode = epoch_objective_mode
        self.batch_weighting = _validate_batch_weighting(getattr(cfg, "batch_weighting", "node_count"))
        self._adam_u: Optional[nn.Parameter] = None
        self._adam_v: Optional[nn.Parameter] = None
        self._adam_p: Optional[nn.Parameter] = None
        self._adam_q: Optional[nn.Parameter] = None
        self._adam_opt: Optional[torch.optim.Optimizer] = None
        self._adam_device: Optional[torch.device] = None

    def _rho_from_q_float(self, q: float) -> float:
        return float(q) * float(self.expected_add_ratio_scale)

    def _rho_from_q_tensor(self, q: torch.Tensor) -> torch.Tensor:
        return q * q.new_tensor(float(self.expected_add_ratio_scale))

    def _q_from_rho_float(self, rho: float) -> float:
        scale = float(self.expected_add_ratio_scale)
        if scale <= 0.0:
            return 0.0
        return float(rho) / scale

    def _q_from_rho_tensor(self, rho: torch.Tensor) -> torch.Tensor:
        scale = float(self.expected_add_ratio_scale)
        if scale <= 0.0:
            return rho.new_zeros(())
        return rho / rho.new_tensor(scale)

    def _add_parameter_upper(self) -> float:
        if self.optimize_rho:
            return float(self._rho_from_q_float(float(self.search_q_max)))
        return float(self.search_q_max)

    def _adam_add_parameter_upper(self) -> float:
        if self.optimize_rho:
            return float(self.rho_upper)
        return float(self.q_probability_upper)

    def _add_parameter_name(self) -> str:
        return "rho" if self.optimize_rho else "q"

    def _add_parameter_from_q_float(self, q: float) -> float:
        if self.optimize_rho:
            return self._rho_from_q_float(q)
        return float(q)

    def _q_from_add_parameter_tensor(self, add_param: torch.Tensor) -> torch.Tensor:
        if self.optimize_rho:
            return self._q_from_rho_tensor(add_param)
        return add_param

    def _densification_penalty_float(self, rho: float) -> float:
        lam = float(getattr(self.cfg, "densification_penalty_lambda", 0.0))
        if lam <= 0.0:
            return 0.0
        rho = float(rho)
        if self.penalty_type == "quadratic":
            return lam * rho * rho
        if self.penalty_type == "linear":
            return lam * rho
        excess = max(0.0, rho - float(self.penalty_rho0))
        return lam * excess * excess

    def _densification_penalty_tensor(self, rho: torch.Tensor) -> torch.Tensor:
        lam = float(getattr(self.cfg, "densification_penalty_lambda", 0.0))
        if lam <= 0.0:
            return rho.new_zeros(())
        if self.penalty_type == "quadratic":
            return rho.new_tensor(lam) * rho * rho
        if self.penalty_type == "linear":
            return rho.new_tensor(lam) * rho
        rho0_t = rho.new_tensor(float(self.penalty_rho0))
        excess = torch.clamp_min(rho - rho0_t, 0.0)
        return rho.new_tensor(lam) * excess * excess

    def _deletion_penalty_float(self, p: float) -> float:
        lam = float(getattr(self.cfg, "deletion_penalty_lambda", 0.0))
        if lam <= 0.0:
            return 0.0
        p = float(p)
        return lam * p * p

    def _deletion_penalty_tensor(self, p: torch.Tensor) -> torch.Tensor:
        lam = float(getattr(self.cfg, "deletion_penalty_lambda", 0.0))
        if lam <= 0.0:
            return p.new_zeros(())
        return p.new_tensor(lam) * p * p

    def _objective_penalty_float(self, p: float, q: float) -> float:
        return self._deletion_penalty_float(p) + self._densification_penalty_float(self._rho_from_q_float(q))

    def _objective_penalty_tensor(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return self._deletion_penalty_tensor(p) + self._densification_penalty_tensor(self._rho_from_q_tensor(q))

    def _annotate_surface_data(self, surface_data: Dict[str, Any], *, q_max: float) -> Dict[str, Any]:
        surface_data["rho_upper"] = float(self._rho_from_q_float(float(q_max)))
        surface_data["expected_add_ratio_scale"] = float(self.expected_add_ratio_scale)
        surface_data["p_cap_enabled"] = bool(self.cap_p)
        surface_data["p_probability_upper"] = float(self.p_probability_upper)
        surface_data["rho_cap_enabled"] = bool(self.cap_rho)
        surface_data["rho_probability_upper"] = float(self.rho_upper)
        surface_data["q_probability_upper"] = float(self.q_probability_upper)
        surface_data["deletion_penalty_lambda"] = float(
            getattr(self.cfg, "deletion_penalty_lambda", 0.0)
        )
        surface_data["densification_penalty_lambda"] = float(
            getattr(self.cfg, "densification_penalty_lambda", 0.0)
        )
        surface_data["densification_penalty_type"] = str(self.penalty_type)
        surface_data["densification_penalty_rho0"] = float(self.penalty_rho0)
        return surface_data

    def _epoch_batch_weight(self, snapshot: Dict[str, Any]) -> float:
        if self.batch_weighting == "uniform":
            return 1.0
        return float(snapshot["batch_train_nodes"])

    @torch.no_grad()
    def _messages_detached(self, emb: torch.Tensor, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = emb.detach()
        w = weight.detach()
        msg = h @ w.t()
        return msg, msg * msg

    def _var_gin_bases(self, cache: BatchGraphCache, msg: torch.Tensor, msg_sq: torch.Tensor, batch_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row, col = cache.edge_index_clean_dir
        neigh_sum_sq = _scatter_add_rows(msg_sq[row], col, batch_nodes)
        comp_sum_sq = cache.complement_sum_unweighted(msg_sq)

        if self.rade_variant == "rade-of":
            comp_sum_m = cache.complement_sum_unweighted(msg)
            comp_size = cache.comp_size.to(msg.dtype).clamp_min(1.0).unsqueeze(1)
            mu = comp_sum_m / comp_size
            add_centered = comp_sum_sq - 2.0 * mu * comp_sum_m + comp_size * (mu * mu)
            return neigh_sum_sq, add_centered

        return neigh_sum_sq, comp_sum_sq

    @torch.no_grad()
    def _var_gcn(self, cache: BatchGraphCache, p: float, q: float, msg: torch.Tensor, msg_sq: torch.Tensor, batch_nodes: int) -> torch.Tensor:
        eps = self.cfg.eps
        row, col = cache.edge_index_clean_dir

        deg = cache.deg_clean.to(torch.float32)
        comp = cache.comp_size.to(torch.float32)
        d_hat = (deg + 1.0).clamp_min(1.0)

        t_a = _delta_E_inv_sqrt_caseA(deg, comp, p=p, q=q)
        u_a = _delta_E_inv_caseA(deg, comp, p=p, q=q, eps=eps)
        e = (1.0 - p) * (t_a[row] * t_a[col])
        e2 = (1.0 - p) * (u_a[row] * u_a[col])
        var_g = (e2 - e * e).clamp_min(0.0)

        alpha_sq = 1.0 / (d_hat[row] * d_hat[col])
        factor = alpha_sq * (var_g / (e * e + eps))
        neigh_var = _scatter_add_rows(factor.unsqueeze(1) * msg_sq[row], col, batch_nodes)

        t_b = _delta_E_inv_sqrt_caseB(deg, comp, p=p, q=q)
        u_b = _delta_E_inv_caseB(deg, comp, p=p, q=q, eps=eps)
        t_b2 = t_b * t_b

        if self.rade_variant == "rade-of":
            mu = cache.complement_mean_weighted(msg, t_b)
            ones = torch.ones((batch_nodes, 1), device=msg.device, dtype=msg.dtype)

            a_u = cache.complement_sum_weighted(msg_sq, u_b)
            b_u = cache.complement_sum_weighted(msg, u_b)
            c_u = cache.complement_sum_weighted(ones, u_b)
            s1 = a_u - 2.0 * mu * b_u + (mu * mu) * c_u

            a_t = cache.complement_sum_weighted(msg_sq, t_b2)
            b_t = cache.complement_sum_weighted(msg, t_b2)
            c_t = cache.complement_sum_weighted(ones, t_b2)
            s2 = a_t - 2.0 * mu * b_t + (mu * mu) * c_t
        else:
            s1 = cache.complement_sum_weighted(msg_sq, u_b)
            s2 = cache.complement_sum_weighted(msg_sq, t_b2)

        add_var = (q * u_b.unsqueeze(1) * s1) - ((q * q) * t_b2.unsqueeze(1) * s2)
        return neigh_var + add_var

    def _var_gcn_tensor(
        self,
        cache: BatchGraphCache,
        p: torch.Tensor,
        q: torch.Tensor,
        msg: torch.Tensor,
        msg_sq: torch.Tensor,
        batch_nodes: int,
    ) -> torch.Tensor:
        eps = self.cfg.eps
        row, col = cache.edge_index_clean_dir

        deg = cache.deg_clean.to(torch.float32)
        comp = cache.comp_size.to(torch.float32)
        d_hat = (deg + 1.0).clamp_min(1.0)

        t_a = _delta_E_inv_sqrt_caseA(deg, comp, p=p, q=q)
        u_a = _delta_E_inv_caseA(deg, comp, p=p, q=q, eps=eps)
        e = (1.0 - p) * (t_a[row] * t_a[col])
        e2 = (1.0 - p) * (u_a[row] * u_a[col])
        var_g = (e2 - e * e).clamp_min(0.0)

        alpha_sq = 1.0 / (d_hat[row] * d_hat[col])
        factor = alpha_sq * (var_g / (e * e + eps))
        neigh_var = _scatter_add_rows(factor.unsqueeze(1) * msg_sq[row], col, batch_nodes)

        t_b = _delta_E_inv_sqrt_caseB(deg, comp, p=p, q=q)
        u_b = _delta_E_inv_caseB(deg, comp, p=p, q=q, eps=eps)
        t_b2 = t_b * t_b

        if self.rade_variant == "rade-of":
            mu = cache.complement_mean_weighted(msg, t_b)
            ones = torch.ones((batch_nodes, 1), device=msg.device, dtype=msg.dtype)

            a_u = cache.complement_sum_weighted(msg_sq, u_b)
            b_u = cache.complement_sum_weighted(msg, u_b)
            c_u = cache.complement_sum_weighted(ones, u_b)
            s1 = a_u - 2.0 * mu * b_u + (mu * mu) * c_u

            a_t = cache.complement_sum_weighted(msg_sq, t_b2)
            b_t = cache.complement_sum_weighted(msg, t_b2)
            c_t = cache.complement_sum_weighted(ones, t_b2)
            s2 = a_t - 2.0 * mu * b_t + (mu * mu) * c_t
        else:
            s1 = cache.complement_sum_weighted(msg_sq, u_b)
            s2 = cache.complement_sum_weighted(msg_sq, t_b2)

        add_var = (q * u_b.unsqueeze(1) * s1) - ((q * q) * t_b2.unsqueeze(1) * s2)
        return neigh_var + add_var

    @staticmethod
    def _collect_gat_modules(model: nn.Module) -> List[nn.Module]:
        modules = [
            module
            for module in model.modules()
            if callable(getattr(module, "gat_variance_proxy", None))
        ]
        return modules[-1:] if modules else []

    def _var_gat(
        self,
        cache: BatchGraphCache,
        p: float,
        q: float,
        msg: torch.Tensor,
        msg_sq: torch.Tensor,
    ) -> torch.Tensor:
        _ = cache
        modules = getattr(self, "_gat_modules", None)
        if not modules:
            return torch.zeros_like(msg_sq)
        estimates = [
            module.gat_variance_proxy(msg, msg_sq, p=float(p), q=float(q), eps=float(self.cfg.eps))
            for module in modules
        ]
        return estimates[0] if len(estimates) == 1 else torch.stack(estimates, dim=0).mean(dim=0)

    def _var_gat_tensor(
        self,
        cache: BatchGraphCache,
        p: torch.Tensor,
        q: torch.Tensor,
        msg: torch.Tensor,
        msg_sq: torch.Tensor,
    ) -> torch.Tensor:
        _ = cache
        modules = getattr(self, "_gat_modules", None)
        if not modules:
            return torch.zeros_like(msg_sq) + (p * 0.0 + q * 0.0)
        estimates = [
            module.gat_variance_proxy(msg, msg_sq, p=p, q=q, eps=float(self.cfg.eps))
            for module in modules
        ]
        value = estimates[0] if len(estimates) == 1 else torch.stack(estimates, dim=0).mean(dim=0)
        return value + (p * 0.0 + q * 0.0)

    def _empirical_gcn_variance(
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
        modules = getattr(self, "_empirical_gcn_modules", None)
        if not modules:
            return torch.zeros_like(msg_sq)
        estimates = []
        for module in modules:
            fn = getattr(module, "empirical_gcn_variance_proxy", None)
            if not callable(fn):
                continue
            value = fn(msg, msg_sq, p=float(p), q=float(q), eps=float(eps))
            if value is not None:
                estimates.append(value)
        if not estimates:
            return torch.zeros_like(msg_sq)
        if len(estimates) == 1:
            return estimates[0]
        return torch.stack(estimates, dim=0).mean(dim=0)

    @staticmethod
    def _collect_empirical_gcn_modules(model: nn.Module) -> List[nn.Module]:
        return [
            module
            for module in model.modules()
            if callable(getattr(module, "empirical_gcn_variance_proxy", None))
        ]

    def _build_q_grid(self, q_max: float) -> List[float]:
        if q_max <= 0.0:
            return [0.0]
        target = max(2, int(self.cfg.grid_size))
        q_grid = torch.linspace(0.0, float(q_max), steps=target, device="cpu").tolist()
        q_grid[0] = 0.0
        q_grid[-1] = float(q_max)
        return [float(q) for q in q_grid]

    @staticmethod
    def _safe_logit_from_bounded(x: float, x_max: float, eps: float = 1e-6) -> float:
        if x_max <= 0.0:
            return 0.0
        ratio = float(x) / float(x_max)
        ratio = min(max(ratio, eps), 1.0 - eps)
        return math.log(ratio / (1.0 - ratio))

    def _ensure_adam_state(
        self,
        *,
        current_p: float,
        current_q: float,
        device: torch.device,
    ) -> None:
        if self._adam_opt is not None and self._adam_device == device and self._adam_p is not None and self._adam_q is not None:
            return

        cfg = self.cfg
        p0 = float(max(0.0, min(current_p, self.p_probability_upper)))
        q0 = float(max(0.0, min(current_q, self.q_probability_upper)))
        add_upper = max(float(self._adam_add_parameter_upper()), 0.0)
        add0 = float(max(0.0, min(self._add_parameter_from_q_float(q0), add_upper)))
        self._adam_p = nn.Parameter(torch.tensor(p0, device=device, dtype=torch.float32))
        self._adam_q = nn.Parameter(torch.tensor(add0, device=device, dtype=torch.float32))
        self._adam_u = None
        self._adam_v = None
        self._adam_opt = torch.optim.Adam(
            [self._adam_p, self._adam_q],
            lr=float(cfg.adam_lr),
            betas=(float(cfg.adam_beta1), float(cfg.adam_beta2)),
            eps=float(cfg.adam_eps),
        )
        self._adam_device = device

    def _adam_rates_tensor(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._adam_p is None or self._adam_q is None:
            raise RuntimeError("Adam pq controller is not initialized.")

        p = torch.clamp(self._adam_p, 0.0, self.p_probability_upper)
        add_param = torch.clamp(self._adam_q, 0.0, float(self._adam_add_parameter_upper()))
        q = torch.clamp(self._q_from_add_parameter_tensor(add_param), 0.0, self.q_probability_upper)
        return p, q

    @torch.no_grad()
    def _project_adam_state_(self) -> None:
        if self._adam_p is None or self._adam_q is None:
            raise RuntimeError("Adam pq controller is not initialized.")
        self._adam_p.copy_(
            torch.nan_to_num(
                self._adam_p,
                nan=0.0,
                posinf=self.p_probability_upper,
                neginf=0.0,
            )
        )
        self._adam_q.copy_(
            torch.nan_to_num(
                self._adam_q,
                nan=0.0,
                posinf=float(self._adam_add_parameter_upper()),
                neginf=0.0,
            )
        )
        if self._adam_opt is not None:
            for state in self._adam_opt.state.values():
                for value in state.values():
                    if torch.is_tensor(value):
                        value.copy_(torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))
        self._adam_p.clamp_(0.0, self.p_probability_upper)
        self._adam_q.clamp_(0.0, float(self._adam_add_parameter_upper()))

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

        p = float(max(0.0, min(p, self.p_probability_upper)))
        q = float(max(0.0, min(q, self.q_probability_upper)))
        return p, q

    def _make_objective(
        self,
        *,
        cache: BatchGraphCache,
        params,
        idx: torch.Tensor,
        weight: torch.Tensor,
        msg: torch.Tensor,
        msg_sq: torch.Tensor,
        log_gd: float,
        eps: float,
        batch_nodes: int,
        empirical_gcn_variance: Optional[torch.Tensor] = None,
    ) -> Callable[[float, float], Tuple[float, float]]:
        if self.gnn == "gin":
            v_drop, v_add = self._var_gin_bases(cache, msg, msg_sq, batch_nodes)
            r_drop = 0.5 * (weight[idx] * v_drop[idx]).sum() / max(int(idx.numel()), 1)
            r_add = 0.5 * (weight[idx] * v_add[idx]).sum() / max(int(idx.numel()), 1)

            g_drop = torch.autograd.grad(r_drop, params, retain_graph=True, create_graph=False, allow_unused=True)
            g_add = torch.autograd.grad(r_add, params, retain_graph=True, create_graph=False, allow_unused=True)

            a = _dot(g_drop, g_drop)
            b = _dot(g_add, g_add)
            c = _dot(g_drop, g_add)

            def objective(p_try: float, q_try: float) -> Tuple[float, float]:
                s = p_try / max(1.0 - p_try, eps)
                t = q_try * (1.0 - q_try)
                g2 = (s * s) * a + (t * t) * b + 2.0 * s * t * c
                g_reg = math.sqrt(max(g2, 0.0))
                obj = (math.log(g_reg + eps) - log_gd) ** 2 + self._objective_penalty_float(
                    float(p_try), float(q_try)
                )
                return float(obj), float(g_reg)

            return objective

        if self.gnn in {"gcn", "gat"}:
            def objective(p_try: float, q_try: float) -> Tuple[float, float]:
                if self.gnn == "gcn":
                    v = (
                        empirical_gcn_variance
                        if empirical_gcn_variance is not None
                        else self._var_gcn(cache, float(p_try), float(q_try), msg, msg_sq, batch_nodes)
                    )
                else:
                    v = self._var_gat(cache, float(p_try), float(q_try), msg, msg_sq)
                regularizer = 0.5 * (weight[idx] * v[idx]).sum() / max(int(idx.numel()), 1)
                g_reg = torch.autograd.grad(
                    regularizer,
                    params,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )
                greg = _grad_norm(g_reg)
                obj = (math.log(greg + eps) - log_gd) ** 2 + self._objective_penalty_float(
                    float(p_try), float(q_try)
                )
                return float(obj), float(greg)

            return objective

        raise ValueError(f"Unsupported gnn={self.gnn}. Expected 'gin', 'gcn', or 'gat'.")

    def _make_objective_tensor(
        self,
        *,
        cache: BatchGraphCache,
        params,
        idx: torch.Tensor,
        weight: torch.Tensor,
        msg: torch.Tensor,
        msg_sq: torch.Tensor,
        log_gd: float,
        eps: float,
        batch_nodes: int,
        empirical_gcn_variance: Optional[torch.Tensor] = None,
    ) -> Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        log_gd_t = weight.new_tensor(float(log_gd))
        eps_t = weight.new_tensor(float(eps))

        if self.gnn == "gin":
            v_drop, v_add = self._var_gin_bases(cache, msg, msg_sq, batch_nodes)
            r_drop = 0.5 * (weight[idx] * v_drop[idx]).sum() / max(int(idx.numel()), 1)
            r_add = 0.5 * (weight[idx] * v_add[idx]).sum() / max(int(idx.numel()), 1)

            g_drop = torch.autograd.grad(r_drop, params, retain_graph=True, create_graph=False, allow_unused=True)
            g_add = torch.autograd.grad(r_add, params, retain_graph=True, create_graph=False, allow_unused=True)

            a = weight.new_tensor(_dot(g_drop, g_drop))
            b = weight.new_tensor(_dot(g_add, g_add))
            c = weight.new_tensor(_dot(g_drop, g_add))

            def objective(p_try: torch.Tensor, q_try: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                one = torch.ones_like(p_try)
                s = p_try / (one - p_try).clamp_min(float(eps))
                t = q_try * (one - q_try)
                g2 = (s * s) * a + (t * t) * b + 2.0 * s * t * c
                g_reg = torch.sqrt(torch.clamp_min(g2, 0.0))
                obj = (torch.log(g_reg + eps_t) - log_gd_t) ** 2 + self._objective_penalty_tensor(
                    p_try, q_try
                )
                return obj, g_reg

            return objective

        if self.gnn in {"gcn", "gat"}:
            def objective(p_try: torch.Tensor, q_try: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                if self.gnn == "gcn":
                    v = (
                        empirical_gcn_variance + (p_try * 0.0 + q_try * 0.0)
                        if empirical_gcn_variance is not None
                        else self._var_gcn_tensor(cache, p_try, q_try, msg, msg_sq, batch_nodes)
                    )
                else:
                    v = self._var_gat_tensor(cache, p_try, q_try, msg, msg_sq)
                regularizer = 0.5 * (weight[idx] * v[idx]).sum() / max(int(idx.numel()), 1)
                g_reg = torch.autograd.grad(
                    regularizer,
                    params,
                    retain_graph=True,
                    create_graph=True,
                    allow_unused=True,
                )
                g2 = weight.new_zeros(())
                for grad in g_reg:
                    if grad is None:
                        continue
                    g2 = g2 + (grad * grad).sum()
                greg = torch.sqrt(torch.clamp_min(g2, 0.0))
                obj = (torch.log(greg + eps_t) - log_gd_t) ** 2 + self._objective_penalty_tensor(
                    p_try, q_try
                )
                return obj, greg

            return objective

        raise ValueError(f"Unsupported gnn={self.gnn}. Expected 'gin', 'gcn', or 'gat'.")

    def _search_grid(
        self,
        objective: Callable[[float, float], Tuple[float, float]],
        *,
        p_max: float,
        q_max: float,
        device: torch.device,
    ) -> Tuple[float, float, float, float, Dict[str, float]]:
        p_grid = [0.0] if p_max <= 0.0 else torch.linspace(
            0.0, float(p_max), steps=int(self.cfg.grid_size), device=device
        ).tolist()
        q_grid = self._build_q_grid(q_max)

        best_obj = float("inf")
        best_p = 0.0
        best_q = 0.0
        best_g = 0.0
        n_eval = 0

        for p_try in p_grid:
            for q_try in q_grid:
                obj, greg = objective(float(p_try), float(q_try))
                n_eval += 1
                if obj < best_obj:
                    best_obj = float(obj)
                    best_p = float(p_try)
                    best_q = float(q_try)
                    best_g = float(greg)

        return best_p, best_q, best_g, best_obj, {"n_eval": float(n_eval), "solver_success": 1.0}

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
            raise ImportError("search_method='powell' requires scipy.")

        active = []
        if p_max > 0.0:
            active.append("p")
        if q_max > 0.0:
            active.append("q")

        if not active:
            obj0, greg0 = objective(0.0, 0.0)
            return 0.0, 0.0, greg0, obj0, {"n_eval": 1.0, "solver_success": 1.0}

        def unpack(x) -> Tuple[float, float]:
            idx = 0
            p_try, q_try = 0.0, 0.0
            if "p" in active:
                p_try = float(x[idx]) * float(p_max)
                idx += 1
            if "q" in active:
                q_try = float(x[idx]) * float(q_max)
            return p_try, q_try

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

        best_seen = {"obj": float("inf"), "p": 0.0, "q": 0.0, "greg": 0.0, "n_eval": 0}

        def fun_np(x) -> float:
            p_try, q_try = unpack(x)
            obj, greg = objective(p_try, q_try)
            best_seen["n_eval"] += 1
            if obj < best_seen["obj"]:
                best_seen["obj"] = float(obj)
                best_seen["p"] = float(p_try)
                best_seen["q"] = float(q_try)
                best_seen["greg"] = float(greg)
            return float(obj)

        res = minimize(
            fun_np,
            x0=x0,
            method="Powell",
            bounds=bounds,
            options={
                "maxiter": int(self.cfg.powell_maxiter),
                "xtol": float(self.cfg.powell_xtol),
                "ftol": float(self.cfg.powell_ftol),
                "disp": False,
            },
        )

        return (
            float(best_seen["p"]),
            float(best_seen["q"]),
            float(best_seen["greg"]),
            float(best_seen["obj"]),
            {"n_eval": float(best_seen["n_eval"]), "solver_success": 1.0 if bool(res.success) else 0.0},
        )

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

        zero = torch.zeros((), device=device, dtype=torch.float32)

        def unpack(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            idx = 0
            p_try = zero
            q_try = zero
            if "p" in active:
                p_try = x[idx] * float(p_max)
                idx += 1
            if "q" in active:
                q_try = x[idx] * float(q_max)
            return p_try, q_try

        if not active:
            obj0, greg0 = objective(zero, zero)
            return 0.0, 0.0, float(greg0.detach()), float(obj0.detach()), {"n_eval": 1.0, "solver_success": 1.0}

        x0 = []
        if "p" in active:
            p0 = float(max(0.0, min(current_p, p_max)))
            x0.append(0.0 if p_max <= 0.0 else p0 / float(p_max))
        if "q" in active:
            q0 = float(max(0.0, min(current_q, q_max)))
            x0.append(0.0 if q_max <= 0.0 else q0 / float(q_max))
        x_best = torch.tensor(x0, device=device, dtype=torch.float32)

        n_eval = 0
        solver_success = 0.0
        damping = max(float(self.cfg.newton_damping), 1e-8)

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
            for idx in range(x_curr.numel()):
                rows.append(
                    torch.autograd.grad(grad[idx], x_curr, retain_graph=(idx + 1) < x_curr.numel())[0]
                )
            hess = torch.stack(rows, dim=0)
            hess = 0.5 * (hess + hess.transpose(0, 1))
            eye = torch.eye(x_curr.numel(), device=device, dtype=torch.float32)

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
                        damping = max(trial_damping * 0.5, max(float(self.cfg.newton_damping), 1e-8))
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
        obj_final, greg_final = objective(p_final, q_final)
        n_eval += 1
        return (
            float(p_final.detach()),
            float(q_final.detach()),
            float(greg_final.detach()),
            float(obj_final.detach()),
            {"n_eval": float(n_eval), "solver_success": float(solver_success)},
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
        objective_tensor: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[float, float, float, float, Dict[str, float]]:
        search_method = _validate_search_method(search_method)
        if search_method == "grid":
            return self._search_grid(objective, p_max=p_max, q_max=q_max, device=device)
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

    def _format_search_result(
        self,
        method: str,
        p_best: float,
        q_best: float,
        g_reg_best: float,
        obj_best: float,
        meta: Dict[str, float],
    ) -> Dict[str, float]:
        rho_best = self._rho_from_q_float(float(q_best))
        return {
            "method": str(method),
            "p_best": float(p_best),
            "q_best": float(q_best),
            "rho_best": float(rho_best),
            "G_reg_best": float(g_reg_best),
            "obj": float(obj_best),
            "deletion_penalty_best": float(self._deletion_penalty_float(p_best)),
            "densification_penalty_best": float(self._densification_penalty_float(rho_best)),
            "rho_penalty_best": float(self._densification_penalty_float(rho_best)),
            "n_eval": float(meta.get("n_eval", 0.0)),
            "solver_success": float(meta.get("solver_success", 0.0)),
        }

    def _collect_search_results(
        self,
        *,
        objective: Callable[[float, float], Tuple[float, float]],
        objective_tensor: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]],
        current_p: float,
        current_q: float,
        p_max: float,
        q_max: float,
        device: torch.device,
        primary_method: str,
        primary_result: Dict[str, float],
    ) -> Dict[str, Dict[str, float]]:
        results = {str(primary_method): dict(primary_result)}
        for method in SEARCH_METHODS:
            if method in results:
                continue
            p_try, q_try, g_try, obj_try, meta_try = self._run_search(
                method,
                objective,
                current_p=current_p,
                current_q=current_q,
                p_max=p_max,
                q_max=q_max,
                device=device,
                objective_tensor=objective_tensor if method == "newton" else None,
            )
            results[method] = self._format_search_result(method, p_try, q_try, g_try, obj_try, meta_try)
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
            if "rho_best" in result:
                info[f"comparison_{method}_rho_best"] = float(result["rho_best"])
            info[f"comparison_{method}_G_reg_best"] = float(result["G_reg_best"])
            if "deletion_penalty_best" in result:
                info[f"comparison_{method}_deletion_penalty_best"] = float(result["deletion_penalty_best"])
            info[f"comparison_{method}_n_eval"] = float(result["n_eval"])
            info[f"comparison_{method}_solver_success"] = float(result["solver_success"])
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

        p_values = torch.linspace(0.0, float(p_max), steps=p_size, device="cpu")
        q_values = torch.linspace(0.0, float(q_max), steps=q_size, device="cpu")

        obj_grid = torch.zeros((q_values.numel(), p_values.numel()), dtype=torch.float64)
        g_grid = torch.zeros_like(obj_grid)
        for q_idx, q_try in enumerate(q_values.tolist()):
            for p_idx, p_try in enumerate(p_values.tolist()):
                obj_val, g_val = objective(float(p_try), float(q_try))
                obj_grid[q_idx, p_idx] = float(obj_val)
                g_grid[q_idx, p_idx] = float(g_val)

        return {
            "p_values": p_values.tolist(),
            "q_values": q_values.tolist(),
            "objective": obj_grid.tolist(),
            "G_reg": g_grid.tolist(),
        }

    def _prepare_batch_inputs(
        self,
        model: nn.Module,
        x_b: torch.Tensor,
        edge_index_b_clean: torch.Tensor,
        y_b: torch.Tensor,
        train_mask_b: torch.Tensor,
        epoch_seed: int,
        batch_id: int,
        *,
        node_ids_b: Optional[torch.Tensor] = None,
    ) -> Optional[Dict[str, Any]]:
        cfg = self.cfg
        device = next(model.parameters()).device

        x_b = x_b.to(device)
        edge_index_b_clean = edge_index_b_clean.to(device=device, dtype=torch.long)
        y_b = y_b.to(device)
        if y_b.dim() == 2 and y_b.size(1) == 1:
            y_b = y_b.squeeze(1)
        y_b = y_b.view(-1).to(torch.long)
        train_mask_b = train_mask_b.to(device=device, dtype=torch.bool)
        node_ids_b = None if node_ids_b is None else node_ids_b.to(device=device, dtype=torch.long)

        idx_all = torch.where(train_mask_b)[0]
        if idx_all.numel() == 0:
            return None

        idx = _pick_subset(idx_all, cfg.subset_nodes, seed=int(epoch_seed) + int(batch_id) + int(cfg.seed))
        devices = []
        if x_b.is_cuda and x_b.device.index is not None:
            devices = [x_b.device.index]

        return {
            "x_b": x_b,
            "edge_index_b_clean": edge_index_b_clean,
            "y_b": y_b,
            "train_mask_b": train_mask_b,
            "node_ids_b": node_ids_b,
            "idx_all": idx_all,
            "idx": idx,
            "batch_nodes": int(x_b.size(0)),
            "device": device,
            "devices": devices,
        }

    def _sample_anchor_masks(
        self,
        *,
        edge_index_b_clean: torch.Tensor,
        batch_nodes: int,
        aug_mode: str,
        epoch_seed: int,
        batch_id: int,
        p0: float,
        q0: float,
        device: torch.device,
        mask_sharing: str,
        num_layers: int,
    ) -> Tuple[Optional[Any], Optional[Any]]:
        mode0 = str(aug_mode).lower().strip()
        if mode0 == "none" or (p0 <= 0.0 and q0 <= 0.0):
            return None, None

        base_seed = int(epoch_seed) + 777_000 + int(batch_id)
        if str(mask_sharing).lower().strip() == "shared":
            _, edge_index_keep, edge_index_add, _ = BatchBernoulliEdgeAugmentor.augment_edge_indices(
                edge_index_clean=edge_index_b_clean,
                num_nodes=batch_nodes,
                p=float(p0),
                q=float(q0),
                mode=mode0,
                seed=base_seed,
                device=device,
            )
            return edge_index_keep, edge_index_add

        keep_list = []
        add_list = []
        for layer in range(max(1, int(num_layers))):
            _, edge_index_keep_l, edge_index_add_l, _ = BatchBernoulliEdgeAugmentor.augment_edge_indices(
                edge_index_clean=edge_index_b_clean,
                num_nodes=batch_nodes,
                p=float(p0),
                q=float(q0),
                mode=mode0,
                seed=base_seed + 1_000_000 * layer,
                device=device,
            )
            keep_list.append(edge_index_keep_l)
            add_list.append(edge_index_add_l)
        return keep_list, add_list

    def _build_batch_gdata_meta(
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
        *,
        node_ids_b: Optional[torch.Tensor] = None,
        mask_sharing: str = "shared",
        num_layers: int = 1,
    ) -> Optional[Dict[str, Any]]:
        prep = self._prepare_batch_inputs(
            model,
            x_b,
            edge_index_b_clean,
            y_b,
            train_mask_b,
            epoch_seed,
            batch_id,
            node_ids_b=node_ids_b,
        )
        if prep is None:
            return None

        eps = self.cfg.eps
        was_training = model.training
        model.train()
        bn_states = _freeze_batchnorm_buffers(model)

        x_b = prep["x_b"]
        edge_index_b_clean = prep["edge_index_b_clean"]
        y_b = prep["y_b"]
        node_ids_b = prep["node_ids_b"]
        idx = prep["idx"]
        idx_all = prep["idx_all"]
        batch_nodes = int(prep["batch_nodes"])
        device = prep["device"]
        devices = prep["devices"]

        with torch.random.fork_rng(devices=devices, enabled=True):
            seed = int(epoch_seed) + 1_000_003 * int(batch_id) + int(self.cfg.seed)
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.set_device(device)
                torch.cuda.manual_seed(seed)

            model.zero_grad(set_to_none=True)

            if self.data_anchor == "aug":
                edge_index_keep0, edge_index_add0 = self._sample_anchor_masks(
                    edge_index_b_clean=edge_index_b_clean,
                    batch_nodes=batch_nodes,
                    aug_mode=aug_mode,
                    epoch_seed=epoch_seed,
                    batch_id=batch_id,
                    p0=float(current_p),
                    q0=float(current_q),
                    device=device,
                    mask_sharing=mask_sharing,
                    num_layers=num_layers,
                )
                if edge_index_keep0 is None and edge_index_add0 is None:
                    logits_anchor = model(x_b, edge_index_b_clean, p=0.0, q=0.0, node_ids=node_ids_b)
                else:
                    logits_anchor = model(
                        x_b,
                        edge_index_b_clean,
                        edge_index_keep=edge_index_keep0,
                        edge_index_add=edge_index_add0,
                        p=float(current_p),
                        q=float(current_q),
                        node_ids=node_ids_b,
                    )
            else:
                logits_anchor = model(x_b, edge_index_b_clean, p=0.0, q=0.0, node_ids=node_ids_b)

            loss_data = criterion(logits_anchor[idx], y_b[idx])
            params = [param for param in model.parameters() if param.requires_grad]
            g_data = torch.autograd.grad(loss_data, params, retain_graph=False, create_graph=False, allow_unused=True)
            g_data_norm = _grad_norm(g_data)
            log_gd = math.log(g_data_norm + eps)

        _restore_batchnorm_buffers(bn_states)
        if not was_training:
            model.eval()

        return {
            "batch_train_nodes": float(idx_all.numel()),
            "batch_train_used": float(idx.numel()),
            "G_data": float(g_data_norm),
            "log_gd": float(log_gd),
        }

    def _build_batch_snapshot(
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
        *,
        node_ids_b: Optional[torch.Tensor] = None,
        mask_sharing: str = "shared",
        num_layers: int = 1,
    ) -> Optional[Dict[str, Any]]:
        cfg = self.cfg
        eps = cfg.eps
        prep = self._prepare_batch_inputs(
            model,
            x_b,
            edge_index_b_clean,
            y_b,
            train_mask_b,
            epoch_seed,
            batch_id,
            node_ids_b=node_ids_b,
        )
        if prep is None:
            return None

        x_b = prep["x_b"]
        edge_index_b_clean = prep["edge_index_b_clean"]
        y_b = prep["y_b"]
        node_ids_b = prep["node_ids_b"]
        idx_all = prep["idx_all"]
        idx = prep["idx"]
        batch_nodes = int(prep["batch_nodes"])
        device = prep["device"]
        devices = prep["devices"]

        cache = BatchGraphCache.from_edge_index(
            edge_index_b_clean,
            batch_nodes,
            node_ids=node_ids_b,
            global_num_nodes_total=getattr(model, "global_num_nodes", batch_nodes),
        )

        was_training = model.training
        model.train()
        bn_states = _freeze_batchnorm_buffers(model)

        with torch.random.fork_rng(devices=devices, enabled=True):
            seed = int(epoch_seed) + 1_000_003 * int(batch_id) + int(cfg.seed)
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.set_device(device)
                torch.cuda.manual_seed(seed)

            model.zero_grad(set_to_none=True)

            if self.data_anchor == "aug":
                edge_index_keep0, edge_index_add0 = self._sample_anchor_masks(
                    edge_index_b_clean=edge_index_b_clean,
                    batch_nodes=batch_nodes,
                    aug_mode=aug_mode,
                    epoch_seed=epoch_seed,
                    batch_id=batch_id,
                    p0=float(current_p),
                    q0=float(current_q),
                    device=device,
                    mask_sharing=mask_sharing,
                    num_layers=num_layers,
                )
                if edge_index_keep0 is None and edge_index_add0 is None:
                    logits_anchor = model(x_b, edge_index_b_clean, p=0.0, q=0.0, node_ids=node_ids_b)
                else:
                    logits_anchor = model(
                        x_b,
                        edge_index_b_clean,
                        edge_index_keep=edge_index_keep0,
                        edge_index_add=edge_index_add0,
                        p=float(current_p),
                        q=float(current_q),
                        node_ids=node_ids_b,
                    )
            else:
                logits_anchor = model(x_b, edge_index_b_clean, p=0.0, q=0.0, node_ids=node_ids_b)

            loss_data = criterion(logits_anchor[idx], y_b[idx])
            params = [param for param in model.parameters() if param.requires_grad]
            g_data = torch.autograd.grad(loss_data, params, retain_graph=False, create_graph=False, allow_unused=True)
            g_data_norm = _grad_norm(g_data)
            log_gd = math.log(g_data_norm + eps)

            model.zero_grad(set_to_none=True)
            logits, emb = model(
                x_b,
                edge_index_b_clean,
                p=0.0,
                q=0.0,
                node_ids=node_ids_b,
                return_embeddings=True,
            )

            weight = _w_multiclass(logits)
            msg, msg_sq = self._messages_detached(emb, model.pred_local.weight)
            self._empirical_gcn_modules = (
                self._collect_empirical_gcn_modules(model)
                if self.ep_expectation_mode == "empirical_ema" and self.gnn == "gcn"
                else []
            )
            self._gat_modules = self._collect_gat_modules(model) if self.gnn == "gat" else []
            empirical_gcn_variance = (
                self._empirical_gcn_variance(msg, msg_sq, p=float(current_p), q=float(current_q), eps=float(eps))
                if self.ep_expectation_mode == "empirical_ema" and self.gnn == "gcn"
                else None
            )

            aug_mode_norm = str(aug_mode).lower().strip()
            p_max = self.search_p_max if aug_mode_norm in ("both", "drop") else 0.0
            q_max = self.search_q_max if aug_mode_norm in ("both", "add") else 0.0

        _restore_batchnorm_buffers(bn_states)
        if not was_training:
            model.eval()

        return {
            "cache": cache,
            "params": params,
            "idx": idx,
            "weight": weight,
            "msg": msg,
            "msg_sq": msg_sq,
            "empirical_gcn_variance": empirical_gcn_variance,
            "log_gd": float(log_gd),
            "eps": float(eps),
            "batch_nodes": int(batch_nodes),
            "batch_train_nodes": float(idx_all.numel()),
            "batch_train_used": float(idx.numel()),
            "G_data": float(g_data_norm),
            "p_max": float(p_max),
            "q_max": float(q_max),
        }

    def _build_snapshot_objectives(
        self,
        snapshot: Dict[str, Any],
        *,
        include_tensor: bool = False,
    ) -> Tuple[
        Callable[[float, float], Tuple[float, float]],
        Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]],
    ]:
        objective = self._make_objective(
            cache=snapshot["cache"],
            params=snapshot["params"],
            idx=snapshot["idx"],
            weight=snapshot["weight"],
            msg=snapshot["msg"],
            msg_sq=snapshot["msg_sq"],
            log_gd=snapshot["log_gd"],
            eps=snapshot["eps"],
            batch_nodes=snapshot["batch_nodes"],
            empirical_gcn_variance=snapshot.get("empirical_gcn_variance"),
        )

        objective_tensor = None
        if include_tensor:
            objective_tensor = self._make_objective_tensor(
                cache=snapshot["cache"],
                params=snapshot["params"],
                idx=snapshot["idx"],
                weight=snapshot["weight"],
                msg=snapshot["msg"],
                msg_sq=snapshot["msg_sq"],
                log_gd=snapshot["log_gd"],
                eps=snapshot["eps"],
                batch_nodes=snapshot["batch_nodes"],
                empirical_gcn_variance=snapshot.get("empirical_gcn_variance"),
            )

        return objective, objective_tensor

    def _build_gin_joint_stats_from_snapshot(self, snapshot: Dict[str, Any]) -> Dict[str, float]:
        cache = snapshot["cache"]
        params = snapshot["params"]
        idx = snapshot["idx"]
        weight = snapshot["weight"]
        msg = snapshot["msg"]
        msg_sq = snapshot["msg_sq"]
        batch_nodes = snapshot["batch_nodes"]

        v_drop, v_add = self._var_gin_bases(cache, msg, msg_sq, batch_nodes)
        r_drop = 0.5 * (weight[idx] * v_drop[idx]).sum() / max(int(idx.numel()), 1)
        r_add = 0.5 * (weight[idx] * v_add[idx]).sum() / max(int(idx.numel()), 1)

        g_drop = torch.autograd.grad(r_drop, params, retain_graph=True, create_graph=False, allow_unused=True)
        g_add = torch.autograd.grad(r_add, params, retain_graph=True, create_graph=False, allow_unused=True)

        return {
            "a": float(_dot(g_drop, g_drop)),
            "b": float(_dot(g_add, g_add)),
            "c": float(_dot(g_drop, g_add)),
            "log_gd": float(snapshot["log_gd"]),
            "G_data": float(snapshot["G_data"]),
            "batch_train_nodes": float(snapshot["batch_train_nodes"]),
            "batch_train_used": float(snapshot["batch_train_used"]),
        }

    def _make_joint_epoch_objective_from_gin_stats(
        self,
        cached_gin_stats: List[Dict[str, Any]],
        *,
        include_tensor: bool,
    ) -> Tuple[
        Callable[[float, float], Tuple[float, float]],
        Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]],
    ]:
        eps = float(self.cfg.eps)
        batch_specs = []
        total_weight = 0.0
        for item in cached_gin_stats:
            weight_b = float(item["weight"])
            if weight_b <= 0.0:
                continue
            batch_specs.append((weight_b, item))
            total_weight += weight_b

        if total_weight <= 0.0:
            raise ValueError("Joint GIN objective requires at least one non-empty cached batch statistic.")

        def objective(p_try: float, q_try: float) -> Tuple[float, float]:
            s = float(p_try) / max(1.0 - float(p_try), eps)
            t = float(q_try) * (1.0 - float(q_try))
            obj_sum = 0.0
            greg_sum = 0.0
            for weight_b, stats in batch_specs:
                g2 = (s * s) * float(stats["a"]) + (t * t) * float(stats["b"]) + 2.0 * s * t * float(stats["c"])
                greg = math.sqrt(max(g2, 0.0))
                obj = (math.log(greg + eps) - float(stats["log_gd"])) ** 2
                obj_sum += weight_b * obj
                greg_sum += weight_b * greg
            return (
                obj_sum / total_weight + self._objective_penalty_float(float(p_try), float(q_try)),
                greg_sum / total_weight,
            )

        objective_tensor = None
        if include_tensor:

            def objective_tensor(p_try: torch.Tensor, q_try: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                one = torch.ones_like(p_try)
                s = p_try / (one - p_try).clamp_min(eps)
                t = q_try * (one - q_try)
                obj_sum_t = torch.zeros((), device=p_try.device, dtype=p_try.dtype)
                greg_sum_t = torch.zeros((), device=p_try.device, dtype=p_try.dtype)
                eps_t = torch.tensor(eps, device=p_try.device, dtype=p_try.dtype)
                for weight_b, stats in batch_specs:
                    a_t = torch.tensor(float(stats["a"]), device=p_try.device, dtype=p_try.dtype)
                    b_t = torch.tensor(float(stats["b"]), device=p_try.device, dtype=p_try.dtype)
                    c_t = torch.tensor(float(stats["c"]), device=p_try.device, dtype=p_try.dtype)
                    log_gd_t = torch.tensor(float(stats["log_gd"]), device=p_try.device, dtype=p_try.dtype)
                    g2 = (s * s) * a_t + (t * t) * b_t + 2.0 * s * t * c_t
                    greg = torch.sqrt(torch.clamp_min(g2, 0.0))
                    obj = (torch.log(greg + eps_t) - log_gd_t) ** 2
                    obj_sum_t = obj_sum_t + float(weight_b) * obj
                    greg_sum_t = greg_sum_t + float(weight_b) * greg
                denom = float(total_weight)
                return obj_sum_t / denom + self._objective_penalty_tensor(p_try, q_try), greg_sum_t / denom

        return objective, objective_tensor

    def _solve_joint_epoch_objective_from_gin_stats(
        self,
        *,
        cached_gin_stats: List[Dict[str, Any]],
        current_p: float,
        current_q: float,
        aug_mode: str,
        device: torch.device,
        capture_surface: bool,
        surface_grid_size: Optional[int],
    ) -> Dict[str, Any]:
        cfg = self.cfg
        if len(cached_gin_stats) == 0:
            return {}

        p_max = self.search_p_max if aug_mode in ("both", "drop") else 0.0
        q_max = self.search_q_max if aug_mode in ("both", "add") else 0.0
        include_tensor = (self.search_method == "newton") or bool(cfg.compare_search_methods) or bool(capture_surface)
        objective, objective_tensor = self._make_joint_epoch_objective_from_gin_stats(
            cached_gin_stats,
            include_tensor=include_tensor,
        )

        best_p, best_q, best_greg, best_obj, search_meta = self._run_search(
            self.search_method,
            objective,
            current_p=float(current_p),
            current_q=float(current_q),
            p_max=p_max,
            q_max=q_max,
            device=device,
            objective_tensor=objective_tensor,
        )
        primary_result = self._format_search_result(
            self.search_method,
            best_p,
            best_q,
            best_greg,
            best_obj,
            search_meta,
        )

        search_results = {self.search_method: dict(primary_result)}
        if bool(cfg.compare_search_methods) or bool(capture_surface):
            search_results = self._collect_search_results(
                objective=objective,
                objective_tensor=objective_tensor,
                current_p=float(current_p),
                current_q=float(current_q),
                p_max=p_max,
                q_max=q_max,
                device=device,
                primary_method=self.search_method,
                primary_result=primary_result,
            )
            selected = search_results[self.search_method]
            best_p = float(selected["p_best"])
            best_q = float(selected["q_best"])
            best_greg = float(selected["G_reg_best"])
            best_obj = float(selected["obj"])
            search_meta = {
                "n_eval": float(selected["n_eval"]),
                "solver_success": float(selected["solver_success"]),
            }

        comparison_info = self._make_comparison_info(search_results) if bool(cfg.compare_search_methods) else {}
        surface_data = None
        if capture_surface:
            surface_data = self._sample_objective_surface(
                objective,
                p_max=p_max,
                q_max=q_max,
                grid_size=int(surface_grid_size or max(11, int(cfg.grid_size))),
            )
            surface_data["search_results"] = {method: dict(result) for method, result in search_results.items()}
            surface_data["selected_method"] = str(self.search_method)
            surface_data["p_max"] = float(p_max)
            surface_data["q_max"] = float(q_max)
            self._annotate_surface_data(surface_data, q_max=q_max)

        info: Dict[str, Any] = {
            "p_epoch_best": float(best_p),
            "q_epoch_best": float(best_q),
            "rho_epoch_best": float(self._rho_from_q_float(best_q)),
            "G_reg_best_mean": float(best_greg),
            "obj_mean": float(best_obj),
            "deletion_penalty_best": float(self._deletion_penalty_float(best_p)),
            "densification_penalty_best": float(
                self._densification_penalty_float(self._rho_from_q_float(best_q))
            ),
            "rho_penalty_best": float(self._densification_penalty_float(self._rho_from_q_float(best_q))),
            **search_meta,
        }
        info.update(comparison_info)
        if surface_data is not None:
            info["surface_data"] = surface_data
        return info

    @staticmethod
    def _freeze_joint_batch_inputs(
        *,
        x_b: torch.Tensor,
        edge_index_b_clean: torch.Tensor,
        y_b: torch.Tensor,
        train_mask_b: torch.Tensor,
        batch_id: int,
        node_ids_b: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        return {
            "x": x_b.detach().cpu(),
            "edge_index": edge_index_b_clean.detach().cpu(),
            "y": y_b.detach().cpu(),
            "train_mask": train_mask_b.detach().cpu(),
            "node_ids": None if node_ids_b is None else node_ids_b.detach().cpu(),
            "batch_id": int(batch_id),
        }

    def _solve_snapshot(
        self,
        snapshot: Dict[str, Any],
        *,
        current_p: float,
        current_q: float,
        aug_mode: str,
        device: torch.device,
        capture_surface: bool = False,
        surface_grid_size: Optional[int] = None,
    ) -> Tuple[float, float, Dict[str, Any]]:
        cfg = self.cfg
        aug_mode = str(aug_mode).lower().strip()
        p_max = float(snapshot["p_max"])
        q_max = float(snapshot["q_max"])
        g_data_norm = float(snapshot["G_data"])
        need_all_search_results = bool(cfg.compare_search_methods) or bool(capture_surface)

        objective, objective_tensor = self._build_snapshot_objectives(
            snapshot,
            include_tensor=(self.search_method in {"newton", "adam"} or need_all_search_results),
        )

        with torch.random.fork_rng(
            devices=[device.index] if device.type == "cuda" and device.index is not None else [],
            enabled=True,
        ):
            if self.search_method == "adam":
                if bool(cfg.compare_search_methods):
                    raise ValueError("pq_search_method='adam' does not support --pq_compare_search_methods.")
                if objective_tensor is None:
                    raise RuntimeError("pq_search_method='adam' requires a differentiable objective.")

                self._ensure_adam_state(
                    current_p=float(current_p),
                    current_q=float(current_q),
                    device=device,
                )

                p_before, q_before = self._adam_rates_tensor()
                if aug_mode == "drop":
                    q_before = q_before * 0.0
                elif aug_mode == "add":
                    p_before = p_before * 0.0
                elif aug_mode == "none":
                    p_before = p_before * 0.0
                    q_before = q_before * 0.0

                self._adam_opt.zero_grad(set_to_none=True)
                obj_t, greg_t = objective_tensor(p_before, q_before)
                obj_t.backward(retain_graph=True)
                self._adam_opt.step()
                self._project_adam_state_()

                p_new, q_new = self._adam_rates_float(aug_mode)
                obj_new, greg_new = objective(float(p_new), float(q_new))

                best_p = float(p_before.detach().item())
                best_q = float(q_before.detach().item())
                best_greg = float(greg_t.detach().item())
                best_obj = float(obj_t.detach().item())

                search_meta = {
                    "n_eval": 2.0,
                    "solver_success": 1.0,
                    "adam_lr": float(cfg.adam_lr),
                    "optimization_variable": self._add_parameter_name(),
                }

                surface_data = None
                if capture_surface:
                    p_surface_upper = self.p_probability_upper if aug_mode in ("both", "drop") else 0.0
                    q_surface_upper = self.q_probability_upper if aug_mode in ("both", "add") else 0.0
                    surface_data = self._sample_objective_surface(
                        objective,
                        p_max=p_surface_upper,
                        q_max=q_surface_upper,
                        grid_size=int(surface_grid_size or max(11, int(cfg.grid_size))),
                    )
                    surface_data["search_results"] = {
                        "adam": {
                            "method": "adam",
                            "p_best": float(best_p),
                            "q_best": float(best_q),
                            "rho_best": float(self._rho_from_q_float(best_q)),
                            "G_reg_best": float(best_greg),
                            "obj": float(best_obj),
                            "p_new": float(p_new),
                            "q_new": float(q_new),
                            "rho_new": float(self._rho_from_q_float(q_new)),
                            "G_reg_new": float(greg_new),
                            "obj_new": float(obj_new),
                            "deletion_penalty_best": float(self._deletion_penalty_float(best_p)),
                            "deletion_penalty_new": float(self._deletion_penalty_float(p_new)),
                            "densification_penalty_best": float(
                                self._densification_penalty_float(self._rho_from_q_float(best_q))
                            ),
                            "densification_penalty_new": float(
                                self._densification_penalty_float(self._rho_from_q_float(q_new))
                            ),
                            "n_eval": 2.0,
                            "solver_success": 1.0,
                        }
                    }
                    surface_data["selected_method"] = "adam"
                    surface_data["optimization_variable"] = self._add_parameter_name()
                    surface_data["p_max"] = float(p_surface_upper)
                    surface_data["q_max"] = float(q_surface_upper)
                    surface_data["G_data"] = float(g_data_norm)
                    self._annotate_surface_data(surface_data, q_max=q_surface_upper)

                info: Dict[str, Any] = {
                    "G_data": float(g_data_norm),
                    "G_reg_best": float(best_greg),
                    "p_best": float(best_p),
                    "q_best": float(best_q),
                    "rho_best": float(self._rho_from_q_float(best_q)),
                    "p_new": float(p_new),
                    "q_new": float(q_new),
                    "rho_new": float(self._rho_from_q_float(q_new)),
                    "obj": float(best_obj),
                    "G_reg_new": float(greg_new),
                    "obj_new": float(obj_new),
                    "deletion_penalty_best": float(self._deletion_penalty_float(best_p)),
                    "densification_penalty_best": float(
                        self._densification_penalty_float(self._rho_from_q_float(best_q))
                    ),
                    "rho_penalty_best": float(self._densification_penalty_float(self._rho_from_q_float(best_q))),
                    "deletion_penalty_new": float(self._deletion_penalty_float(p_new)),
                    "densification_penalty_new": float(
                        self._densification_penalty_float(self._rho_from_q_float(q_new))
                    ),
                    "rho_penalty_new": float(self._densification_penalty_float(self._rho_from_q_float(q_new))),
                    "batch_train_nodes": float(snapshot["batch_train_nodes"]),
                    "batch_train_used": float(snapshot["batch_train_used"]),
                    "search_method": "adam",
                    "optimize_rho": bool(self.optimize_rho),
                    "optimization_variable": self._add_parameter_name(),
                    "deletion_penalty_lambda": float(getattr(cfg, "deletion_penalty_lambda", 0.0)),
                    "densification_penalty_lambda": float(getattr(cfg, "densification_penalty_lambda", 0.0)),
                    "densification_penalty_type": str(self.penalty_type),
                    "densification_penalty_rho0": float(self.penalty_rho0),
                    **search_meta,
                }
                if surface_data is not None:
                    info["surface_data"] = surface_data
                return float(p_new), float(q_new), info

            best_p, best_q, best_greg, best_obj, search_meta = self._run_search(
                self.search_method,
                objective,
                current_p=float(current_p),
                current_q=float(current_q),
                p_max=p_max,
                q_max=q_max,
                device=device,
                objective_tensor=objective_tensor,
            )
            primary_result = self._format_search_result(
                self.search_method,
                best_p,
                best_q,
                best_greg,
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
                    device=device,
                    primary_method=self.search_method,
                    primary_result=primary_result,
                )
                selected = search_results[self.search_method]
                best_p = float(selected["p_best"])
                best_q = float(selected["q_best"])
                best_greg = float(selected["G_reg_best"])
                best_obj = float(selected["obj"])
                search_meta = {
                    "n_eval": float(selected["n_eval"]),
                    "solver_success": float(selected["solver_success"]),
                }

            comparison_info = self._make_comparison_info(search_results) if bool(cfg.compare_search_methods) else {}
            surface_data = None
            if capture_surface:
                surface_data = self._sample_objective_surface(
                    objective,
                    p_max=p_max,
                    q_max=q_max,
                    grid_size=int(surface_grid_size or max(11, int(cfg.grid_size))),
                )
                surface_data["search_results"] = {
                    method: dict(result) for method, result in (search_results or {self.search_method: primary_result}).items()
                }
                surface_data["selected_method"] = str(self.search_method)
                surface_data["p_max"] = float(p_max)
                surface_data["q_max"] = float(q_max)
                surface_data["G_data"] = float(g_data_norm)
                self._annotate_surface_data(surface_data, q_max=q_max)

        if aug_mode == "drop":
            best_q = 0.0
        elif aug_mode == "add":
            best_p = 0.0
        elif aug_mode == "none":
            best_p, best_q = 0.0, 0.0

        best_p = float(max(0.0, min(best_p, self.search_p_max)))
        best_q = float(max(0.0, min(best_q, self.search_q_max)))

        info: Dict[str, Any] = {
            "G_data": float(g_data_norm),
            "G_reg_best": float(best_greg),
            "p_best": float(best_p),
            "q_best": float(best_q),
            "rho_best": float(self._rho_from_q_float(best_q)),
            "obj": float(best_obj),
            "deletion_penalty_best": float(self._deletion_penalty_float(best_p)),
            "densification_penalty_best": float(
                self._densification_penalty_float(self._rho_from_q_float(best_q))
            ),
            "rho_penalty_best": float(self._densification_penalty_float(self._rho_from_q_float(best_q))),
            "batch_train_nodes": float(snapshot["batch_train_nodes"]),
            "batch_train_used": float(snapshot["batch_train_used"]),
            "search_method": str(self.search_method),
            "optimize_rho": bool(self.optimize_rho),
            "deletion_penalty_lambda": float(getattr(cfg, "deletion_penalty_lambda", 0.0)),
            "densification_penalty_lambda": float(getattr(cfg, "densification_penalty_lambda", 0.0)),
            "densification_penalty_type": str(self.penalty_type),
            "densification_penalty_rho0": float(self.penalty_rho0),
            **search_meta,
        }
        info.update(comparison_info)
        if search_results is not None:
            info["search_results"] = {method: dict(result) for method, result in search_results.items()}
        if surface_data is not None:
            info["surface_data"] = surface_data
        return best_p, best_q, info

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
        *,
        node_ids_b: Optional[torch.Tensor] = None,
        mask_sharing: str = "shared",
        num_layers: int = 1,
        capture_surface: bool = False,
        surface_grid_size: Optional[int] = None,
    ) -> Tuple[float, float, Dict[str, Any]]:
        device = next(model.parameters()).device
        snapshot = self._build_batch_snapshot(
            model=model,
            x_b=x_b,
            edge_index_b_clean=edge_index_b_clean,
            y_b=y_b,
            train_mask_b=train_mask_b,
            criterion=criterion,
            current_p=current_p,
            current_q=current_q,
            aug_mode=aug_mode,
            epoch_seed=epoch_seed,
            batch_id=batch_id,
            node_ids_b=node_ids_b,
            mask_sharing=mask_sharing,
            num_layers=num_layers,
        )
        if snapshot is None:
            return 0.0, 0.0, {"skip": 1.0}
        return self._solve_snapshot(
            snapshot,
            current_p=current_p,
            current_q=current_q,
            aug_mode=aug_mode,
            device=device,
            capture_surface=capture_surface,
            surface_grid_size=surface_grid_size,
        )

    def _make_joint_epoch_objective(
        self,
        *,
        model: nn.Module,
        frozen_batches: List[Dict[str, Any]],
        criterion,
        current_p: float,
        current_q: float,
        aug_mode: str,
        epoch_seed: int,
        mask_sharing: str,
        num_layers: int,
        include_tensor: bool,
    ) -> Tuple[
        Callable[[float, float], Tuple[float, float]],
        Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]],
    ]:
        total_weight = 0.0
        batch_specs = []

        for batch_data in frozen_batches:
            weight_b = float(batch_data["weight"])
            if weight_b <= 0.0:
                continue
            batch_specs.append((weight_b, batch_data))
            total_weight += weight_b

        if total_weight <= 0.0:
            raise ValueError("Joint epoch objective requires at least one non-empty frozen batch.")

        def _build_objectives_for_batch(
            batch_data: Dict[str, Any],
            *,
            need_tensor: bool,
        ) -> Tuple[
            Callable[[float, float], Tuple[float, float]],
            Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]],
        ]:
            snapshot = self._build_batch_snapshot(
                model=model,
                x_b=batch_data["x"],
                edge_index_b_clean=batch_data["edge_index"],
                y_b=batch_data["y"],
                train_mask_b=batch_data["train_mask"],
                criterion=criterion,
                current_p=current_p,
                current_q=current_q,
                aug_mode=aug_mode,
                epoch_seed=epoch_seed,
                batch_id=int(batch_data["batch_id"]),
                node_ids_b=batch_data.get("node_ids"),
                mask_sharing=mask_sharing,
                num_layers=num_layers,
            )
            if snapshot is None:
                raise ValueError("Joint epoch objective encountered an empty batch after freezing inputs.")
            return self._build_snapshot_objectives(snapshot, include_tensor=need_tensor)

        def objective(p_try: float, q_try: float) -> Tuple[float, float]:
            obj_sum = 0.0
            greg_sum = 0.0
            for weight_b, batch_data in batch_specs:
                objective_b, _ = _build_objectives_for_batch(batch_data, need_tensor=False)
                obj_b, greg_b = objective_b(float(p_try), float(q_try))
                obj_sum += weight_b * float(obj_b)
                greg_sum += weight_b * float(greg_b)
            return obj_sum / total_weight, greg_sum / total_weight

        objective_tensor = None
        if include_tensor:

            def objective_tensor(p_try: torch.Tensor, q_try: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                ref = p_try if isinstance(p_try, torch.Tensor) else q_try
                obj_sum_t = torch.zeros((), device=ref.device, dtype=ref.dtype)
                greg_sum_t = torch.zeros((), device=ref.device, dtype=ref.dtype)
                for weight_b, batch_data in batch_specs:
                    _, objective_tensor_b = _build_objectives_for_batch(batch_data, need_tensor=True)
                    if objective_tensor_b is None:
                        raise ValueError("Tensor objective requested but a per-batch tensor objective is missing.")
                    obj_b, greg_b = objective_tensor_b(p_try, q_try)
                    obj_sum_t = obj_sum_t + float(weight_b) * obj_b
                    greg_sum_t = greg_sum_t + float(weight_b) * greg_b
                denom = float(total_weight)
                return obj_sum_t / denom, greg_sum_t / denom

        return objective, objective_tensor

    def _solve_joint_epoch_objective(
        self,
        *,
        model: nn.Module,
        frozen_batches: List[Dict[str, Any]],
        criterion,
        current_p: float,
        current_q: float,
        aug_mode: str,
        epoch_seed: int,
        mask_sharing: str,
        num_layers: int,
        device: torch.device,
        capture_surface: bool,
        surface_grid_size: Optional[int],
    ) -> Dict[str, Any]:
        cfg = self.cfg
        if len(frozen_batches) == 0:
            return {}

        p_max = self.search_p_max if aug_mode in ("both", "drop") else 0.0
        q_max = self.search_q_max if aug_mode in ("both", "add") else 0.0
        include_tensor = (self.search_method == "newton") or bool(cfg.compare_search_methods) or bool(capture_surface)

        objective, objective_tensor = self._make_joint_epoch_objective(
            model=model,
            frozen_batches=frozen_batches,
            criterion=criterion,
            current_p=current_p,
            current_q=current_q,
            aug_mode=aug_mode,
            epoch_seed=epoch_seed,
            mask_sharing=mask_sharing,
            num_layers=num_layers,
            include_tensor=include_tensor,
        )

        best_p, best_q, best_greg, best_obj, search_meta = self._run_search(
            self.search_method,
            objective,
            current_p=float(current_p),
            current_q=float(current_q),
            p_max=p_max,
            q_max=q_max,
            device=device,
            objective_tensor=objective_tensor,
        )
        primary_result = self._format_search_result(
            self.search_method,
            best_p,
            best_q,
            best_greg,
            best_obj,
            search_meta,
        )

        search_results = {self.search_method: dict(primary_result)}
        if bool(cfg.compare_search_methods) or bool(capture_surface):
            search_results = self._collect_search_results(
                objective=objective,
                objective_tensor=objective_tensor,
                current_p=float(current_p),
                current_q=float(current_q),
                p_max=p_max,
                q_max=q_max,
                device=device,
                primary_method=self.search_method,
                primary_result=primary_result,
            )
            selected = search_results[self.search_method]
            best_p = float(selected["p_best"])
            best_q = float(selected["q_best"])
            best_greg = float(selected["G_reg_best"])
            best_obj = float(selected["obj"])
            search_meta = {
                "n_eval": float(selected["n_eval"]),
                "solver_success": float(selected["solver_success"]),
            }

        comparison_info = self._make_comparison_info(search_results) if bool(cfg.compare_search_methods) else {}
        surface_data = None
        if capture_surface:
            surface_data = self._sample_objective_surface(
                objective,
                p_max=p_max,
                q_max=q_max,
                grid_size=int(surface_grid_size or max(11, int(cfg.grid_size))),
            )
            surface_data["search_results"] = {method: dict(result) for method, result in search_results.items()}
            surface_data["selected_method"] = str(self.search_method)
            surface_data["p_max"] = float(p_max)
            surface_data["q_max"] = float(q_max)
            self._annotate_surface_data(surface_data, q_max=q_max)

        info: Dict[str, Any] = {
            "p_epoch_best": float(best_p),
            "q_epoch_best": float(best_q),
            "rho_epoch_best": float(self._rho_from_q_float(best_q)),
            "G_reg_best_mean": float(best_greg),
            "obj_mean": float(best_obj),
            "deletion_penalty_best": float(self._deletion_penalty_float(best_p)),
            "densification_penalty_best": float(
                self._densification_penalty_float(self._rho_from_q_float(best_q))
            ),
            "rho_penalty_best": float(self._densification_penalty_float(self._rho_from_q_float(best_q))),
            **search_meta,
        }
        info.update(comparison_info)
        if surface_data is not None:
            info["surface_data"] = surface_data
        return info

    def _init_gcn_joint_grid_stream_state(
        self,
        *,
        p_max: float,
        q_max: float,
        capture_surface: bool,
        surface_grid_size: Optional[int],
    ) -> Dict[str, Any]:
        # Stream one full batch snapshot through the whole grid, accumulate epoch totals,
        # then discard the snapshot before moving to the next batch.
        p_values = [0.0] if p_max <= 0.0 else torch.linspace(
            0.0, float(p_max), steps=int(self.cfg.grid_size), device="cpu"
        ).tolist()
        q_values = self._build_q_grid(q_max)

        state: Dict[str, Any] = {
            "p_values": [float(p) for p in p_values],
            "q_values": [float(q) for q in q_values],
            "objective_accum": torch.zeros((len(q_values), len(p_values)), dtype=torch.float64),
            "greg_accum": torch.zeros((len(q_values), len(p_values)), dtype=torch.float64),
            "total_weight": 0.0,
        }

        if capture_surface:
            p_size = max(2, int(surface_grid_size or max(11, int(self.cfg.grid_size)))) if p_max > 0.0 else 1
            q_size = max(2, int(surface_grid_size or max(11, int(self.cfg.grid_size)))) if q_max > 0.0 else 1
            p_surface = torch.linspace(0.0, float(p_max), steps=p_size, device="cpu").tolist()
            q_surface = torch.linspace(0.0, float(q_max), steps=q_size, device="cpu").tolist()
            state["surface_p_values"] = [float(p) for p in p_surface]
            state["surface_q_values"] = [float(q) for q in q_surface]
            state["surface_objective_accum"] = torch.zeros((len(q_surface), len(p_surface)), dtype=torch.float64)
            state["surface_greg_accum"] = torch.zeros((len(q_surface), len(p_surface)), dtype=torch.float64)

        return state

    def _accumulate_gcn_joint_grid_stream_snapshot(
        self,
        snapshot: Dict[str, Any],
        *,
        weight_b: float,
        state: Dict[str, Any],
    ) -> None:
        objective, _ = self._build_snapshot_objectives(snapshot, include_tensor=False)

        for q_idx, q_try in enumerate(state["q_values"]):
            for p_idx, p_try in enumerate(state["p_values"]):
                obj_val, greg_val = objective(float(p_try), float(q_try))
                state["objective_accum"][q_idx, p_idx] += float(weight_b) * float(obj_val)
                state["greg_accum"][q_idx, p_idx] += float(weight_b) * float(greg_val)

        if "surface_p_values" in state and "surface_q_values" in state:
            for q_idx, q_try in enumerate(state["surface_q_values"]):
                for p_idx, p_try in enumerate(state["surface_p_values"]):
                    obj_val, greg_val = objective(float(p_try), float(q_try))
                    state["surface_objective_accum"][q_idx, p_idx] += float(weight_b) * float(obj_val)
                    state["surface_greg_accum"][q_idx, p_idx] += float(weight_b) * float(greg_val)

        state["total_weight"] += float(weight_b)

    def _finalize_gcn_joint_grid_stream_state(
        self,
        state: Dict[str, Any],
        *,
        p_max: float,
        q_max: float,
    ) -> Dict[str, Any]:
        total_weight = float(state["total_weight"])
        if total_weight <= 0.0:
            return {}

        obj_mean = state["objective_accum"] / total_weight
        greg_mean = state["greg_accum"] / total_weight

        best_flat = int(torch.argmin(obj_mean).item())
        num_p = len(state["p_values"])
        q_idx = best_flat // num_p
        p_idx = best_flat % num_p
        best_p = float(state["p_values"][p_idx])
        best_q = float(state["q_values"][q_idx])
        best_obj = float(obj_mean[q_idx, p_idx].item())
        best_greg = float(greg_mean[q_idx, p_idx].item())

        info: Dict[str, Any] = {
            "p_epoch_best": best_p,
            "q_epoch_best": best_q,
            "rho_epoch_best": float(self._rho_from_q_float(best_q)),
            "G_reg_best_mean": best_greg,
            "obj_mean": best_obj,
            "deletion_penalty_best": float(self._deletion_penalty_float(best_p)),
            "densification_penalty_best": float(
                self._densification_penalty_float(self._rho_from_q_float(best_q))
            ),
            "rho_penalty_best": float(self._densification_penalty_float(self._rho_from_q_float(best_q))),
            "n_eval": float(len(state["p_values"]) * len(state["q_values"])),
            "solver_success": 1.0,
        }

        if "surface_p_values" in state and "surface_q_values" in state:
            info["surface_data"] = {
                "p_values": list(state["surface_p_values"]),
                "q_values": list(state["surface_q_values"]),
                "objective": (state["surface_objective_accum"] / total_weight).tolist(),
                "G_reg": (state["surface_greg_accum"] / total_weight).tolist(),
                "p_max": float(p_max),
                "q_max": float(q_max),
                "search_results": {
                    "grid": {
                        "method": "grid",
                        "p_best": best_p,
                        "q_best": best_q,
                        "rho_best": float(self._rho_from_q_float(best_q)),
                        "G_reg_best": best_greg,
                        "obj": best_obj,
                        "deletion_penalty_best": float(self._deletion_penalty_float(best_p)),
                        "densification_penalty_best": float(
                            self._densification_penalty_float(self._rho_from_q_float(best_q))
                        ),
                        "n_eval": float(len(state["p_values"]) * len(state["q_values"])),
                        "solver_success": 1.0,
                    }
                },
                "selected_method": "grid",
            }
            self._annotate_surface_data(info["surface_data"], q_max=q_max)

        return info

    def suggest_pq_for_epoch(
        self,
        model: nn.Module,
        batches: Iterable[Any],
        criterion,
        current_p: float,
        current_q: float,
        aug_mode: str,
        epoch_seed: int,
        *,
        mask_sharing: str = "shared",
        num_layers: int = 1,
        capture_surface: bool = False,
        surface_grid_size: Optional[int] = None,
    ) -> Tuple[float, float, Dict[str, Any]]:
        cfg = self.cfg
        aug_mode = str(aug_mode).lower().strip()

        if self.search_method == "adam":
            raise ValueError(
                "pq_search_method='adam' uses online mini-batch updates via suggest_pq_for_batch(), "
                "not epoch-level aggregation via suggest_pq_for_epoch()."
            )

        device = next(model.parameters()).device
        epoch_mode = str(self.epoch_objective_mode)
        cached_gin_joint_stats: List[Dict[str, Any]] = []
        frozen_joint_batches: List[Dict[str, Any]] = []
        # For GCN joint-objective grid search, reuse one live snapshot per batch across
        # all candidate points, then drop it immediately to avoid retaining epoch-wide
        # snapshot state on GPU.
        use_streamed_gcn_joint_grid = (
            epoch_mode == "joint_objective"
            and self.gnn == "gcn"
            and self.search_method == "grid"
            and not bool(cfg.compare_search_methods)
        )
        gcn_joint_grid_state: Optional[Dict[str, Any]] = None

        sum_w = 0.0
        sum_p = 0.0
        sum_q = 0.0
        g_data_list: List[float] = []
        g_reg_list: List[float] = []
        obj_list: List[float] = []
        used_batches = 0
        skipped_batches = 0

        method_agg = {
            method: {
                "obj": 0.0,
                "p_best": 0.0,
                "q_best": 0.0,
                "rho_best": 0.0,
                "G_reg_best": 0.0,
                "deletion_penalty_best": 0.0,
                "densification_penalty_best": 0.0,
                "n_eval": 0.0,
                "solver_success": 0.0,
                "weight": 0.0,
            }
            for method in SEARCH_METHODS
        }

        surface_accum = None
        surface_weight = 0.0
        surface_axes = None

        max_batches = int(cfg.max_batches_for_tuning)
        for batch_id, batch in enumerate(batches):
            if max_batches > 0 and used_batches >= max_batches:
                break

            if isinstance(batch, (tuple, list)) and len(batch) >= 4:
                x_b, edge_index_b, y_b, train_mask_b = batch[0], batch[1], batch[2], batch[3]
                node_ids_b = batch[4] if len(batch) > 4 else None
            elif isinstance(batch, dict):
                x_b = batch["x"]
                edge_index_b = batch["edge_index"]
                y_b = batch["y"]
                train_mask_b = batch["train_mask"]
                node_ids_b = batch.get("node_ids")
            elif hasattr(batch, "x") and hasattr(batch, "edge_index") and hasattr(batch, "y"):
                x_b = batch.x
                edge_index_b = batch.edge_index
                y_b = batch.y
                seed_n = int(getattr(batch, "batch_size", x_b.size(0)))
                train_mask_b = torch.zeros((x_b.size(0),), device=x_b.device, dtype=torch.bool)
                train_mask_b[:seed_n] = True
                node_ids_b = getattr(batch, "n_id", None)
            else:
                raise ValueError("Unsupported batch format for PQ tuning.")

            if y_b.dim() == 2 and y_b.size(1) == 1:
                y_b_use = y_b.squeeze(1)
            else:
                y_b_use = y_b

            if use_streamed_gcn_joint_grid:
                snapshot = self._build_batch_snapshot(
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
                    batch_id=int(batch_id),
                    node_ids_b=node_ids_b,
                    mask_sharing=mask_sharing,
                    num_layers=num_layers,
                )
                if snapshot is None:
                    skipped_batches += 1
                    continue

                batch_train_nodes = float(snapshot.get("batch_train_nodes", 0.0))
                if batch_train_nodes <= 0.0:
                    skipped_batches += 1
                    continue

                weight_b = self._epoch_batch_weight(snapshot)
                used_batches += 1
                g_data_list.append(float(snapshot["G_data"]))

                if gcn_joint_grid_state is None:
                    gcn_joint_grid_state = self._init_gcn_joint_grid_stream_state(
                        p_max=float(snapshot["p_max"]),
                        q_max=float(snapshot["q_max"]),
                        capture_surface=bool(capture_surface),
                        surface_grid_size=surface_grid_size,
                    )

                self._accumulate_gcn_joint_grid_stream_snapshot(
                    snapshot,
                    weight_b=weight_b,
                    state=gcn_joint_grid_state,
                )
                continue

            if epoch_mode == "joint_objective":
                if self.gnn == "gin":
                    snapshot = self._build_batch_snapshot(
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
                        batch_id=int(batch_id),
                        node_ids_b=node_ids_b,
                        mask_sharing=mask_sharing,
                        num_layers=num_layers,
                    )
                    if snapshot is None:
                        skipped_batches += 1
                        continue

                    batch_train_nodes = float(snapshot.get("batch_train_nodes", 0.0))
                    if batch_train_nodes <= 0.0:
                        skipped_batches += 1
                        continue

                    weight_b = self._epoch_batch_weight(snapshot)
                    used_batches += 1
                    g_data_list.append(float(snapshot["G_data"]))
                    cached_gin_joint_stats.append(
                        {
                            **self._build_gin_joint_stats_from_snapshot(snapshot),
                            "weight": float(weight_b),
                        }
                    )
                    continue

                meta = self._build_batch_gdata_meta(
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
                    batch_id=int(batch_id),
                    node_ids_b=node_ids_b,
                    mask_sharing=mask_sharing,
                    num_layers=num_layers,
                )
                if meta is None:
                    skipped_batches += 1
                    continue

                batch_train_nodes = float(meta.get("batch_train_nodes", 0.0))
                if batch_train_nodes <= 0.0:
                    skipped_batches += 1
                    continue

                weight_b = self._epoch_batch_weight(meta)
                used_batches += 1
                g_data_list.append(float(meta["G_data"]))
                frozen_joint_batches.append(
                    {
                        **self._freeze_joint_batch_inputs(
                            x_b=x_b,
                            edge_index_b_clean=edge_index_b,
                            y_b=y_b_use,
                            train_mask_b=train_mask_b,
                            batch_id=int(batch_id),
                            node_ids_b=node_ids_b,
                        ),
                        "weight": float(weight_b),
                    }
                )
                continue

            snapshot = self._build_batch_snapshot(
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
                batch_id=int(batch_id),
                node_ids_b=node_ids_b,
                mask_sharing=mask_sharing,
                num_layers=num_layers,
            )

            if snapshot is None:
                skipped_batches += 1
                continue

            batch_train_nodes = float(snapshot.get("batch_train_nodes", 0.0))
            if batch_train_nodes <= 0.0:
                skipped_batches += 1
                continue

            weight_b = self._epoch_batch_weight(snapshot)
            used_batches += 1
            g_data_list.append(float(snapshot["G_data"]))

            batch_capture_surface = bool(capture_surface)
            p_b, q_b, info_b = self._solve_snapshot(
                snapshot,
                current_p=current_p,
                current_q=current_q,
                aug_mode=aug_mode,
                device=device,
                capture_surface=batch_capture_surface,
                surface_grid_size=surface_grid_size,
            )

            sum_w += weight_b
            sum_p += weight_b * float(p_b)
            sum_q += weight_b * float(q_b)
            g_reg_list.append(float(info_b["G_reg_best"]))
            obj_list.append(float(info_b["obj"]))

            search_results = info_b.get("search_results", None)
            if search_results is None:
                search_results = {
                    str(info_b["search_method"]): {
                        "obj": float(info_b["obj"]),
                        "p_best": float(info_b["p_best"]),
                        "q_best": float(info_b["q_best"]),
                        "rho_best": float(info_b.get("rho_best", self._rho_from_q_float(float(info_b["q_best"])))),
                        "G_reg_best": float(info_b["G_reg_best"]),
                        "deletion_penalty_best": float(
                            info_b.get(
                                "deletion_penalty_best",
                                self._deletion_penalty_float(float(info_b["p_best"])),
                            )
                        ),
                        "densification_penalty_best": float(
                            info_b.get(
                                "densification_penalty_best",
                                self._densification_penalty_float(
                                    self._rho_from_q_float(float(info_b["q_best"]))
                                ),
                            )
                        ),
                        "n_eval": float(info_b.get("n_eval", 0.0)),
                        "solver_success": float(info_b.get("solver_success", 0.0)),
                    }
                }

            for method, result in search_results.items():
                method_agg[method]["obj"] += weight_b * float(result["obj"])
                method_agg[method]["p_best"] += weight_b * float(result["p_best"])
                method_agg[method]["q_best"] += weight_b * float(result["q_best"])
                method_agg[method]["rho_best"] += weight_b * float(
                    result.get("rho_best", self._rho_from_q_float(float(result["q_best"])))
                )
                method_agg[method]["G_reg_best"] += weight_b * float(result["G_reg_best"])
                method_agg[method]["deletion_penalty_best"] += weight_b * float(
                    result.get(
                        "deletion_penalty_best",
                        self._deletion_penalty_float(float(result["p_best"])),
                    )
                )
                method_agg[method]["densification_penalty_best"] += weight_b * float(
                    result.get(
                        "densification_penalty_best",
                        self._densification_penalty_float(self._rho_from_q_float(float(result["q_best"]))),
                    )
                )
                method_agg[method]["n_eval"] += weight_b * float(result.get("n_eval", 0.0))
                method_agg[method]["solver_success"] += weight_b * float(result.get("solver_success", 0.0))
                method_agg[method]["weight"] += weight_b

            if capture_surface and "surface_data" in info_b:
                surface_data = info_b["surface_data"]
                obj_grid = torch.tensor(surface_data["objective"], dtype=torch.float64)
                g_grid = torch.tensor(surface_data["G_reg"], dtype=torch.float64)
                if surface_accum is None:
                    surface_accum = {
                        "objective": torch.zeros_like(obj_grid),
                        "G_reg": torch.zeros_like(g_grid),
                    }
                    surface_axes = {
                        "p_values": list(surface_data["p_values"]),
                        "q_values": list(surface_data["q_values"]),
                        "p_max": float(surface_data["p_max"]),
                        "q_max": float(surface_data["q_max"]),
                    }
                surface_accum["objective"] += weight_b * obj_grid
                surface_accum["G_reg"] += weight_b * g_grid
                surface_weight += weight_b

        if epoch_mode == "joint_objective":
            if self.gnn == "gin":
                joint_empty = len(cached_gin_joint_stats) == 0
            elif use_streamed_gcn_joint_grid:
                joint_empty = (gcn_joint_grid_state is None) or (float(gcn_joint_grid_state["total_weight"]) <= 0.0)
            else:
                joint_empty = len(frozen_joint_batches) == 0

            if joint_empty:
                p_best = float(current_p)
                q_best = float(current_q)
                p_new = float(current_p)
                q_new = float(current_q)
                info = {
                    "p_epoch_best": p_best,
                    "q_epoch_best": q_best,
                    "rho_epoch_best": float(self._rho_from_q_float(q_best)),
                    "p_epoch_avg_best": p_best,
                    "q_epoch_avg_best": q_best,
                    "p_new": p_new,
                    "q_new": q_new,
                    "rho_new": float(self._rho_from_q_float(q_new)),
                    "used_batches": float(used_batches),
                    "skipped_batches": float(skipped_batches),
                    "G_data_mean": 0.0,
                    "G_reg_best_mean": 0.0,
                    "obj_mean": 0.0,
                    "epoch_objective_mode": str(epoch_mode),
                    "deletion_penalty_best": float(self._deletion_penalty_float(p_best)),
                    "deletion_penalty_new": float(self._deletion_penalty_float(p_new)),
                    "deletion_penalty_lambda": float(getattr(cfg, "deletion_penalty_lambda", 0.0)),
                }
                return p_new, q_new, info

            if self.gnn == "gin":
                joint_info = self._solve_joint_epoch_objective_from_gin_stats(
                    cached_gin_stats=cached_gin_joint_stats,
                    current_p=current_p,
                    current_q=current_q,
                    aug_mode=aug_mode,
                    device=device,
                    capture_surface=capture_surface,
                    surface_grid_size=surface_grid_size,
                )
            elif use_streamed_gcn_joint_grid:
                p_max = self.search_p_max if aug_mode in ("both", "drop") else 0.0
                q_max = self.search_q_max if aug_mode in ("both", "add") else 0.0
                joint_info = self._finalize_gcn_joint_grid_stream_state(
                    gcn_joint_grid_state,
                    p_max=p_max,
                    q_max=q_max,
                )
            else:
                joint_info = self._solve_joint_epoch_objective(
                    model=model,
                    frozen_batches=frozen_joint_batches,
                    criterion=criterion,
                    current_p=current_p,
                    current_q=current_q,
                    aug_mode=aug_mode,
                    epoch_seed=epoch_seed,
                    mask_sharing=mask_sharing,
                    num_layers=num_layers,
                    device=device,
                    capture_surface=capture_surface,
                    surface_grid_size=surface_grid_size,
                )

            p_best = float(joint_info["p_epoch_best"])
            q_best = float(joint_info["q_epoch_best"])
            p_new = cfg.ema * float(current_p) + (1.0 - cfg.ema) * p_best
            q_new = cfg.ema * float(current_q) + (1.0 - cfg.ema) * q_best

            if aug_mode == "drop":
                q_new = 0.0
            elif aug_mode == "add":
                p_new = 0.0
            elif aug_mode == "none":
                p_new, q_new = 0.0, 0.0

            p_new = float(max(0.0, min(p_new, self.search_p_max)))
            q_new = float(max(0.0, min(q_new, self.search_q_max)))

            def _mean(values: List[float]) -> float:
                return float(sum(values) / max(len(values), 1))

            info = {
                "p_epoch_best": p_best,
                "q_epoch_best": q_best,
                "rho_epoch_best": float(self._rho_from_q_float(q_best)),
                "p_epoch_avg_best": p_best,
                "q_epoch_avg_best": q_best,
                "p_new": float(p_new),
                "q_new": float(q_new),
                "rho_new": float(self._rho_from_q_float(q_new)),
                "used_batches": float(used_batches),
                "skipped_batches": float(skipped_batches),
                "G_data_mean": _mean(g_data_list),
                "G_reg_best_mean": float(joint_info["G_reg_best_mean"]),
                "obj_mean": float(joint_info["obj_mean"]),
                "epoch_objective_mode": str(epoch_mode),
                "optimize_rho": bool(self.optimize_rho),
                "deletion_penalty_new": float(self._deletion_penalty_float(p_new)),
                "deletion_penalty_lambda": float(getattr(cfg, "deletion_penalty_lambda", 0.0)),
                "densification_penalty_lambda": float(getattr(cfg, "densification_penalty_lambda", 0.0)),
                "densification_penalty_type": str(self.penalty_type),
                "densification_penalty_rho0": float(self.penalty_rho0),
            }
            for key, value in joint_info.items():
                if key in {"p_epoch_best", "q_epoch_best", "G_reg_best_mean", "obj_mean"}:
                    continue
                info[key] = value
            return p_new, q_new, info

        if sum_w <= 0.0 or used_batches == 0:
            p_avg = float(current_p)
            q_avg = float(current_q)
        else:
            p_avg = sum_p / sum_w
            q_avg = sum_q / sum_w

        p_new = cfg.ema * float(current_p) + (1.0 - cfg.ema) * float(p_avg)
        q_new = cfg.ema * float(current_q) + (1.0 - cfg.ema) * float(q_avg)

        if aug_mode == "drop":
            q_new = 0.0
        elif aug_mode == "add":
            p_new = 0.0
        elif aug_mode == "none":
            p_new, q_new = 0.0, 0.0

        p_new = float(max(0.0, min(p_new, self.search_p_max)))
        q_new = float(max(0.0, min(q_new, self.search_q_max)))

        def _mean(values: List[float]) -> float:
            return float(sum(values) / max(len(values), 1))

        aggregated_results = {}
        if bool(cfg.compare_search_methods) or bool(capture_surface):
            for method in SEARCH_METHODS:
                weight = method_agg[method]["weight"]
                if weight <= 0.0:
                    continue
                aggregated_results[method] = {
                    "obj": method_agg[method]["obj"] / weight,
                    "p_best": method_agg[method]["p_best"] / weight,
                    "q_best": method_agg[method]["q_best"] / weight,
                    "rho_best": method_agg[method]["rho_best"] / weight,
                    "G_reg_best": method_agg[method]["G_reg_best"] / weight,
                    "deletion_penalty_best": method_agg[method]["deletion_penalty_best"] / weight,
                    "densification_penalty_best": method_agg[method]["densification_penalty_best"] / weight,
                    "n_eval": method_agg[method]["n_eval"] / weight,
                    "solver_success": method_agg[method]["solver_success"] / weight,
                }

        info: Dict[str, Any] = {
            "p_epoch_best": float(p_avg),
            "q_epoch_best": float(q_avg),
            "rho_epoch_best": float(self._rho_from_q_float(q_avg)),
            "p_epoch_avg_best": float(p_avg),
            "q_epoch_avg_best": float(q_avg),
            "p_new": float(p_new),
            "q_new": float(q_new),
            "rho_new": float(self._rho_from_q_float(q_new)),
            "used_batches": float(used_batches),
            "skipped_batches": float(skipped_batches),
            "G_data_mean": _mean(g_data_list),
            "G_reg_best_mean": _mean(g_reg_list),
            "obj_mean": _mean(obj_list),
            "epoch_objective_mode": str(epoch_mode),
            "optimize_rho": bool(self.optimize_rho),
            "deletion_penalty_best": float(self._deletion_penalty_float(p_avg)),
            "deletion_penalty_new": float(self._deletion_penalty_float(p_new)),
            "deletion_penalty_lambda": float(getattr(cfg, "deletion_penalty_lambda", 0.0)),
            "densification_penalty_lambda": float(getattr(cfg, "densification_penalty_lambda", 0.0)),
            "densification_penalty_type": str(self.penalty_type),
            "densification_penalty_rho0": float(self.penalty_rho0),
        }

        if bool(cfg.compare_search_methods):
            info.update(self._make_comparison_info(aggregated_results))

        if capture_surface and surface_accum is not None and surface_weight > 0.0 and surface_axes is not None:
            info["surface_data"] = {
                "p_values": surface_axes["p_values"],
                "q_values": surface_axes["q_values"],
                "objective": (surface_accum["objective"] / surface_weight).tolist(),
                "G_reg": (surface_accum["G_reg"] / surface_weight).tolist(),
                "p_max": float(surface_axes["p_max"]),
                "q_max": float(surface_axes["q_max"]),
                "search_results": {method: dict(result) for method, result in aggregated_results.items()},
                "selected_method": str(self.search_method),
            }
            self._annotate_surface_data(info["surface_data"], q_max=float(surface_axes["q_max"]))

        return p_new, q_new, info
