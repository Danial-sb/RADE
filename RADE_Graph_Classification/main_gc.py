# graph_classification/main_gc.py
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

try:
    import wandb
except Exception:
    wandb = None


def _load_viz_pq_module():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    module_path = os.path.join(repo_root, "RADE_Node_Classification", "full_batch", "viz_pq.py")
    spec = importlib.util.spec_from_file_location("rade_node_viz_pq", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load viz_pq module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_VIZ_PQ_MODULE = _load_viz_pq_module()
save_pq_plots = _VIZ_PQ_MODULE.save_pq_plots

GC_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    from .augmentation_gc import BernoulliEdgeAugmentor
    from .dataset_gc import load_gc_dataset, load_powerful_gnns_tu_dataset, make_powerful_gnns_tu_folds
    from .eval_gc import evaluate_loader, evaluate_splits, masked_bce_with_logits
    from .parse_gc import apply_gc_aug_defaults, apply_gc_auto_defaults, get_explicit_arg_names, parse_method_gc, parser_add_gc_args
    from .pq_gradnorm_gc import PQGradNormTunerGC, PQTunerGCConfig
except Exception:
    from augmentation_gc import BernoulliEdgeAugmentor
    from dataset_gc import load_gc_dataset, load_powerful_gnns_tu_dataset, make_powerful_gnns_tu_folds
    from eval_gc import evaluate_loader, evaluate_splits, masked_bce_with_logits
    from parse_gc import apply_gc_aug_defaults, apply_gc_auto_defaults, get_explicit_arg_names, parse_method_gc, parser_add_gc_args
    from pq_gradnorm_gc import PQGradNormTunerGC, PQTunerGCConfig


EdgeIndexObj = Union[torch.Tensor, List[torch.Tensor]]


def fix_seed(seed: int, device: Optional[torch.device] = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_deterministic_debug_mode("error")


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _sync_for_timing(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _gc_path(*parts: str) -> str:
    return os.path.join(GC_DIR, *parts)


def _enforce_aug_mode(p_val: float, q_val: float, aug_mode: str) -> Tuple[float, float]:
    aug_mode = str(aug_mode).lower().strip()
    if aug_mode == "drop":
        return float(p_val), 0.0
    if aug_mode == "add":
        return 0.0, float(q_val)
    if aug_mode == "none":
        return 0.0, 0.0
    return float(p_val), float(q_val)


def _require_no_rade_contrib(args, name: str) -> None:
    if bool(getattr(args, "ep_correction", False)):
        raise ValueError(f"--aug_tech={name} must be used with --ep_correction False.")
    if bool(getattr(args, "pq_gradnorm", False)):
        raise ValueError(f"--aug_tech={name} must be used with --pq_gradnorm False.")
    if str(getattr(args, "aug_mode", "none")).lower().strip() != "none":
        raise ValueError(f"--aug_tech={name} requires --aug_mode none (edge augmentation disabled).")
    if bool(getattr(args, "linear", False)):
        raise ValueError(f"--aug_tech={name} is incompatible with --linear.")


def _resolve_pq_batch_update_interval(args, train_loader) -> Tuple[int, Optional[int]]:
    num_batches: Optional[int]
    try:
        num_batches = int(len(train_loader))
    except TypeError:
        num_batches = None

    update_unit = str(getattr(args, "pq_update_unit", "batch")).lower().strip()
    if update_unit == "epoch":
        if num_batches is None or num_batches <= 0:
            raise ValueError("--pq_update_unit=epoch requires a train loader with a known positive length.")
        return int(num_batches), num_batches

    fraction = float(getattr(args, "pq_update_batch_fraction", 0.0))
    if fraction > 0.0:
        if num_batches is None or num_batches <= 0:
            raise ValueError("--pq_update_batch_fraction requires a train loader with a known length.")
        return max(1, int(math.ceil(float(num_batches) * fraction))), num_batches

    return max(1, int(getattr(args, "pq_update_every_batches", 1))), num_batches


def _resolve_pq_batch_update_interval_from_count(args, num_batches: int) -> Tuple[int, int]:
    update_unit = str(getattr(args, "pq_update_unit", "batch")).lower().strip()
    if update_unit == "epoch":
        if int(num_batches) <= 0:
            raise ValueError("--pq_update_unit=epoch requires a positive batch count.")
        return int(num_batches), int(num_batches)

    fraction = float(getattr(args, "pq_update_batch_fraction", 0.0))
    if fraction > 0.0:
        return max(1, int(math.ceil(float(num_batches) * fraction))), int(num_batches)
    return max(1, int(getattr(args, "pq_update_every_batches", 1))), int(num_batches)


def _resolve_tu_iters_per_epoch(args, train_size: int) -> int:
    requested = getattr(args, "iters_per_epoch", 50)
    if isinstance(requested, str) and requested.lower().strip() == "auto":
        return max(1, int(math.ceil(float(train_size) / float(max(int(args.batch_size), 1)))))
    return int(requested)


def _make_loaders(
    dataset,
    split_idx: Dict[str, torch.Tensor],
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> Dict[str, DataLoader]:
    def subset(name: str) -> Subset:
        return Subset(dataset, split_idx[name].tolist())

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": bool(pin_memory),
        "worker_init_fn": _seed_worker,
    }
    if int(num_workers) > 0:
        loader_kwargs["persistent_workers"] = True

    def make_generator(offset: int) -> torch.Generator:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + int(offset))
        return generator

    return {
        "train": DataLoader(subset("train"), shuffle=True, generator=make_generator(0), **loader_kwargs),
        "valid": DataLoader(subset("valid"), shuffle=False, generator=make_generator(1), **loader_kwargs),
        "test": DataLoader(subset("test"), shuffle=False, generator=make_generator(2), **loader_kwargs),
    }


def _make_eval_loaders(
    dataset,
    split_idx: Dict[str, torch.Tensor],
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> Dict[str, DataLoader]:
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": bool(pin_memory),
        "worker_init_fn": _seed_worker,
    }
    if int(num_workers) > 0:
        loader_kwargs["persistent_workers"] = True

    def make_generator(offset: int) -> torch.Generator:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + int(offset))
        return generator

    return {
        name: DataLoader(
            Subset(dataset, indices.tolist()),
            shuffle=False,
            generator=make_generator(offset),
            **loader_kwargs,
        )
        for offset, (name, indices) in enumerate(split_idx.items())
    }


def _sample_tu_batch(dataset, train_idx: torch.Tensor, batch_size: int) -> Batch:
    local_perm = np.random.permutation(len(train_idx))[: int(batch_size)]
    global_idx = train_idx[torch.as_tensor(local_perm, dtype=torch.long)]
    data_list = [dataset[int(idx)] for idx in global_idx.tolist()]
    return Batch.from_data_list(data_list)


def _save_checkpoint(args, model, optimizer, run: int, epoch: int, best_valid: float) -> None:
    out_dir = _gc_path("models_gc", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"GraphMPNN_{args.gnn}_run{run:02d}.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "best_valid": float(best_valid),
            "args": vars(args),
        },
        path,
    )


def _maybe_cast_batch_for_linear(batch, *, use_ogb_encoders: bool, use_edge_attr: bool) -> None:
    if not use_ogb_encoders:
        if batch.x is not None and not torch.is_floating_point(batch.x):
            batch.x = batch.x.float()
        if use_edge_attr and hasattr(batch, "edge_attr") and batch.edge_attr is not None and not torch.is_floating_point(batch.edge_attr):
            batch.edge_attr = batch.edge_attr.float()


def _cat_edge_indices(chunks: List[torch.Tensor]) -> torch.Tensor:
    if not chunks:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.cat(chunks, dim=1)


def _augment_batch_graphwise(
    batch,
    *,
    p: float,
    q: float,
    mode: str,
    seed: int,
    device: torch.device,
    use_edge_attr: bool,
    ep_correction: bool,
    mask_sharing: str,
    local_layers: int,
    safety_max_additions: int = 2_000_000,
    augmentor_cache: Optional[Dict[Tuple[int, bool], BernoulliEdgeAugmentor]] = None,
) -> Tuple[object, dict, Optional[EdgeIndexObj], Optional[EdgeIndexObj]]:
    mode = str(mode).lower().strip()
    mask_sharing = str(mask_sharing).lower().strip()
    if mode == "none" or (p <= 0.0 and q <= 0.0):
        stats = {"graphs": int(getattr(batch, "num_graphs", 0)), "p": float(p), "q": float(q), "mode": mode}
        return batch.to(device), stats, None, None

    data_list = batch.to_data_list()
    dropped_total = 0
    added_total = 0
    graphs_in_batch = int(getattr(batch, "num_graphs", len(data_list)))

    if bool(ep_correction) and mask_sharing == "layerwise":
        keep_layers = [[] for _ in range(int(local_layers))]
        add_layers = [[] for _ in range(int(local_layers))]
        dropped_layers = [0 for _ in range(int(local_layers))]
        added_layers = [0 for _ in range(int(local_layers))]
    else:
        keep_list = []
        add_list = []

    node_offset = 0
    for graph_idx, data in enumerate(data_list):
        edge_attr = data.edge_attr if (use_edge_attr and hasattr(data, "edge_attr") and data.edge_attr is not None) else None
        augmentor = None
        cache_key = None
        graph_id = getattr(data, "graph_id", None)
        if augmentor_cache is not None and graph_id is not None:
            graph_id_t = graph_id.view(-1) if torch.is_tensor(graph_id) else torch.as_tensor(graph_id).view(-1)
            if graph_id_t.numel() == 1:
                cache_key = (int(graph_id_t.item()), bool(use_edge_attr))
                augmentor = augmentor_cache.get(cache_key)
        if augmentor is None:
            augmentor = BernoulliEdgeAugmentor.from_graph(data.edge_index, int(data.num_nodes), edge_attr=edge_attr)
            if augmentor_cache is not None and cache_key is not None:
                augmentor_cache[cache_key] = augmentor

        if bool(ep_correction):
            data.edge_index = data.edge_index.to(torch.long)
            if edge_attr is not None:
                data.edge_attr = edge_attr

            if mask_sharing == "shared":
                _, edge_index_keep, edge_index_add, _, _, _, stats = augmentor.augment_edge_indices_with_attr(
                    p=float(p),
                    q=float(q),
                    mode=mode,
                    seed=int(seed) + 97 * int(graph_idx),
                    device=torch.device("cpu"),
                    safety_max_additions=int(safety_max_additions),
                )
                if edge_index_keep.numel() > 0:
                    keep_list.append(edge_index_keep.to(torch.long) + int(node_offset))
                if edge_index_add.numel() > 0:
                    add_list.append(edge_index_add.to(torch.long) + int(node_offset))
                dropped_total += int(stats.get("dropped_undir", 0))
                added_total += int(stats.get("added_undir", 0))
            else:
                for layer_idx in range(int(local_layers)):
                    _, edge_index_keep, edge_index_add, _, _, _, stats = augmentor.augment_edge_indices_with_attr(
                        p=float(p),
                        q=float(q),
                        mode=mode,
                        seed=int(seed) + 97 * int(graph_idx) + 1_000_000 * int(layer_idx),
                        device=torch.device("cpu"),
                        safety_max_additions=int(safety_max_additions),
                    )
                    if edge_index_keep.numel() > 0:
                        keep_layers[layer_idx].append(edge_index_keep.to(torch.long) + int(node_offset))
                    if edge_index_add.numel() > 0:
                        add_layers[layer_idx].append(edge_index_add.to(torch.long) + int(node_offset))
                    dropped_layers[layer_idx] += int(stats.get("dropped_undir", 0))
                    added_layers[layer_idx] += int(stats.get("added_undir", 0))
        else:
            edge_index_aug, _, _, edge_attr_aug, _, _, stats = augmentor.augment_edge_indices_with_attr(
                p=float(p),
                q=float(q),
                mode=mode,
                seed=int(seed) + 97 * int(graph_idx),
                device=torch.device("cpu"),
                safety_max_additions=int(safety_max_additions),
            )
            data.edge_index = edge_index_aug.to(torch.long)
            if use_edge_attr and edge_attr_aug is not None:
                data.edge_attr = edge_attr_aug
            dropped_total += int(stats.get("dropped_undir", 0))
            added_total += int(stats.get("added_undir", 0))

        node_offset += int(data.num_nodes)

    batch_out = type(batch).from_data_list(data_list).to(device)
    if bool(ep_correction):
        if mask_sharing == "shared":
            keep_obj = _cat_edge_indices(keep_list).to(device)
            add_obj = _cat_edge_indices(add_list).to(device)
            stats = {
                "graphs": graphs_in_batch,
                "dropped_undir": int(dropped_total),
                "added_undir": int(added_total),
                "mask_sharing": "shared",
            }
            return batch_out, stats, keep_obj, add_obj
        keep_obj = [_cat_edge_indices(keep_layers[layer_idx]).to(device) for layer_idx in range(int(local_layers))]
        add_obj = [_cat_edge_indices(add_layers[layer_idx]).to(device) for layer_idx in range(int(local_layers))]
        stats = {
            "graphs": graphs_in_batch,
            "dropped_undir_avg": float(sum(dropped_layers)) / float(max(int(local_layers), 1)),
            "added_undir_avg": float(sum(added_layers)) / float(max(int(local_layers), 1)),
            "mask_sharing": "layerwise",
        }
        return batch_out, stats, keep_obj, add_obj

    stats = {"graphs": graphs_in_batch, "dropped_undir": int(dropped_total), "added_undir": int(added_total)}
    return batch_out, stats, None, None


def _mean(values: List[float]) -> float:
    return float(sum(values) / max(len(values), 1)) if values else 0.0


def _compute_graph_density(dataset) -> Tuple[float, float]:
    total_edges = 0.0
    total_pairs = 0.0
    for data in dataset:
        num_nodes = int(getattr(data, "num_nodes", 0) or 0)
        if num_nodes <= 1:
            continue
        total_pairs += float(num_nodes * (num_nodes - 1) // 2)
        edge_index = getattr(data, "edge_index", None)
        if edge_index is None or edge_index.numel() == 0:
            continue
        try:
            augmentor = BernoulliEdgeAugmentor.from_graph(edge_index, num_nodes, edge_attr=None)
            total_edges += float(augmentor.edge_u.numel())
        except Exception:
            row, col = edge_index[0], edge_index[1]
            total_edges += float((row != col).sum().item()) / 2.0

    density = total_edges / max(total_pairs, 1.0)
    rho_scale = 0.0 if density <= 0.0 else 1.0 / density
    return float(density), float(rho_scale)


def _save_obj_plot(obj_histories: List[List[float]], *, tag: str, out_dir: Optional[str] = None, show_runs: bool = True) -> None:
    if out_dir is None:
        out_dir = _gc_path("visualization")
    plot_schedule = getattr(_VIZ_PQ_MODULE, "_plot_schedule", None)
    if plot_schedule is None:
        raise RuntimeError("viz_pq._plot_schedule is unavailable; cannot save objective plot.")
    plot_schedule(obj_histories, tag=tag, name="obj", out_dir=out_dir, show_runs=show_runs)


def _safe_tag(value: object) -> str:
    text = str(value).lower().strip()
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    return safe.strip("_") or "unknown"


def _write_single_dataset_summary(record: Dict[str, object], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    file_exists = os.path.exists(out_path) and os.path.getsize(out_path) > 0
    with open(out_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def _is_tu_dataset_name(name: str) -> bool:
    return str(name).lower().strip() in {"mutag", "proteins", "imdb-b", "imdb-m"}


def print_training_banner(args, model) -> None:
    aug_tech = str(getattr(args, "aug_tech", "rade")).lower().strip()
    aug_mode = str(getattr(args, "aug_mode", "none")).lower().strip()
    mask_sharing = str(getattr(args, "mask_sharing", "layerwise")).lower().strip()
    if aug_tech == "dropout":
        method_rate = f"dropout={float(getattr(args, 'dropout', 0.0)):.4f}"
    elif aug_tech == "dropedge":
        method_rate = f"dropedge={float(getattr(args, 'p', 0.0)):.4f}"
    elif aug_tech == "dropnode":
        method_rate = f"dropnode={float(getattr(args, 'dropnode_rate', 0.0)):.4f}"
    elif aug_tech == "dropmessage":
        method_rate = f"dropmessage={float(getattr(args, 'dropmessage_rate', 0.0)):.4f}"
    elif aug_tech == "rade":
        method_rate = (
            f"rade(p={float(getattr(args, 'p', 0.0)):.4f}, "
            f"q={float(getattr(args, 'q', 0.0)):.6f})"
        )
    else:
        method_rate = "none"
    print(
        f"[GC] tech={aug_tech} model={model.__class__.__name__} "
        f"backbone_raw={getattr(args, 'gnn_raw', args.gnn)} -> backbone={args.gnn} "
        f"| lr={float(getattr(args, 'lr', 0.0)):g} "
        f"| method_rate={method_rate} "
        f"| aug_mode={aug_mode}(p={float(getattr(args, 'p', 0.0)):.4f}, q={float(getattr(args, 'q', 0.0)):.6f}) "
        f"| ep_correction={'ON' if bool(getattr(args, 'ep_correction', False)) else 'OFF'}"
        f"({getattr(args, 'rade_variant', 'off')}, {getattr(args, 'ep_expectation_mode', 'analytic')}) "
        f"| gradnorm={'ON' if bool(getattr(args, 'pq_gradnorm', False)) else 'OFF'}"
        f"(adam/direct, rho_opt={bool(getattr(args, 'pq_optimize_rho', False))}, "
        f"update_unit={getattr(args, 'pq_update_unit', 'batch')}, "
        f"p_max={float(getattr(args, 'pq_p_max', 1.0 - 1e-6)):.4f}, "
        f"del_pen={float(getattr(args, 'pq_deletion_penalty_lambda', 0.0)):g}, "
        f"dens_pen={float(getattr(args, 'pq_densification_penalty_lambda', 0.0)):g}/"
        f"{getattr(args, 'pq_densification_penalty_type', 'linear')}, "
        f"gat_den={float(getattr(args, 'pq_gat_denom_penalty_lambda', 0.0)):g}@"
        f"{float(getattr(args, 'pq_gat_denom_cv2_threshold', 1.0)):g}, "
        f"gat_corr={float(getattr(args, 'pq_gat_corr_penalty_lambda', 0.0)):g}@"
        f"{float(getattr(args, 'pq_gat_corr_threshold', 2.0)):g}, "
        f"corr_clip={float(getattr(args, 'gat_ep_corr_clip', 2.0)):g}) "
        f"| mask={mask_sharing}"
    )
    print(model, flush=True)


def count_trainable_params(model: torch.nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def _run_tu_reference_protocol(args, *, device: torch.device, use_wandb: bool) -> None:
    attach_global_node_ids = (
        str(getattr(args, "aug_tech", "rade")).lower().strip() == "rade"
        and bool(getattr(args, "ep_correction", False))
        and str(getattr(args, "ep_expectation_mode", "analytic")).lower().strip() == "empirical_ema"
    )
    raw_root = str(getattr(args, "tu_raw_root", "./tu_raw_data"))
    if not os.path.isabs(raw_root):
        raw_root = os.path.abspath(os.path.join(GC_DIR, raw_root))

    dataset, meta = load_powerful_gnns_tu_dataset(
        raw_root=raw_root,
        name=args.dataset,
        attach_graph_ids=(str(getattr(args, "aug_tech", "rade")).lower().strip() == "rade"),
        attach_global_node_ids=attach_global_node_ids,
    )
    folds = make_powerful_gnns_tu_folds(dataset, seed=int(args.seed))
    args.global_num_nodes_total = int(meta.get("global_num_nodes_total", 0))

    graph_density, rho_scale = _compute_graph_density(dataset)
    args.graph_density = float(graph_density)
    args.pq_expected_add_ratio_scale = float(rho_scale)

    loss_type = meta["loss"] if args.loss == "auto" else args.loss
    metric = meta["metric"] if args.metric == "auto" else args.metric
    criterion = nn.CrossEntropyLoss() if loss_type == "ce" else nn.BCEWithLogitsLoss(reduction="none")
    aug_tech = str(getattr(args, "aug_tech", "rade")).lower().strip()
    pq_enabled = bool(getattr(args, "pq_gradnorm", False))
    mask_sharing = str(getattr(args, "mask_sharing", "layerwise")).lower().strip()

    if use_wandb:
        wandb.config.update(vars(args), allow_val_change=True)

    print(
        "[TU] reference protocol enabled: "
        f"source={meta['raw_source']} | folds={len(folds)} | "
        f"degree_as_tag={meta['degree_as_tag']} | "
        f"epochs={int(args.epochs)} | iters_per_epoch={getattr(args, 'iters_per_epoch', 50)} | "
        f"batch_size={int(args.batch_size)} | lr={float(args.lr):g}"
    )
    if pq_enabled:
        print(f"[PQ] graph_density={graph_density:.6e} rho_scale={rho_scale:.6e}")

    fold_valid_curves: List[List[float]] = []
    fold_train_curves: List[List[float]] = []
    fold_loss_curves: List[List[float]] = []
    p_histories: List[List[float]] = []
    q_histories: List[List[float]] = []
    obj_histories: List[List[float]] = []
    rho_histories: List[List[float]] = []
    epoch_time_histories: List[List[float]] = []

    for fold_idx, split_idx in enumerate(folds):
        fix_seed(int(getattr(args, "tu_train_seed", 0)), device=device)
        eval_loaders = _make_eval_loaders(
            dataset,
            split_idx,
            batch_size=64,
            num_workers=int(args.num_workers),
            pin_memory=(device.type == "cuda"),
            seed=int(getattr(args, "tu_train_seed", 0)),
        )

        model = parse_method_gc(args, in_channels=int(meta["in_dim"]), out_channels=int(meta["out_dim"]), device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        resolved_iters_per_epoch = _resolve_tu_iters_per_epoch(args, train_size=len(split_idx["train"]))
        print(
            f"[TU] fold={fold_idx + 1:02d}/10 train={len(split_idx['train'])} "
            f"valid={len(split_idx['valid'])} iters_per_epoch={resolved_iters_per_epoch}"
        )
        print(f"[GC] Trainable params: {count_trainable_params(model):,}")
        print_training_banner(args, model)

        use_ogb_encoders = bool(getattr(args, "use_ogb_encoders", False))
        use_edge_attr = bool(getattr(args, "use_edge_attr", False))
        local_layers = int(getattr(getattr(model, "cfg", None), "local_layers", getattr(args, "local_layers", 1)))
        rade_variant = str(getattr(args, "rade_variant", "rade-of")).lower().strip()
        p_eff, q_eff = _enforce_aug_mode(float(getattr(args, "p", 0.0)), float(getattr(args, "q", 0.0)), args.aug_mode)
        if pq_enabled:
            p_eff = min(float(p_eff), float(getattr(args, "pq_p_max", 1.0 - 1e-6)))
        if attach_global_node_ids:
            if not hasattr(model, "reset_ep_stats"):
                raise RuntimeError("ep_expectation_mode='empirical_ema' requires model.reset_ep_stats(...).")
            model.reset_ep_stats(p_eff, q_eff)

        pq_tuner = None
        if pq_enabled:
            pq_tuner = PQGradNormTunerGC(
                gnn=str(args.gnn),
                rade_variant=rade_variant,
                pooling=str(args.pooling),
                cfg=PQTunerGCConfig(
                    expected_add_ratio_scale=float(getattr(args, "pq_expected_add_ratio_scale", 0.0)),
                    deletion_penalty_lambda=float(getattr(args, "pq_deletion_penalty_lambda", 0.0)),
                    densification_penalty_lambda=float(getattr(args, "pq_densification_penalty_lambda", 1.0)),
                    densification_penalty_type=str(getattr(args, "pq_densification_penalty_type", "linear")),
                    densification_penalty_rho0=float(getattr(args, "pq_densification_penalty_rho0", 1.0)),
                    p_max=float(getattr(args, "pq_p_max", 1.0 - 1e-6)),
                    optimize_rho=bool(getattr(args, "pq_optimize_rho", False)),
                    eps=1e-12,
                    seed=int(getattr(args, "tu_train_seed", 0)),
                    data_anchor=str(getattr(args, "pq_data_anchor", "clean")),
                    adam_lr=float(getattr(args, "pq_adam_lr", 0.1)),
                    adam_beta1=float(getattr(args, "pq_adam_beta1", 0.9)),
                    adam_beta2=float(getattr(args, "pq_adam_beta2", 0.999)),
                    adam_eps=float(getattr(args, "pq_adam_eps", 1e-8)),
                    ep_expectation_mode=str(getattr(args, "ep_expectation_mode", "analytic")),
                    gat_denom_penalty_lambda=float(getattr(args, "pq_gat_denom_penalty_lambda", 0.0)),
                    gat_denom_cv2_threshold=float(getattr(args, "pq_gat_denom_cv2_threshold", 1.0)),
                    gat_corr_penalty_lambda=float(getattr(args, "pq_gat_corr_penalty_lambda", 0.0)),
                    gat_corr_threshold=float(getattr(args, "pq_gat_corr_threshold", 2.0)),
                ),
            )

        pq_batch_update_interval, train_batches_per_epoch = _resolve_pq_batch_update_interval_from_count(
            args,
            resolved_iters_per_epoch,
        )
        augmentor_cache: Dict[Tuple[int, bool], BernoulliEdgeAugmentor] = {}
        valid_curve: List[float] = []
        train_curve: List[float] = []
        loss_curve: List[float] = []
        p_trace: List[float] = []
        q_trace: List[float] = []
        obj_trace: List[float] = []
        rho_trace: List[float] = []
        epoch_time_trace: List[float] = []
        best_valid = -1e18

        for epoch in range(1, int(args.epochs) + 1):
            _sync_for_timing(device)
            epoch_time_start = time.perf_counter()
            model.train()
            p_trace.append(float(p_eff))
            q_trace.append(float(q_eff))
            rho_trace.append(float(q_eff) * float(rho_scale))
            obj_trace.append(float("nan"))

            epoch_loss_sum_tensor: Optional[torch.Tensor] = None
            epoch_graphs = 0
            dropped_epoch_acc = 0.0
            added_epoch_acc = 0.0
            aug_batches = 0
            stats_are_layer_avg = bool(getattr(args, "ep_correction", False)) and mask_sharing == "layerwise"

            warmup = int(getattr(args, "pq_warmup_epochs", 0))
            every = int(getattr(args, "pq_update_every", 1))
            do_pq_update = pq_tuner is not None and epoch >= warmup and every > 0 and (epoch % every) == 0
            adam_batch_infos: List[Dict[str, float]] = []

            for step in range(resolved_iters_per_epoch):
                batch = _sample_tu_batch(dataset, split_idx["train"], int(args.batch_size))
                optimizer.zero_grad(set_to_none=True)

                edge_index_keep_obj: Optional[EdgeIndexObj] = None
                edge_index_add_obj: Optional[EdgeIndexObj] = None
                do_aug = (
                    aug_tech in {"rade", "dropedge"}
                    and str(getattr(args, "aug_mode", "none")).lower().strip() != "none"
                    and (p_eff > 0.0 or q_eff > 0.0)
                )
                if do_aug:
                    aug_p = float(p_eff)
                    aug_q = 0.0 if aug_tech == "dropedge" else float(q_eff)
                    aug_mode = "drop" if aug_tech == "dropedge" else str(getattr(args, "aug_mode", "both"))
                    batch, stats, edge_index_keep_obj, edge_index_add_obj = _augment_batch_graphwise(
                        batch,
                        p=aug_p,
                        q=aug_q,
                        mode=aug_mode,
                        seed=int(getattr(args, "tu_train_seed", 0)) + 10_000 * int(fold_idx) + 1_000 * int(epoch) + int(step),
                        device=device,
                        use_edge_attr=use_edge_attr,
                        ep_correction=bool(getattr(args, "ep_correction", False)) and aug_tech == "rade",
                        mask_sharing=mask_sharing,
                        local_layers=local_layers,
                        safety_max_additions=int(getattr(args, "safety_max_additions", 2_000_000)),
                        augmentor_cache=augmentor_cache,
                    )
                    if "dropped_undir" in stats:
                        dropped_epoch_acc += float(stats["dropped_undir"])
                        added_epoch_acc += float(stats["added_undir"])
                        stats_are_layer_avg = False
                    elif "dropped_undir_avg" in stats:
                        dropped_epoch_acc += float(stats["dropped_undir_avg"])
                        added_epoch_acc += float(stats["added_undir_avg"])
                        stats_are_layer_avg = True
                    aug_batches += 1
                else:
                    batch = batch.to(device)

                _maybe_cast_batch_for_linear(batch, use_ogb_encoders=use_ogb_encoders, use_edge_attr=use_edge_attr)
                edge_attr = batch.edge_attr if (use_edge_attr and hasattr(batch, "edge_attr")) else None
                node_ids = getattr(batch, "n_id", None)

                if aug_tech == "dropmessage":
                    out = model(
                        batch.x,
                        batch.edge_index,
                        batch.batch,
                        edge_attr=edge_attr,
                        dropmessage_rate=float(getattr(args, "dropmessage_rate", 0.0)),
                        node_ids=node_ids,
                        p=0.0,
                        q=0.0,
                    )
                elif aug_tech == "dropnode":
                    out = model(
                        batch.x,
                        batch.edge_index,
                        batch.batch,
                        edge_attr=edge_attr,
                        dropnode_rate=float(getattr(args, "dropnode_rate", 0.0)),
                        node_ids=node_ids,
                        p=0.0,
                        q=0.0,
                    )
                elif aug_tech == "dropedge":
                    out = model(
                        batch.x,
                        batch.edge_index,
                        batch.batch,
                        edge_attr=edge_attr,
                        node_ids=node_ids,
                        p=0.0,
                        q=0.0,
                    )
                else:
                    out = model(
                        batch.x,
                        batch.edge_index,
                        batch.batch,
                        edge_attr=edge_attr,
                        edge_index_keep=edge_index_keep_obj if bool(getattr(args, "ep_correction", False)) and do_aug else None,
                        edge_index_add=edge_index_add_obj if bool(getattr(args, "ep_correction", False)) and do_aug else None,
                        p=float(p_eff),
                        q=float(q_eff),
                        node_ids=node_ids,
                    )

                y = batch.y
                if loss_type == "ce":
                    if y.dim() == 2 and y.size(1) == 1:
                        y = y.squeeze(1)
                    y = y.view(-1).to(torch.long)
                    loss = criterion(out, y)
                else:
                    if y.dim() == 1:
                        y = y.view(-1, 1)
                    if out.dim() == 1:
                        out = out.view(-1, 1)
                    loss = masked_bce_with_logits(out, y)

                loss.backward()
                optimizer.step()

                if (
                    attach_global_node_ids
                    and do_aug
                    and bool(getattr(args, "ep_correction", False))
                    and str(getattr(args, "ep_expectation_mode", "analytic")).lower().strip() == "empirical_ema"
                ):
                    if not hasattr(model, "update_ep_stats"):
                        raise RuntimeError("ep_expectation_mode='empirical_ema' requires model.update_ep_stats(...).")
                    model.update_ep_stats(
                        edge_index_keep=edge_index_keep_obj,
                        edge_index_add=edge_index_add_obj,
                        p=float(p_eff),
                        q=float(q_eff),
                    )

                batch_graphs = int(batch.num_graphs) if hasattr(batch, "num_graphs") else int(y.size(0))
                loss_term = loss.detach() * float(batch_graphs)
                epoch_loss_sum_tensor = loss_term if epoch_loss_sum_tensor is None else epoch_loss_sum_tensor + loss_term
                epoch_graphs += batch_graphs

                if do_pq_update and (((int(step) + 1) % int(pq_batch_update_interval)) == 0):
                    p_next, q_next, info = pq_tuner.suggest_pq_batch(
                        model=model,
                        batch=batch,
                        loss_type=str(loss_type),
                        aug_mode=str(getattr(args, "aug_mode", "both")).lower().strip(),
                        current_p=float(p_eff),
                        current_q=float(q_eff),
                        seed=int(getattr(args, "tu_train_seed", 0)) + 100_000 * int(fold_idx) + 10_000 * int(epoch) + int(step),
                        mask_sharing=mask_sharing,
                        num_layers=local_layers,
                    )
                    p_eff, q_eff = _enforce_aug_mode(p_next, q_next, args.aug_mode)
                    adam_batch_infos.append(info)

            epoch_loss_sum = float(epoch_loss_sum_tensor.item()) if epoch_loss_sum_tensor is not None else 0.0
            train_loss = epoch_loss_sum / max(epoch_graphs, 1)
            current_lr = float(optimizer.param_groups[0]["lr"])
            p_trace[-1] = float(p_eff)
            q_trace[-1] = float(q_eff)
            rho_trace[-1] = float(q_eff) * float(rho_scale)
            if adam_batch_infos:
                obj_trace[-1] = _mean([float(item["obj_new"]) for item in adam_batch_infos])

            p_eval, q_eval = 0.0, 0.0
            if aug_tech == "rade" and bool(getattr(args, "ep_correction", False)) and rade_variant == "rade-ofs":
                p_eval, q_eval = float(p_eff), float(q_eff)

            eval_device = torch.device("cpu") if bool(getattr(args, "full_eval_cpu", False)) and device.type == "cuda" else device
            if eval_device.type == "cpu" and device.type == "cuda":
                model.to(eval_device)
            train_score, _ = evaluate_loader(
                model,
                eval_loaders["train"],
                args.dataset,
                criterion,
                metric=metric,
                use_edge_attr=use_edge_attr,
                device=eval_device,
                p=float(p_eval),
                q=float(q_eval),
            )
            valid_score, valid_loss = evaluate_loader(
                model,
                eval_loaders["valid"],
                args.dataset,
                criterion,
                metric=metric,
                use_edge_attr=use_edge_attr,
                device=eval_device,
                p=float(p_eval),
                q=float(q_eval),
            )
            if eval_device.type == "cpu" and device.type == "cuda":
                model.to(device)
                torch.cuda.empty_cache()

            _sync_for_timing(device)
            epoch_time_sec = time.perf_counter() - epoch_time_start
            epoch_time_trace.append(float(epoch_time_sec))
            train_curve.append(float(train_score))
            valid_curve.append(float(valid_score))
            loss_curve.append(float(train_loss))
            best_valid = max(float(best_valid), float(valid_score))

            if epoch % int(getattr(args, "display_step", 1)) == 0:
                msg = (
                    f"[TU] fold={fold_idx + 1:02d} epoch={epoch:03d} "
                    f"lr={current_lr:.6g} loss={train_loss:.4f} train_{metric}={train_score:.4f} "
                    f"valid_{metric}={valid_score:.4f} best_valid={best_valid:.4f}"
                )
                if aug_tech == "rade" and aug_batches > 0:
                    if stats_are_layer_avg:
                        msg += (
                            f" | drop/add(avg per layer, undir): -{dropped_epoch_acc:.1f} +{added_epoch_acc:.1f} "
                            f"(p={p_eff:.4f}, q={q_eff:.6f}, rho={float(q_eff) * float(rho_scale):.6f})"
                        )
                    else:
                        msg += (
                            f" | drop/add(undir): -{int(dropped_epoch_acc)} +{int(added_epoch_acc)} "
                            f"(p={p_eff:.4f}, q={q_eff:.6f}, rho={float(q_eff) * float(rho_scale):.6f})"
                        )
                elif aug_tech == "dropedge":
                    msg += f" | dropedge={float(p_eff):.4f}"
                    if aug_batches > 0:
                        msg += f" dropped_undir={int(dropped_epoch_acc)}"
                elif aug_tech == "dropmessage":
                    msg += f" | dropmessage={float(getattr(args, 'dropmessage_rate', 0.0)):.4f}"
                elif aug_tech == "dropnode":
                    msg += f" | dropnode={float(getattr(args, 'dropnode_rate', 0.0)):.4f}"
                elif aug_tech == "dropout":
                    msg += f" | dropout={float(getattr(args, 'dropout', 0.0)):.4f}"
                if adam_batch_infos:
                    msg += (
                        f" | pq: Gd={_mean([float(item['G_data']) for item in adam_batch_infos]):.3e} "
                        f"Gr={_mean([float(item['G_reg_new']) for item in adam_batch_infos]):.3e} "
                        f"obj={_mean([float(item['obj_new']) for item in adam_batch_infos]):.2e} "
                        f"rho={_mean([float(item.get('rho_new', float(item['q_new']) * float(rho_scale))) for item in adam_batch_infos]):.6f} "
                        f"gat_den={_mean([float(item.get('gat_denom_penalty_new', 0.0)) for item in adam_batch_infos]):.2e} "
                        f"gat_corr={_mean([float(item.get('gat_corr_penalty_new', 0.0)) for item in adam_batch_infos]):.2e} "
                        f"updates={len(adam_batch_infos)} unit={getattr(args, 'pq_update_unit', 'batch')} "
                        f"interval={int(pq_batch_update_interval)}"
                    )
                msg += f" | epoch_sec={epoch_time_sec:.4f}"
                print(msg)

            if use_wandb and epoch % int(getattr(args, "wandb_log_every", 1)) == 0:
                log_obj = {
                    "epoch": epoch,
                    "fold": fold_idx + 1,
                    "train/loss": train_loss,
                    "valid/loss": valid_loss,
                    f"train/{metric}": train_score,
                    f"valid/{metric}": valid_score,
                    "best/valid_so_far": best_valid,
                    "valid/primary": valid_score,
                    "best/valid_primary": best_valid,
                    "aug/tech": aug_tech,
                    "aug/p": float(p_eff),
                    "aug/q": float(q_eff),
                    "aug/rho": float(q_eff) * float(rho_scale),
                    "runtime/epoch_sec": float(epoch_time_sec),
                    "optim/lr": current_lr,
                    "tu/protocol": "powerful_gnns",
                }
                if adam_batch_infos:
                    log_obj.update(
                        {
                            "pq/G_data_mean": _mean([float(item["G_data"]) for item in adam_batch_infos]),
                            "pq/G_reg_best_mean": _mean([float(item["G_reg_best"]) for item in adam_batch_infos]),
                            "pq/G_reg_new_mean": _mean([float(item["G_reg_new"]) for item in adam_batch_infos]),
                            "pq/obj_mean": _mean([float(item["obj"]) for item in adam_batch_infos]),
                            "pq/obj_new_mean": _mean([float(item["obj_new"]) for item in adam_batch_infos]),
                            "pq/p_new": float(p_eff),
                            "pq/q_new": float(q_eff),
                            "pq/rho_new": float(q_eff) * float(rho_scale),
                            "pq/p_max": float(getattr(args, "pq_p_max", 1.0 - 1e-6)),
                            "pq/used_batches": float(len(adam_batch_infos)),
                            "pq/skipped_batches": float(max(0, int(train_batches_per_epoch) - len(adam_batch_infos))),
                            "pq/update_unit": str(getattr(args, "pq_update_unit", "batch")),
                            "pq/gat_denom_penalty_new_mean": _mean(
                                [float(item.get("gat_denom_penalty_new", 0.0)) for item in adam_batch_infos]
                            ),
                            "pq/gat_denom_cv2_mean_new": _mean(
                                [float(item.get("gat_denom_cv2_mean_new", 0.0)) for item in adam_batch_infos]
                            ),
                            "pq/gat_denom_cv2_max_new": _mean(
                                [float(item.get("gat_denom_cv2_max_new", 0.0)) for item in adam_batch_infos]
                            ),
                            "pq/gat_corr_penalty_new_mean": _mean(
                                [float(item.get("gat_corr_penalty_new", 0.0)) for item in adam_batch_infos]
                            ),
                            "pq/gat_corr_gain_mean_new": _mean(
                                [float(item.get("gat_corr_gain_mean_new", 0.0)) for item in adam_batch_infos]
                            ),
                            "pq/gat_corr_gain_max_new": _mean(
                                [float(item.get("gat_corr_gain_max_new", 0.0)) for item in adam_batch_infos]
                            ),
                            "pq/gat_denom_penalty_lambda": float(
                                getattr(args, "pq_gat_denom_penalty_lambda", 0.0)
                            ),
                            "pq/gat_denom_cv2_threshold": float(
                                getattr(args, "pq_gat_denom_cv2_threshold", 1.0)
                            ),
                            "pq/gat_corr_penalty_lambda": float(
                                getattr(args, "pq_gat_corr_penalty_lambda", 0.0)
                            ),
                            "pq/gat_corr_threshold": float(getattr(args, "pq_gat_corr_threshold", 2.0)),
                        }
                    )
                wandb.log(log_obj)

            scheduler.step()

        fold_valid_curves.append(valid_curve)
        fold_train_curves.append(train_curve)
        fold_loss_curves.append(loss_curve)
        p_histories.append(p_trace)
        q_histories.append(q_trace)
        obj_histories.append(obj_trace)
        rho_histories.append(rho_trace)
        epoch_time_histories.append(epoch_time_trace)

    valid_matrix = np.asarray(fold_valid_curves, dtype=float)
    train_matrix = np.asarray(fold_train_curves, dtype=float)
    mean_valid_curve = valid_matrix.mean(axis=0)
    selected_epoch_idx = int(np.argmax(mean_valid_curve))
    selected_epoch = selected_epoch_idx + 1
    selected_valid_scores = valid_matrix[:, selected_epoch_idx]
    selected_train_scores = train_matrix[:, selected_epoch_idx]
    mean_selected_valid = float(np.mean(selected_valid_scores))
    std_selected_valid = float(np.std(selected_valid_scores))
    runtime_values = [value for history in epoch_time_histories for value in history]

    print(
        "[TU] selected shared epoch from mean validation curve: "
        f"{selected_epoch:03d} (mean_valid_{metric}={mean_valid_curve[selected_epoch_idx]:.4f})"
    )
    print(
        f"[TU] validation_{metric}@shared_epoch = "
        f"{100.0 * mean_selected_valid:.2f} +/- {100.0 * std_selected_valid:.2f} (%) over 10 folds"
    )
    if runtime_values:
        print(f"[TU] epoch_time_sec = {float(np.mean(runtime_values)):.4f} +/- {float(np.std(runtime_values)):.4f}")

    dataset_tag = _safe_tag(args.dataset)
    gnn_tag = _safe_tag(getattr(args, "gnn_raw", getattr(args, "gnn", "gcn")))
    rade_variant_tag = _safe_tag(getattr(args, "rade_variant", "rade-of")).replace("-", "_")
    single_dataset_path = _gc_path(
        "single_dataset",
        f"results_gc_{dataset_tag}_{gnn_tag}_{rade_variant_tag}_tu_powerful_gnns.csv",
    )
    single_dataset_record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": str(args.dataset),
        "gnn_raw": str(getattr(args, "gnn_raw", getattr(args, "gnn", "gcn"))),
        "gnn": str(args.gnn),
        "model": "GraphMPNN",
        "protocol": "tu_powerful_gnns",
        "metric": str(metric),
        "loss": str(loss_type),
        "shared_epoch": int(selected_epoch),
        "mean_valid_at_shared_epoch_pct": 100.0 * mean_selected_valid,
        "std_valid_at_shared_epoch_pct": 100.0 * std_selected_valid,
        "mean_train_at_shared_epoch_pct": 100.0 * float(np.mean(selected_train_scores)),
        "std_train_at_shared_epoch_pct": 100.0 * float(np.std(selected_train_scores)),
        "folds": int(len(folds)),
        "epochs": int(args.epochs),
        "split_seed": int(args.seed),
        "train_seed": int(getattr(args, "tu_train_seed", 0)),
        "iters_per_epoch": getattr(args, "iters_per_epoch", 50),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "degree_as_tag": bool(meta["degree_as_tag"]),
        "raw_source": str(meta["raw_source"]),
        "pooling": str(args.pooling),
        "hidden_channels": int(args.hidden_channels),
        "local_layers": int(args.local_layers),
        "dropout": float(args.dropout),
        "bn": bool(getattr(args, "bn", False)),
        "linear": bool(getattr(args, "linear", False)),
        "aug_tech": str(args.aug_tech),
        "aug_mode": str(args.aug_mode),
        "p": float(getattr(args, "p", 0.0)),
        "q": float(getattr(args, "q", 0.0)),
        "rade_variant": str(getattr(args, "rade_variant", "rade-of")),
        "mask_sharing": str(getattr(args, "mask_sharing", "layerwise")),
        "pq_gradnorm": bool(getattr(args, "pq_gradnorm", False)),
        "pq_update_unit": str(getattr(args, "pq_update_unit", "batch")),
        "runtime_epoch_mean_sec": float(np.mean(runtime_values)) if runtime_values else "",
        "runtime_epoch_std_sec": float(np.std(runtime_values)) if runtime_values else "",
    }
    _write_single_dataset_summary(single_dataset_record, single_dataset_path)
    print(f"[SINGLE-DATASET-GC] Saved TU compatibility summary CSV row to {single_dataset_path}")

    if pq_enabled and aug_tech == "rade":
        visualization_dir = _gc_path("visualization")
        variant_tag = str(getattr(args, "rade_variant", "rade-of")).lower().strip().replace("-", "")
        lr_tag = str(f"{float(getattr(args, 'pq_adam_lr', 0.1)):g}").replace(".", "p").replace("-", "m")
        rho_tag = "_rhoopt" if bool(getattr(args, "pq_optimize_rho", False)) else ""
        tag = f"{dataset_tag}_{gnn_tag}_{variant_tag}_tu_powerfulgnns_adam_direct_lr{lr_tag}{rho_tag}_gc"
        save_pq_plots(
            p_histories,
            q_histories,
            obj_histories,
            rho_histories,
            tag=tag,
            out_dir=visualization_dir,
            show_runs=True,
        )

    if use_wandb:
        wandb.summary[f"final/shared_epoch"] = int(selected_epoch)
        wandb.summary[f"final/mean_valid_at_shared_epoch_{metric}"] = 100.0 * mean_selected_valid
        wandb.summary[f"final/std_valid_at_shared_epoch_{metric}"] = 100.0 * std_selected_valid
        if runtime_values:
            wandb.summary["runtime/epoch_sec_mean"] = float(np.mean(runtime_values))
            wandb.summary["runtime/epoch_sec_std"] = float(np.std(runtime_values))
        wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser_add_gc_args(parser)
    args = parser.parse_args()
    explicit_arg_names = get_explicit_arg_names(sys.argv[1:])
    cli_explicit_arg_names = set(explicit_arg_names)

    if getattr(args, "linear", False):
        args.dropout = 0.0
        if hasattr(args, "bn"):
            args.bn = False

    use_wandb = bool(getattr(args, "wandb", False)) and wandb is not None and str(getattr(args, "wandb_mode", "online")) != "disabled"
    if use_wandb:
        parser_defaults = dict(vars(args))
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_name,
            tags=[tag.strip() for tag in str(args.wandb_tags).split(",") if tag.strip()],
            mode=args.wandb_mode,
            config=vars(args),
        )
        wandb_config = dict(wandb.config)
        wandb_explicit_arg_names = set()
        for key, value in wandb_config.items():
            if hasattr(args, key):
                if parser_defaults.get(key) != value:
                    wandb_explicit_arg_names.add(key)
                setattr(args, key, value)
        explicit_arg_names.update(wandb_explicit_arg_names)
        cli_explicit_arg_names.update(wandb_explicit_arg_names)
        if getattr(args, "linear", False):
            args.dropout = 0.0
            if hasattr(args, "bn"):
                args.bn = False

    gnn_raw = str(getattr(args, "gnn", "gcn")).lower().strip()
    args.gnn_raw = gnn_raw
    tu_reference_mode = _is_tu_dataset_name(args.dataset)

    preset_key, auto_applied, auto_skipped = apply_gc_auto_defaults(args, explicit_arg_names)
    if preset_key is not None and auto_applied:
        print(
            f"[AUTO-CONFIG] graph-classification preset {preset_key[0]}/{preset_key[1]} applied: "
            + ", ".join(f"{name}={value}" for name, value in auto_applied.items())
        )
    if preset_key is not None and auto_skipped:
        print(
            f"[AUTO-CONFIG] kept explicit overrides for {preset_key[0]}/{preset_key[1]}: "
            + ", ".join(f"{name}={value}" for name, value in auto_skipped.items())
        )

    aug_preset_key, aug_auto_applied, aug_auto_skipped = apply_gc_aug_defaults(args, cli_explicit_arg_names)
    if aug_preset_key is not None and aug_auto_applied:
        print(
            f"[AUTO-CONFIG] graph-classification augmentation preset {aug_preset_key[0]}/{aug_preset_key[1]} applied: "
            + ", ".join(f"{name}={value}" for name, value in aug_auto_applied.items())
        )
    if aug_preset_key is not None and aug_auto_skipped:
        print(
            f"[AUTO-CONFIG] kept explicit augmentation overrides for {aug_preset_key[0]}/{aug_preset_key[1]}: "
            + ", ".join(f"{name}={value}" for name, value in aug_auto_skipped.items())
        )

    if gnn_raw == "sgc":
        args.gnn = "gcn"
        args.linear = True
    elif gnn_raw == "gin-linear":
        args.gnn = "gin"
        args.linear = True

    if bool(getattr(args, "linear", False)) and gnn_raw in {"sgc", "gin-linear"}:
        args.aug_tech = "none"
        args.aug_mode = "none"
        args.ep_correction = False
        args.pq_gradnorm = False
        args.dropout = 0.0
        if hasattr(args, "bn"):
            args.bn = False

    aug_tech = str(getattr(args, "aug_tech", "rade")).lower().strip()
    if aug_tech not in {"rade", "dropout", "dropedge", "dropmessage", "dropnode", "none"}:
        raise ValueError(
            "--aug_tech must be one of {rade, dropout, dropedge, dropmessage, dropnode, none}. "
            f"Got {aug_tech}"
        )
    if aug_tech == "dropout":
        if "dropout" not in explicit_arg_names and float(getattr(args, "dropout", 0.0)) <= 0.0:
            args.dropout = 0.5
    elif aug_tech in {"dropedge", "dropmessage", "dropnode", "none"}:
        if "dropout" not in explicit_arg_names:
            args.dropout = 0.0
    if aug_tech == "dropmessage":
        args.aug_mode = "none"
        args.ep_correction = False
        args.pq_gradnorm = False
        args.p = 0.0
        args.q = 0.0
        _require_no_rade_contrib(args, "dropmessage")
    if aug_tech == "dropnode":
        args.aug_mode = "none"
        args.ep_correction = False
        args.pq_gradnorm = False
        args.p = 0.0
        args.q = 0.0
        _require_no_rade_contrib(args, "dropnode")
    if aug_tech == "dropout":
        args.aug_mode = "none"
        args.ep_correction = False
        args.pq_gradnorm = False
        args.p = 0.0
        args.q = 0.0
        _require_no_rade_contrib(args, "dropout")
    if aug_tech == "dropedge":
        args.ep_correction = False
        args.pq_gradnorm = False
        args.aug_mode = "drop"
        if "p" not in explicit_arg_names:
            args.p = float(getattr(args, "dropedge_rate", 0.2))
        args.q = 0.0
    if aug_tech == "none":
        args.aug_mode = "none"
        args.ep_correction = False
        args.pq_gradnorm = False
        args.p = 0.0
        args.q = 0.0

    pq_enabled = bool(getattr(args, "pq_gradnorm", False))
    if pq_enabled and str(getattr(args, "gnn", "gcn")).lower().strip() == "gat":
        if int(getattr(args, "gat_heads", 1)) != 1:
            raise ValueError("GAT PQ-GradNorm currently supports only --gat_heads 1.")
        if str(getattr(args, "ep_expectation_mode", "analytic")).lower().strip() != "analytic":
            raise ValueError("GAT PQ-GradNorm currently requires --ep_expectation_mode analytic.")
    if pq_enabled and aug_tech != "rade":
        raise ValueError("--pq_gradnorm is only valid with --aug_tech=rade.")
    if pq_enabled and str(getattr(args, "aug_mode", "none")).lower().strip() == "none":
        raise ValueError("--pq_gradnorm requires --aug_mode != none.")
    if pq_enabled and not bool(getattr(args, "ep_correction", False)):
        raise ValueError("--pq_gradnorm currently requires --ep_correction True.")
    pq_update_unit = str(getattr(args, "pq_update_unit", "batch")).lower().strip()
    if pq_update_unit not in {"batch", "epoch"}:
        raise ValueError("--pq_update_unit must be one of {batch, epoch}.")
    args.pq_update_unit = pq_update_unit
    if int(getattr(args, "pq_update_every_batches", 1)) <= 0:
        raise ValueError("--pq_update_every_batches must be >= 1.")
    if float(getattr(args, "pq_update_batch_fraction", 0.0)) < 0.0:
        raise ValueError("--pq_update_batch_fraction must be >= 0.")
    if float(getattr(args, "pq_update_batch_fraction", 0.0)) > 1.0:
        raise ValueError("--pq_update_batch_fraction must be <= 1.")
    pq_p_max = float(getattr(args, "pq_p_max", -1.0))
    if pq_enabled:
        if pq_p_max < 0.0:
            pq_p_max = 0.9
        if not (0.0 <= pq_p_max < 1.0):
            raise ValueError(f"--pq_p_max must be in [0,1) or negative for auto. Got {pq_p_max}.")
        if float(getattr(args, "pq_gat_denom_penalty_lambda", 0.0)) < 0.0:
            raise ValueError("--pq_gat_denom_penalty_lambda must be >= 0.")
        if float(getattr(args, "pq_gat_denom_cv2_threshold", 1.0)) < 0.0:
            raise ValueError("--pq_gat_denom_cv2_threshold must be >= 0.")
        if float(getattr(args, "pq_gat_corr_penalty_lambda", 0.0)) < 0.0:
            raise ValueError("--pq_gat_corr_penalty_lambda must be >= 0.")
        if float(getattr(args, "pq_gat_corr_threshold", 2.0)) < 1.0:
            raise ValueError("--pq_gat_corr_threshold must be >= 1.")
        if float(getattr(args, "gat_ep_corr_clip", 2.0)) < 0.0:
            raise ValueError("--gat_ep_corr_clip must be >= 0.")
    args.pq_p_max = float(pq_p_max)

    args.ep_emp_average_mode = "ema" if pq_enabled else "running_mean"

    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() else torch.device(f"cuda:{int(args.device)}")
    if tu_reference_mode:
        _run_tu_reference_protocol(args, device=device, use_wandb=use_wandb)
        return

    attach_global_node_ids = (
        aug_tech == "rade"
        and bool(getattr(args, "ep_correction", False))
        and str(getattr(args, "ep_expectation_mode", "analytic")).lower().strip() == "empirical_ema"
    )

    dataset, base_split_idx, meta = load_gc_dataset(
        root=args.data_dir,
        name=args.dataset,
        seed=args.seed,
        train_prop=args.train_prop,
        valid_prop=args.valid_prop,
        attach_graph_ids=(aug_tech == "rade"),
        attach_global_node_ids=attach_global_node_ids,
    )
    args.global_num_nodes_total = int(meta.get("global_num_nodes_total", 0))
    graph_density, rho_scale = _compute_graph_density(dataset)
    args.graph_density = float(graph_density)
    args.pq_expected_add_ratio_scale = float(rho_scale)
    if pq_enabled:
        print(f"[PQ] graph_density={graph_density:.6e} rho_scale={rho_scale:.6e}")

    if use_wandb:
        wandb.config.update(vars(args), allow_val_change=True)

    split_idx_list: List[Dict[str, torch.Tensor]] = [
        {key: value.to(torch.long).cpu() for key, value in base_split_idx.items()}
        for _ in range(int(args.runs))
    ]

    dataset_name = str(args.dataset).lower().strip()
    is_ogbg_molhiv = dataset_name == "ogbg-molhiv"
    is_peptides_func = dataset_name == "peptides-func"
    if is_ogbg_molhiv:
        if str(getattr(args, "loss", "auto")).lower().strip() not in {"auto", "bce"}:
            raise ValueError("ogbg-molhiv requires BCE loss for OGB compatibility; use --loss auto or --loss bce.")
        if str(getattr(args, "metric", "auto")).lower().strip() not in {"auto", "rocauc"}:
            raise ValueError("ogbg-molhiv requires ROC-AUC for OGB compatibility; use --metric auto or --metric rocauc.")
    if is_peptides_func:
        if str(getattr(args, "loss", "auto")).lower().strip() not in {"auto", "bce"}:
            raise ValueError("peptides-func requires BCE loss for LRGB compatibility; use --loss auto or --loss bce.")
        if str(getattr(args, "metric", "auto")).lower().strip() not in {"auto", "ap"}:
            raise ValueError("peptides-func requires AP for LRGB compatibility; use --metric auto or --metric ap.")

    in_dim = int(meta["in_dim"])
    out_dim = int(meta["out_dim"])
    loss_type = meta["loss"] if args.loss == "auto" else args.loss
    metric = meta["metric"] if args.metric == "auto" else args.metric
    criterion = nn.CrossEntropyLoss() if loss_type == "ce" else nn.BCEWithLogitsLoss(reduction="none")

    mask_sharing = str(getattr(args, "mask_sharing", "layerwise")).lower().strip()
    if mask_sharing not in {"shared", "layerwise"}:
        raise ValueError(f"mask_sharing must be 'shared' or 'layerwise'. Got {mask_sharing}")

    print(args)
    print(
        f"[CONFIG] dataset={args.dataset} | backbone_raw={args.gnn_raw} -> backbone={args.gnn} | "
        f"linear={bool(getattr(args, 'linear', False))} | aug_tech={args.aug_tech} aug_mode={args.aug_mode} | "
        f"ep_correction={bool(getattr(args, 'ep_correction', False))} "
        f"(mode={getattr(args, 'ep_expectation_mode', 'analytic')}, avg={getattr(args, 'ep_emp_average_mode', 'running_mean')}, "
        f"variant={getattr(args, 'rade_variant', '-')}) | "
        f"pq_gradnorm={pq_enabled}(adam/direct, anchor={getattr(args, 'pq_data_anchor', 'clean')}, "
        f"update_unit={getattr(args, 'pq_update_unit', 'batch')}, "
        f"batch_update_every={int(getattr(args, 'pq_update_every_batches', 1))}, "
        f"batch_update_fraction={float(getattr(args, 'pq_update_batch_fraction', 0.0)):g}, "
        f"rho_opt={bool(getattr(args, 'pq_optimize_rho', False))}, "
        f"p_max={float(getattr(args, 'pq_p_max', 1.0 - 1e-6)):.4f}, "
        f"rho_scale={float(getattr(args, 'pq_expected_add_ratio_scale', 0.0)):.6e}, "
        f"del_pen={float(getattr(args, 'pq_deletion_penalty_lambda', 0.0)):g}, "
        f"dens_pen={float(getattr(args, 'pq_densification_penalty_lambda', 0.0)):g}/"
        f"{getattr(args, 'pq_densification_penalty_type', 'linear')}, "
        f"gat_den={float(getattr(args, 'pq_gat_denom_penalty_lambda', 0.0)):g}@"
        f"{float(getattr(args, 'pq_gat_denom_cv2_threshold', 1.0)):g}, "
        f"gat_corr={float(getattr(args, 'pq_gat_corr_penalty_lambda', 0.0)):g}@"
        f"{float(getattr(args, 'pq_gat_corr_threshold', 2.0)):g}, "
        f"corr_clip={float(getattr(args, 'gat_ep_corr_clip', 2.0)):g}) | "
        f"p={float(getattr(args, 'p', 0.0)):.4f} q={float(getattr(args, 'q', 0.0)):.6f} | mask_sharing={mask_sharing}"
    )
    if is_ogbg_molhiv:
        print(
            "[OGBG-MOLHIV] official OGB split/evaluator | "
            f"epochs={int(args.epochs)} batch_size={int(args.batch_size)} lr={float(args.lr):g} "
            f"hidden={int(args.hidden_channels)} layers={int(args.local_layers)} pooling={args.pooling} | "
            "RADE-local backbone with AtomEncoder only "
            f"(no BondEncoder, no virtual node, bn={bool(getattr(args, 'bn', False))}, "
            f"dropout={float(getattr(args, 'dropout', 0.0)):.4f})"
        )
    if is_peptides_func:
        print(
            "[PEPTIDES-FUNC] official LRGB split | "
            f"epochs={int(args.epochs)} batch_size={int(args.batch_size)} lr={float(args.lr):g} "
            f"hidden={int(args.hidden_channels)} layers={int(args.local_layers)} pooling={args.pooling} | "
            "AdamW + ReduceLROnPlateau | "
            "RADE-local backbone with AtomEncoder only "
            f"(no BondEncoder, no virtual node, bn={bool(getattr(args, 'bn', False))}, "
            f"dropout={float(getattr(args, 'dropout', 0.0)):.4f})"
        )

    all_best_tests = []
    p_histories: List[List[float]] = []
    q_histories: List[List[float]] = []
    obj_histories: List[List[float]] = []
    rho_histories: List[List[float]] = []
    epoch_time_histories: List[List[float]] = []

    for run in range(int(args.runs)):
        fix_seed(int(args.seed) + int(run), device=device)
        loaders = _make_loaders(
            dataset,
            split_idx_list[run],
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            pin_memory=(device.type == "cuda"),
            seed=int(args.seed) + int(run),
        )
        pq_batch_update_interval, train_batches_per_epoch = _resolve_pq_batch_update_interval(args, loaders["train"])
        augmentor_cache: Dict[Tuple[int, bool], BernoulliEdgeAugmentor] = {}

        model = parse_method_gc(args, in_channels=in_dim, out_channels=out_dim, device=device)
        if is_peptides_func:
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=20,
                min_lr=1e-5,
            )
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
            scheduler = None
        print(f"[GC] Trainable params: {count_trainable_params(model):,}")
        print_training_banner(args, model)

        use_ogb_encoders = bool(getattr(args, "use_ogb_encoders", False))
        use_edge_attr = bool(getattr(args, "use_edge_attr", False))
        local_layers = int(getattr(getattr(model, "cfg", None), "local_layers", getattr(args, "local_layers", 1)))
        rade_variant = str(getattr(args, "rade_variant", "rade-of")).lower().strip()
        p_eff, q_eff = _enforce_aug_mode(float(getattr(args, "p", 0.0)), float(getattr(args, "q", 0.0)), args.aug_mode)
        if pq_enabled:
            p_eff = min(float(p_eff), float(getattr(args, "pq_p_max", 1.0 - 1e-6)))

        if attach_global_node_ids:
            if not hasattr(model, "reset_ep_stats"):
                raise RuntimeError("ep_expectation_mode='empirical_ema' requires model.reset_ep_stats(...).")
            model.reset_ep_stats(p_eff, q_eff)

        pq_tuner = None
        if pq_enabled:
            pq_tuner = PQGradNormTunerGC(
                gnn=str(args.gnn),
                rade_variant=rade_variant,
                pooling=str(args.pooling),
                cfg=PQTunerGCConfig(
                    expected_add_ratio_scale=float(getattr(args, "pq_expected_add_ratio_scale", 0.0)),
                    deletion_penalty_lambda=float(getattr(args, "pq_deletion_penalty_lambda", 0.0)),
                    densification_penalty_lambda=float(getattr(args, "pq_densification_penalty_lambda", 1.0)),
                    densification_penalty_type=str(getattr(args, "pq_densification_penalty_type", "linear")),
                    densification_penalty_rho0=float(getattr(args, "pq_densification_penalty_rho0", 1.0)),
                    p_max=float(getattr(args, "pq_p_max", 1.0 - 1e-6)),
                    optimize_rho=bool(getattr(args, "pq_optimize_rho", False)),
                    eps=1e-12,
                    seed=int(args.seed),
                    data_anchor=str(getattr(args, "pq_data_anchor", "clean")),
                    adam_lr=float(getattr(args, "pq_adam_lr", 0.1)),
                    adam_beta1=float(getattr(args, "pq_adam_beta1", 0.9)),
                    adam_beta2=float(getattr(args, "pq_adam_beta2", 0.999)),
                    adam_eps=float(getattr(args, "pq_adam_eps", 1e-8)),
                    ep_expectation_mode=str(getattr(args, "ep_expectation_mode", "analytic")),
                    gat_denom_penalty_lambda=float(getattr(args, "pq_gat_denom_penalty_lambda", 0.0)),
                    gat_denom_cv2_threshold=float(getattr(args, "pq_gat_denom_cv2_threshold", 1.0)),
                    gat_corr_penalty_lambda=float(getattr(args, "pq_gat_corr_penalty_lambda", 0.0)),
                    gat_corr_threshold=float(getattr(args, "pq_gat_corr_threshold", 2.0)),
                ),
            )

        best_valid = -1e18
        best_test = -1e18
        best_epoch = -1
        epochs_no_improve = 0
        patience = int(getattr(args, "patience", 100))
        p_trace: List[float] = []
        q_trace: List[float] = []
        obj_trace: List[float] = []
        rho_trace: List[float] = []
        epoch_time_trace: List[float] = []

        for epoch in range(1, int(args.epochs) + 1):
            _sync_for_timing(device)
            epoch_time_start = time.perf_counter()
            model.train()
            p_trace.append(float(p_eff))
            q_trace.append(float(q_eff))
            rho_trace.append(float(q_eff) * float(rho_scale))
            obj_trace.append(float("nan"))

            epoch_loss_sum_tensor: Optional[torch.Tensor] = None
            epoch_graphs = 0
            dropped_epoch_acc = 0.0
            added_epoch_acc = 0.0
            aug_batches = 0
            stats_are_layer_avg = bool(getattr(args, "ep_correction", False)) and mask_sharing == "layerwise"

            warmup = int(getattr(args, "pq_warmup_epochs", 0))
            every = int(getattr(args, "pq_update_every", 1))
            do_pq_update = pq_tuner is not None and epoch >= warmup and every > 0 and (epoch % every) == 0
            adam_batch_infos: List[Dict[str, float]] = []

            for step, batch in enumerate(loaders["train"]):
                optimizer.zero_grad(set_to_none=True)

                edge_index_keep_obj: Optional[EdgeIndexObj] = None
                edge_index_add_obj: Optional[EdgeIndexObj] = None
                do_aug = (
                    aug_tech in {"rade", "dropedge"}
                    and str(getattr(args, "aug_mode", "none")).lower().strip() != "none"
                    and (p_eff > 0.0 or q_eff > 0.0)
                )
                if do_aug:
                    aug_p = float(p_eff)
                    aug_q = 0.0 if aug_tech == "dropedge" else float(q_eff)
                    aug_mode = "drop" if aug_tech == "dropedge" else str(getattr(args, "aug_mode", "both"))
                    batch, stats, edge_index_keep_obj, edge_index_add_obj = _augment_batch_graphwise(
                        batch,
                        p=aug_p,
                        q=aug_q,
                        mode=aug_mode,
                        seed=int(args.seed) + 10_000 * int(run) + 1_000 * int(epoch) + int(step),
                        device=device,
                        use_edge_attr=use_edge_attr,
                        ep_correction=bool(getattr(args, "ep_correction", False)) and aug_tech == "rade",
                        mask_sharing=mask_sharing,
                        local_layers=local_layers,
                        safety_max_additions=int(getattr(args, "safety_max_additions", 2_000_000)),
                        augmentor_cache=augmentor_cache,
                    )
                    if "dropped_undir" in stats:
                        dropped_epoch_acc += float(stats["dropped_undir"])
                        added_epoch_acc += float(stats["added_undir"])
                        stats_are_layer_avg = False
                    elif "dropped_undir_avg" in stats:
                        dropped_epoch_acc += float(stats["dropped_undir_avg"])
                        added_epoch_acc += float(stats["added_undir_avg"])
                        stats_are_layer_avg = True
                    aug_batches += 1
                else:
                    batch = batch.to(device)

                _maybe_cast_batch_for_linear(batch, use_ogb_encoders=use_ogb_encoders, use_edge_attr=use_edge_attr)
                edge_attr = batch.edge_attr if (use_edge_attr and hasattr(batch, "edge_attr")) else None
                node_ids = getattr(batch, "n_id", None)

                if aug_tech == "dropmessage":
                    out = model(
                        batch.x,
                        batch.edge_index,
                        batch.batch,
                        edge_attr=edge_attr,
                        dropmessage_rate=float(getattr(args, "dropmessage_rate", 0.0)),
                        node_ids=node_ids,
                        p=0.0,
                        q=0.0,
                    )
                elif aug_tech == "dropnode":
                    out = model(
                        batch.x,
                        batch.edge_index,
                        batch.batch,
                        edge_attr=edge_attr,
                        dropnode_rate=float(getattr(args, "dropnode_rate", 0.0)),
                        node_ids=node_ids,
                        p=0.0,
                        q=0.0,
                    )
                elif aug_tech == "dropedge":
                    out = model(
                        batch.x,
                        batch.edge_index,
                        batch.batch,
                        edge_attr=edge_attr,
                        node_ids=node_ids,
                        p=0.0,
                        q=0.0,
                    )
                else:
                    out = model(
                        batch.x,
                        batch.edge_index,
                        batch.batch,
                        edge_attr=edge_attr,
                        edge_index_keep=edge_index_keep_obj if bool(getattr(args, "ep_correction", False)) and do_aug else None,
                        edge_index_add=edge_index_add_obj if bool(getattr(args, "ep_correction", False)) and do_aug else None,
                        p=float(p_eff),
                        q=float(q_eff),
                        node_ids=node_ids,
                    )

                y = batch.y
                if loss_type == "ce":
                    if y.dim() == 2 and y.size(1) == 1:
                        y = y.squeeze(1)
                    y = y.view(-1).to(torch.long)
                    loss = criterion(out, y)
                else:
                    if y.dim() == 1:
                        y = y.view(-1, 1)
                    if out.dim() == 1:
                        out = out.view(-1, 1)
                    loss = masked_bce_with_logits(out, y)

                loss.backward()
                optimizer.step()

                if (
                    attach_global_node_ids
                    and do_aug
                    and bool(getattr(args, "ep_correction", False))
                    and str(getattr(args, "ep_expectation_mode", "analytic")).lower().strip() == "empirical_ema"
                ):
                    if not hasattr(model, "update_ep_stats"):
                        raise RuntimeError("ep_expectation_mode='empirical_ema' requires model.update_ep_stats(...).")
                    model.update_ep_stats(
                        edge_index_keep=edge_index_keep_obj,
                        edge_index_add=edge_index_add_obj,
                        p=float(p_eff),
                        q=float(q_eff),
                    )

                batch_graphs = int(batch.num_graphs) if hasattr(batch, "num_graphs") else int(y.size(0))
                loss_term = loss.detach() * float(batch_graphs)
                epoch_loss_sum_tensor = loss_term if epoch_loss_sum_tensor is None else epoch_loss_sum_tensor + loss_term
                epoch_graphs += batch_graphs

                if do_pq_update and (((int(step) + 1) % int(pq_batch_update_interval)) == 0):
                    p_next, q_next, info = pq_tuner.suggest_pq_batch(
                        model=model,
                        batch=batch,
                        loss_type=str(loss_type),
                        aug_mode=str(getattr(args, "aug_mode", "both")).lower().strip(),
                        current_p=float(p_eff),
                        current_q=float(q_eff),
                        seed=int(args.seed) + 100_000 * int(run) + 10_000 * int(epoch) + int(step),
                        mask_sharing=mask_sharing,
                        num_layers=local_layers,
                    )
                    p_eff, q_eff = _enforce_aug_mode(p_next, q_next, args.aug_mode)
                    adam_batch_infos.append(info)

            epoch_loss_sum = float(epoch_loss_sum_tensor.item()) if epoch_loss_sum_tensor is not None else 0.0
            train_loss = epoch_loss_sum / max(epoch_graphs, 1)
            current_lr = float(optimizer.param_groups[0]["lr"])
            p_trace[-1] = float(p_eff)
            q_trace[-1] = float(q_eff)
            rho_trace[-1] = float(q_eff) * float(rho_scale)
            if adam_batch_infos:
                obj_trace[-1] = _mean([float(item["obj_new"]) for item in adam_batch_infos])

            p_eval, q_eval = 0.0, 0.0
            if aug_tech == "rade" and bool(getattr(args, "ep_correction", False)) and rade_variant == "rade-ofs":
                p_eval, q_eval = float(p_eff), float(q_eff)

            if bool(getattr(args, "full_eval_cpu", False)) and device.type == "cuda":
                model.to(torch.device("cpu"))
                train_score, valid_score, test_score, valid_loss = evaluate_splits(
                    model,
                    loaders,
                    args.dataset,
                    criterion,
                    metric=metric,
                    use_edge_attr=use_edge_attr,
                    device=torch.device("cpu"),
                    p=float(p_eval),
                    q=float(q_eval),
                )
                model.to(device)
                torch.cuda.empty_cache()
            else:
                train_score, valid_score, test_score, valid_loss = evaluate_splits(
                    model,
                    loaders,
                    args.dataset,
                    criterion,
                    metric=metric,
                    use_edge_attr=use_edge_attr,
                    device=device,
                    p=float(p_eval),
                    q=float(q_eval),
                )

            _sync_for_timing(device)
            epoch_time_sec = time.perf_counter() - epoch_time_start
            epoch_time_trace.append(float(epoch_time_sec))

            improved = float(valid_score) > float(best_valid)
            if improved:
                best_valid = float(valid_score)
                best_test = float(test_score)
                best_epoch = int(epoch)
                epochs_no_improve = 0
                if bool(getattr(args, "save_model", False)):
                    _save_checkpoint(args, model, optimizer, run=run + 1, epoch=epoch, best_valid=best_valid)
            else:
                epochs_no_improve += 1

            if scheduler is not None:
                scheduler.step(float(valid_score))

            if epoch % int(getattr(args, "display_step", 1)) == 0:
                msg = (
                    f"[GC] run={run + 1:02d} epoch={epoch:03d} "
                    f"lr={current_lr:.6g} loss={train_loss:.4f} train_{metric}={train_score:.4f} "
                    f"valid_{metric}={valid_score:.4f} test_{metric}={test_score:.4f} "
                    f"best_valid={best_valid:.4f} best_test@best={best_test:.4f} (ep={best_epoch})"
                )
                if aug_tech == "rade" and aug_batches > 0:
                    if stats_are_layer_avg:
                        msg += (
                            f" | drop/add(avg per layer, undir): -{dropped_epoch_acc:.1f} +{added_epoch_acc:.1f} "
                            f"(p={p_eff:.4f}, q={q_eff:.6f}, rho={float(q_eff) * float(rho_scale):.6f})"
                        )
                    else:
                        msg += (
                            f" | drop/add(undir): -{int(dropped_epoch_acc)} +{int(added_epoch_acc)} "
                            f"(p={p_eff:.4f}, q={q_eff:.6f}, rho={float(q_eff) * float(rho_scale):.6f})"
                        )
                elif aug_tech == "dropedge":
                    msg += f" | dropedge={float(p_eff):.4f}"
                    if aug_batches > 0:
                        msg += f" dropped_undir={int(dropped_epoch_acc)}"
                elif aug_tech == "dropmessage":
                    msg += f" | dropmessage={float(getattr(args, 'dropmessage_rate', 0.0)):.4f}"
                elif aug_tech == "dropnode":
                    msg += f" | dropnode={float(getattr(args, 'dropnode_rate', 0.0)):.4f}"
                elif aug_tech == "dropout":
                    msg += f" | dropout={float(getattr(args, 'dropout', 0.0)):.4f}"
                if adam_batch_infos:
                    msg += (
                        f" | pq: Gd={_mean([float(item['G_data']) for item in adam_batch_infos]):.3e} "
                        f"Gr={_mean([float(item['G_reg_new']) for item in adam_batch_infos]):.3e} "
                        f"obj={_mean([float(item['obj_new']) for item in adam_batch_infos]):.2e} "
                        f"rho={_mean([float(item.get('rho_new', float(item['q_new']) * float(rho_scale))) for item in adam_batch_infos]):.6f} "
                        f"gat_den={_mean([float(item.get('gat_denom_penalty_new', 0.0)) for item in adam_batch_infos]):.2e} "
                        f"gat_corr={_mean([float(item.get('gat_corr_penalty_new', 0.0)) for item in adam_batch_infos]):.2e} "
                        f"updates={len(adam_batch_infos)} unit={getattr(args, 'pq_update_unit', 'batch')} "
                        f"interval={int(pq_batch_update_interval)}"
                    )
                msg += f" | epoch_sec={epoch_time_sec:.4f}"
                print(msg)

            if use_wandb and epoch % int(getattr(args, "wandb_log_every", 1)) == 0:
                log_obj = {
                    "epoch": epoch,
                    "run": run + 1,
                    "train/loss": train_loss,
                    "valid/loss": valid_loss,
                    f"train/{metric}": train_score,
                    f"valid/{metric}": valid_score,
                    f"test/{metric}": test_score,
                    f"best/valid_{metric}": best_valid,
                    f"best/test_at_best_valid_{metric}": best_test,
                    "valid/primary": valid_score,
                    "best/valid_primary": best_valid,
                    "aug/tech": aug_tech,
                    "aug/p": float(p_eff),
                    "aug/q": float(q_eff),
                    "aug/rho": float(q_eff) * float(rho_scale),
                    "runtime/epoch_sec": float(epoch_time_sec),
                    "optim/lr": current_lr,
                    "rade/ep_expectation_mode": str(getattr(args, "ep_expectation_mode", "analytic")),
                    "rade/variant": str(getattr(args, "rade_variant", "rade-of")),
                }
                if aug_tech == "dropedge":
                    log_obj["aug/dropedge_rate"] = float(p_eff)
                    if aug_batches > 0:
                        log_obj["aug/dropped_undir"] = float(dropped_epoch_acc)
                elif aug_tech == "dropmessage":
                    log_obj["aug/dropmessage_rate"] = float(getattr(args, "dropmessage_rate", 0.0))
                elif aug_tech == "dropnode":
                    log_obj["aug/dropnode_rate"] = float(getattr(args, "dropnode_rate", 0.0))
                elif aug_tech == "dropout":
                    log_obj["regularization/dropout"] = float(getattr(args, "dropout", 0.0))
                if adam_batch_infos:
                    log_obj.update(
                        {
                            "pq/G_data_mean": _mean([float(item["G_data"]) for item in adam_batch_infos]),
                            "pq/G_reg_best_mean": _mean([float(item["G_reg_best"]) for item in adam_batch_infos]),
                            "pq/G_reg_new_mean": _mean([float(item["G_reg_new"]) for item in adam_batch_infos]),
                            "pq/obj_mean": _mean([float(item["obj"]) for item in adam_batch_infos]),
                            "pq/obj_new_mean": _mean([float(item["obj_new"]) for item in adam_batch_infos]),
                            "pq/p_new": float(p_eff),
                            "pq/q_new": float(q_eff),
                            "pq/rho_new": float(q_eff) * float(rho_scale),
                            "pq/p_max": float(getattr(args, "pq_p_max", 1.0 - 1e-6)),
                            "pq/used_batches": float(len(adam_batch_infos)),
                            "pq/skipped_batches": float(
                                max(0, (train_batches_per_epoch or 0) - len(adam_batch_infos))
                            ),
                            "pq/update_unit": str(getattr(args, "pq_update_unit", "batch")),
                            "pq/update_every_batches": float(pq_batch_update_interval),
                            "pq/update_batch_fraction": float(getattr(args, "pq_update_batch_fraction", 0.0)),
                            "pq/rho_best_mean": _mean(
                                [
                                    float(item.get("rho_best", float(item["q_best"]) * float(rho_scale)))
                                    for item in adam_batch_infos
                                ]
                            ),
                            "pq/rho_new_mean": _mean(
                                [
                                    float(item.get("rho_new", float(item["q_new"]) * float(rho_scale)))
                                    for item in adam_batch_infos
                                ]
                            ),
                            "pq/deletion_penalty_best_mean": _mean(
                                [float(item.get("deletion_penalty_best", 0.0)) for item in adam_batch_infos]
                            ),
                            "pq/deletion_penalty_new_mean": _mean(
                                [float(item.get("deletion_penalty_new", 0.0)) for item in adam_batch_infos]
                            ),
                            "pq/densification_penalty_best_mean": _mean(
                                [float(item.get("densification_penalty_best", 0.0)) for item in adam_batch_infos]
                            ),
                            "pq/densification_penalty_new_mean": _mean(
                                [float(item.get("densification_penalty_new", 0.0)) for item in adam_batch_infos]
                            ),
                            "pq/gat_denom_penalty_best_mean": _mean(
                                [float(item.get("gat_denom_penalty_best", 0.0)) for item in adam_batch_infos]
                            ),
                            "pq/gat_denom_penalty_new_mean": _mean(
                                [float(item.get("gat_denom_penalty_new", 0.0)) for item in adam_batch_infos]
                            ),
                            "pq/gat_denom_cv2_mean_best": _mean(
                                [float(item.get("gat_denom_cv2_mean_best", 0.0)) for item in adam_batch_infos]
                            ),
                            "pq/gat_denom_cv2_mean_new": _mean(
                                [float(item.get("gat_denom_cv2_mean_new", 0.0)) for item in adam_batch_infos]
                            ),
                            "pq/gat_denom_cv2_max_best": _mean(
                                [float(item.get("gat_denom_cv2_max_best", 0.0)) for item in adam_batch_infos]
                            ),
                            "pq/gat_denom_cv2_max_new": _mean(
                                [float(item.get("gat_denom_cv2_max_new", 0.0)) for item in adam_batch_infos]
                            ),
                            "pq/gat_corr_penalty_best_mean": _mean(
                                [float(item.get("gat_corr_penalty_best", 0.0)) for item in adam_batch_infos]
                            ),
                            "pq/gat_corr_penalty_new_mean": _mean(
                                [float(item.get("gat_corr_penalty_new", 0.0)) for item in adam_batch_infos]
                            ),
                            "pq/gat_corr_gain_mean_best": _mean(
                                [float(item.get("gat_corr_gain_mean_best", 0.0)) for item in adam_batch_infos]
                            ),
                            "pq/gat_corr_gain_mean_new": _mean(
                                [float(item.get("gat_corr_gain_mean_new", 0.0)) for item in adam_batch_infos]
                            ),
                            "pq/gat_corr_gain_max_best": _mean(
                                [float(item.get("gat_corr_gain_max_best", 0.0)) for item in adam_batch_infos]
                            ),
                            "pq/gat_corr_gain_max_new": _mean(
                                [float(item.get("gat_corr_gain_max_new", 0.0)) for item in adam_batch_infos]
                            ),
                            "pq/optimize_rho": bool(getattr(args, "pq_optimize_rho", False)),
                            "pq/expected_add_ratio_scale": float(rho_scale),
                            "pq/deletion_penalty_lambda": float(
                                getattr(args, "pq_deletion_penalty_lambda", 0.0)
                            ),
                            "pq/densification_penalty_lambda": float(
                                getattr(args, "pq_densification_penalty_lambda", 0.0)
                            ),
                            "pq/gat_denom_penalty_lambda": float(
                                getattr(args, "pq_gat_denom_penalty_lambda", 0.0)
                            ),
                            "pq/gat_denom_cv2_threshold": float(
                                getattr(args, "pq_gat_denom_cv2_threshold", 1.0)
                            ),
                            "pq/gat_corr_penalty_lambda": float(
                                getattr(args, "pq_gat_corr_penalty_lambda", 0.0)
                            ),
                            "pq/gat_corr_threshold": float(getattr(args, "pq_gat_corr_threshold", 2.0)),
                            "pq/search_method": "adam",
                        }
                    )
                wandb.log(log_obj)

            if patience > 0 and epochs_no_improve >= patience:
                print(
                    f"[EarlyStop] run={run + 1:02d} stopping at epoch {epoch:03d} "
                    f"(no val improvement for {patience} epochs). "
                    f"Best val={best_valid:.4f} at epoch {best_epoch:03d}, test@best={best_test:.4f}."
                )
                break

        all_best_tests.append(best_test)
        p_histories.append(p_trace)
        q_histories.append(q_trace)
        obj_histories.append(obj_trace)
        rho_histories.append(rho_trace)
        epoch_time_histories.append(epoch_time_trace)

    mean_best = float(np.mean(all_best_tests)) if all_best_tests else 0.0
    std_best = float(np.std(all_best_tests)) if all_best_tests else 0.0
    print(f"[GC] done: best_test@best_valid_{metric} = {100.0 * mean_best:.2f} +/- {100.0 * std_best:.2f} (%) over {args.runs} runs")
    runtime_values = [value for history in epoch_time_histories for value in history]
    if runtime_values:
        print(f"[GC] epoch_time_sec = {float(np.mean(runtime_values)):.4f} +/- {float(np.std(runtime_values)):.4f} over all epochs")

    dataset_tag = _safe_tag(args.dataset)
    gnn_tag = _safe_tag(getattr(args, "gnn_raw", getattr(args, "gnn", "gcn")))
    rade_variant_tag = _safe_tag(getattr(args, "rade_variant", "rade-of")).replace("-", "_")
    single_dataset_path = _gc_path("single_dataset", f"results_gc_{dataset_tag}_{gnn_tag}_{rade_variant_tag}.csv")
    single_dataset_record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": str(args.dataset),
        "gnn_raw": str(getattr(args, "gnn_raw", getattr(args, "gnn", "gcn"))),
        "gnn": str(args.gnn),
        "model": "GraphMPNN",
        "metric": str(metric),
        "loss": str(loss_type),
        "final_best_test_mean_pct": 100.0 * mean_best,
        "final_best_test_std_pct": 100.0 * std_best,
        "final_best_test_mean_raw": mean_best,
        "final_best_test_std_raw": std_best,
        "runs": int(args.runs),
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "full_eval_cpu": bool(getattr(args, "full_eval_cpu", False)),
        "runtime_epoch_mean_sec": float(np.mean(runtime_values)) if runtime_values else "",
        "runtime_epoch_std_sec": float(np.std(runtime_values)) if runtime_values else "",
        "runtime_num_epochs": int(len(runtime_values)),
        "train_prop": float(args.train_prop),
        "valid_prop": float(args.valid_prop),
        "pooling": str(args.pooling),
        "hidden_channels": int(args.hidden_channels),
        "local_layers": int(args.local_layers),
        "gat_heads": int(getattr(args, "gat_heads", 1)),
        "gat_concat": bool(getattr(args, "gat_concat", True)),
        "gat_negative_slope": float(getattr(args, "gat_negative_slope", 0.2)),
        "gat_moment_chunk_size": int(getattr(args, "gat_moment_chunk_size", 1024)),
        "gat_moment_mode": str(getattr(args, "gat_moment_mode", "exact")),
        "gat_moment_samples": int(getattr(args, "gat_moment_samples", 256)),
        "gat_nonedge_samples": int(getattr(args, "gat_nonedge_samples", 256)),
        "gat_moment_seed": int(getattr(args, "gat_moment_seed", 0)),
        "gat_ep_corr_clip": float(getattr(args, "gat_ep_corr_clip", 2.0)),
        "dropout": float(args.dropout),
        "bn": bool(getattr(args, "bn", False)),
        "linear": bool(getattr(args, "linear", False)),
        "pre_linear": bool(getattr(args, "pre_linear", False)),
        "lr": float(args.lr),
        "optimizer": "adamw" if is_peptides_func else "adam",
        "scheduler": "reduce_on_plateau" if is_peptides_func else "none",
        "weight_decay": float(args.weight_decay),
        "patience": int(getattr(args, "patience", 0)),
        "use_ogb_encoders": bool(getattr(args, "use_ogb_encoders", False)),
        "use_edge_attr": bool(getattr(args, "use_edge_attr", False)),
        "aug_tech": str(args.aug_tech),
        "aug_mode": str(args.aug_mode),
        "p": float(getattr(args, "p", 0.0)),
        "q": float(getattr(args, "q", 0.0)),
        "dropedge_rate": float(getattr(args, "dropedge_rate", 0.0)),
        "dropmessage_rate": float(getattr(args, "dropmessage_rate", 0.0)),
        "dropnode_rate": float(getattr(args, "dropnode_rate", 0.0)),
        "ep_correction": bool(getattr(args, "ep_correction", False)),
        "ep_expectation_mode": str(getattr(args, "ep_expectation_mode", "analytic")),
        "ep_emp_average_mode": str(getattr(args, "ep_emp_average_mode", "running_mean")),
        "ep_emp_beta": float(getattr(args, "ep_emp_beta", 0.1)),
        "ep_emp_eps": float(getattr(args, "ep_emp_eps", 1e-12)),
        "rade_variant": str(getattr(args, "rade_variant", "rade-of")),
        "mask_sharing": str(getattr(args, "mask_sharing", "layerwise")),
        "graph_density": float(getattr(args, "graph_density", 0.0)),
        "pq_expected_add_ratio_scale": float(getattr(args, "pq_expected_add_ratio_scale", 0.0)),
        "pq_gradnorm": bool(getattr(args, "pq_gradnorm", False)),
        "pq_optimizer": "adam" if bool(getattr(args, "pq_gradnorm", False)) else "",
        "pq_adam_update": (
            "direct_p_and_rho"
            if bool(getattr(args, "pq_gradnorm", False)) and bool(getattr(args, "pq_optimize_rho", False))
            else "direct_p_and_q" if bool(getattr(args, "pq_gradnorm", False)) else ""
        ),
        "pq_optimize_rho": bool(getattr(args, "pq_optimize_rho", False)),
        "pq_p_max": float(getattr(args, "pq_p_max", 1.0 - 1e-6)),
        "pq_warmup_epochs": int(getattr(args, "pq_warmup_epochs", 0)),
        "pq_update_every": int(getattr(args, "pq_update_every", 1)),
        "pq_update_unit": str(getattr(args, "pq_update_unit", "batch")),
        "pq_update_every_batches": int(getattr(args, "pq_update_every_batches", 1)),
        "pq_update_batch_fraction": float(getattr(args, "pq_update_batch_fraction", 0.0)),
        "pq_data_anchor": str(getattr(args, "pq_data_anchor", "clean")),
        "pq_adam_lr": float(getattr(args, "pq_adam_lr", 0.1)),
        "pq_adam_beta1": float(getattr(args, "pq_adam_beta1", 0.9)),
        "pq_adam_beta2": float(getattr(args, "pq_adam_beta2", 0.999)),
        "pq_adam_eps": float(getattr(args, "pq_adam_eps", 1e-8)),
        "pq_deletion_penalty_lambda": float(getattr(args, "pq_deletion_penalty_lambda", 0.0)),
        "pq_densification_penalty_lambda": float(getattr(args, "pq_densification_penalty_lambda", 0.0)),
        "pq_densification_penalty_type": str(getattr(args, "pq_densification_penalty_type", "linear")),
        "pq_densification_penalty_rho0": float(getattr(args, "pq_densification_penalty_rho0", 1.0)),
        "pq_gat_denom_penalty_lambda": float(getattr(args, "pq_gat_denom_penalty_lambda", 0.0)),
        "pq_gat_denom_cv2_threshold": float(getattr(args, "pq_gat_denom_cv2_threshold", 1.0)),
        "pq_gat_corr_penalty_lambda": float(getattr(args, "pq_gat_corr_penalty_lambda", 0.0)),
        "pq_gat_corr_threshold": float(getattr(args, "pq_gat_corr_threshold", 2.0)),
    }
    _write_single_dataset_summary(single_dataset_record, single_dataset_path)
    print(f"[SINGLE-DATASET-GC] Saved summary CSV row to {single_dataset_path}")

    if pq_enabled and aug_tech == "rade":
        visualization_dir = _gc_path("visualization")
        variant_tag = str(getattr(args, "rade_variant", "rade-of")).lower().strip().replace("-", "")
        lr_tag = str(f"{float(getattr(args, 'pq_adam_lr', 0.1)):g}").replace(".", "p").replace("-", "m")
        rho_tag = "_rhoopt" if bool(getattr(args, "pq_optimize_rho", False)) else ""
        deletion_lambda_tag = str(f"{float(getattr(args, 'pq_deletion_penalty_lambda', 0.0)):g}").replace(
            ".", "p"
        ).replace("-", "m")
        penalty_lambda_tag = str(f"{float(getattr(args, 'pq_densification_penalty_lambda', 0.0)):g}").replace(
            ".", "p"
        ).replace("-", "m")
        gat_denom_lambda_tag = str(f"{float(getattr(args, 'pq_gat_denom_penalty_lambda', 0.0)):g}").replace(
            ".", "p"
        ).replace("-", "m")
        gat_denom_threshold_tag = str(f"{float(getattr(args, 'pq_gat_denom_cv2_threshold', 1.0)):g}").replace(
            ".", "p"
        ).replace("-", "m")
        gat_corr_lambda_tag = str(f"{float(getattr(args, 'pq_gat_corr_penalty_lambda', 0.0)):g}").replace(
            ".", "p"
        ).replace("-", "m")
        gat_corr_threshold_tag = str(f"{float(getattr(args, 'pq_gat_corr_threshold', 2.0)):g}").replace(
            ".", "p"
        ).replace("-", "m")
        penalty_tag = (
            f"_del{deletion_lambda_tag}"
            f"_dens{str(getattr(args, 'pq_densification_penalty_type', 'linear')).lower()}"
            f"{penalty_lambda_tag}"
            f"_gatden{gat_denom_lambda_tag}t{gat_denom_threshold_tag}"
            f"_gatcorr{gat_corr_lambda_tag}t{gat_corr_threshold_tag}"
        )
        tag = f"{dataset_tag}_{gnn_tag}_{variant_tag}_adam_direct_lr{lr_tag}{rho_tag}{penalty_tag}_gc"
        save_pq_plots(
            p_histories,
            q_histories,
            obj_histories,
            rho_histories,
            tag=tag,
            out_dir=visualization_dir,
            show_runs=True,
        )

    if use_wandb:
        wandb.summary[f"final/mean_best_test_at_best_valid_{metric}"] = 100.0 * mean_best
        wandb.summary[f"final/std_best_test_at_best_valid_{metric}"] = 100.0 * std_best
        wandb.summary[f"final/raw_mean_best_test_at_best_valid_{metric}"] = mean_best
        wandb.summary[f"final/raw_std_best_test_at_best_valid_{metric}"] = std_best
        if runtime_values:
            wandb.summary["runtime/epoch_sec_mean"] = float(np.mean(runtime_values))
            wandb.summary["runtime/epoch_sec_std"] = float(np.std(runtime_values))
        wandb.finish()


if __name__ == "__main__":
    main()
