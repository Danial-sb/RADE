import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import wandb

from ..full_batch.data_utils import class_rand_splits, eval_acc_nc, eval_f1_nc, make_random_split_idx
from ..full_batch.dataset import load_nc_dataset
from ..full_batch.logger import Logger, save_model, save_result
from ..full_batch.viz_pq import (
    save_pq_objective_surface_plots,
    save_pq_plots,
    save_pq_search_comparison_plots,
)

from .mb_augmentation import BatchBernoulliEdgeAugmentor
from .mb_eval import evaluate_minibatch
from .mb_loaders import NeighborLoaderConfig, build_neighbor_loaders
from .mb_parse import (
    apply_mini_batch_auto_defaults,
    get_explicit_arg_names,
    parse_method,
    parser_add_main_args,
)
from .mb_pq_gradnorm import PQGradNormTunerMB, PQTunerMBConfig, SEARCH_METHODS


def fix_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_deterministic_debug_mode("error")


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
    if getattr(args, "ep_correction", False):
        raise ValueError(f"--aug_tech={name} must be used with --ep_correction False.")
    if getattr(args, "pq_gradnorm", False):
        raise ValueError(f"--aug_tech={name} must be used with --pq_gradnorm False.")
    if str(getattr(args, "aug_mode", "none")).lower().strip() != "none":
        raise ValueError(f"--aug_tech={name} requires --aug_mode none.")
    if getattr(args, "linear", False):
        raise ValueError(f"--aug_tech={name} is incompatible with --linear.")


def _batch_seed(base_seed: int, batch_id: int, layer_id: int = 0) -> int:
    return int(base_seed) + 1_000_003 * int(batch_id) + 100_000_019 * int(layer_id)


def _sync_for_timing(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


parser = argparse.ArgumentParser(description="Mini-batch Training Pipeline for Node Classification")
parser_add_main_args(parser)
args = parser.parse_args()
explicit_arg_names = get_explicit_arg_names(sys.argv[1:])

if getattr(args, "linear", False):
    args.dropout = 0.0
    if hasattr(args, "bn"):
        args.bn = False
    if hasattr(args, "ln"):
        args.ln = False

wandb_run = None
if getattr(args, "wandb", False):
    tags = [tag.strip() for tag in str(getattr(args, "wandb_tags", "")).split(",") if tag.strip()]
    wandb_run = wandb.init(
        project=str(getattr(args, "wandb_project", "rade-nc-mb")),
        entity=getattr(args, "wandb_entity", None) or None,
        group=getattr(args, "wandb_group", None) or None,
        name=getattr(args, "wandb_name", None) or None,
        tags=tags,
        mode=str(getattr(args, "wandb_mode", "online")),
    )

    for key, value in dict(wandb.config).items():
        if hasattr(args, key):
            setattr(args, key, value)
    explicit_arg_names.update({key for key in dict(wandb.config).keys() if hasattr(args, key)})

    if getattr(args, "linear", False):
        args.dropout = 0.0
        if hasattr(args, "bn"):
            args.bn = False
        if hasattr(args, "ln"):
            args.ln = False

gnn_raw = str(getattr(args, "gnn", "gcn")).lower().strip()
args.gnn_raw = gnn_raw

preset_key, auto_applied, auto_skipped = apply_mini_batch_auto_defaults(args, explicit_arg_names)
if preset_key is not None:
    if auto_applied:
        print(
            f"[AUTO-CONFIG] mini-batch preset {preset_key[0]}/{preset_key[1]} applied: "
            + ", ".join(f"{name}={value}" for name, value in auto_applied.items())
        )
    if auto_skipped:
        print(
            f"[AUTO-CONFIG] kept explicit overrides for {preset_key[0]}/{preset_key[1]}: "
            + ", ".join(f"{name}={value}" for name, value in auto_skipped.items())
        )

if gnn_raw == "sgc":
    args.gnn = "gcn"
    args.linear = True
elif gnn_raw == "gin-linear":
    args.gnn = "gin"
    args.linear = True

if getattr(args, "linear", False) and gnn_raw in {"sgc", "gin-linear"}:
    args.aug_tech = "none"
    args.aug_mode = "none"
    args.ep_correction = False
    args.pq_gradnorm = False
    args.dropout = 0.0
    if hasattr(args, "bn"):
        args.bn = False
    if hasattr(args, "ln"):
        args.ln = False

aug_tech = str(getattr(args, "aug_tech", "none")).lower().strip()
if aug_tech not in {"rade", "dropmessage", "dropnode", "none"}:
    raise ValueError(f"--aug_tech must be one of {{rade, dropmessage, dropnode, none}}. Got {aug_tech}")

if aug_tech == "dropmessage":
    _require_no_rade_contrib(args, "dropmessage")
if aug_tech == "dropnode":
    _require_no_rade_contrib(args, "dropnode")
if aug_tech == "none":
    args.aug_mode = "none"

mb_rade_mode = str(getattr(args, "mb_rade_mode", "local")).lower().strip()
if mb_rade_mode not in {"local", "disabled"}:
    raise ValueError(f"--mb_rade_mode must be one of {{local, disabled}}. Got {mb_rade_mode}")

ep_emp_average_mode = "ema" if bool(getattr(args, "pq_gradnorm", False)) else "running_mean"
pq_search_method = str(getattr(args, "pq_search_method", "grid")).lower().strip()

print(args)
print(
    f"[CONFIG] dataset={args.dataset} | backbone_raw={getattr(args, 'gnn_raw', args.gnn)} -> backbone={args.gnn} | "
    f"linear={bool(getattr(args, 'linear', False))} | aug_tech={args.aug_tech} aug_mode={args.aug_mode} | "
    f"ep_correction={bool(getattr(args, 'ep_correction', False))} "
    f"(mode={getattr(args, 'ep_expectation_mode', 'analytic')}, avg={ep_emp_average_mode}, variant={getattr(args, 'rade_variant', '-')}) | "
    f"pq_gradnorm={bool(getattr(args, 'pq_gradnorm', False))} "
    f"(search={getattr(args, 'pq_search_method', '-')}, epoch_mode={getattr(args, 'pq_epoch_objective_mode', '-')}, "
    f"batch_weighting={getattr(args, 'pq_batch_weighting', '-')}, "
    f"compare={bool(getattr(args, 'pq_compare_search_methods', False))}) | "
    f"p={float(getattr(args, 'p', 0.0)):.4f} q={float(getattr(args, 'q', 0.0)):.6f} | "
    f"mask_sharing={getattr(args, 'mask_sharing', '-')} | mb_mode={mb_rade_mode}"
)

if wandb_run is not None:
    wandb.config.update(vars(args), allow_val_change=True)

fix_seed(int(args.seed))
device = torch.device("cpu") if args.cpu else (
    torch.device(f"cuda:{int(args.device)}") if torch.cuda.is_available() else torch.device("cpu")
)

data, base_split_idx = load_nc_dataset(
    root=args.data_dir,
    name=args.dataset,
    normalize_features=True,
    make_undirected_edges=True,
    remove_citeseer_isolated_nodes=bool(getattr(args, "remove_citeseer_isolated_nodes", False)),
    seed=int(args.seed),
    train_prop=float(args.train_prop),
    valid_prop=float(args.valid_prop),
)
data.n_id = torch.arange(data.num_nodes, dtype=torch.long)

n = int(data.num_nodes)
d = int(data.x.size(1))
c = int(data.y.max().item()) + 1
e_undir = int(data.edge_index.size(1)) // 2 if data.edge_index is not None else 0

print(f"dataset {args.dataset} | num nodes {n} | num edges {e_undir} | num feats {d} | num classes {c}")
if wandb_run is not None:
    wandb.config.update(
        {"num_nodes": n, "num_edges": e_undir, "num_feats": d, "num_classes": c},
        allow_val_change=True,
    )

split_idx_lst: List[Dict[str, torch.Tensor]] = []
for run in range(int(args.runs)):
    if getattr(args, "rand_split", False):
        split_idx = make_random_split_idx(n, float(args.train_prop), float(args.valid_prop), seed=int(args.seed) + run)
    elif getattr(args, "rand_split_class", False):
        split_idx = class_rand_splits(
            data.y,
            int(getattr(args, "label_num_per_class", 20)),
            valid_num=int(getattr(args, "valid_num", 500)),
            test_num=int(getattr(args, "test_num", 1000)),
            seed=int(args.seed) + run,
        )
    else:
        split_idx = base_split_idx

    split_idx_lst.append({key: value.to(torch.long).cpu() for key, value in split_idx.items()})

model = parse_method(args, n, c, d, device)
criterion = nn.CrossEntropyLoss()
eval_func = eval_f1_nc if args.dataset.lower() == "flickr" else eval_acc_nc
logger = Logger(int(args.runs), args)
print("MODEL:", model)

num_workers = int(getattr(args, "mb_num_workers", 0))
batch_size_train = int(getattr(args, "batch_size", 2048))
default_num_neighbors = [15, 10, 5]
num_layers = int(getattr(args, "local_layers", 1))
if len(default_num_neighbors) < num_layers:
    default_num_neighbors += [default_num_neighbors[-1]] * (num_layers - len(default_num_neighbors))
default_num_neighbors = tuple(default_num_neighbors[:num_layers])

nl_cfg = NeighborLoaderConfig(
    num_neighbors=default_num_neighbors,
    batch_size_train=batch_size_train,
    batch_size_eval=max(1024, batch_size_train * 2),
    shuffle_train=bool(getattr(args, "mb_shuffle", False)),
    num_workers=num_workers,
    pin_memory=bool(getattr(args, "mb_pin_memory", False)),
    persistent_workers=(num_workers > 0),
    prefetch_factor=2,
)

p_histories: List[List[float]] = []
q_histories: List[List[float]] = []
method_obj_histories = {method: [] for method in SEARCH_METHODS}
surface_snapshots_by_run: Dict[int, List[Dict[str, Any]]] = {}

for run in range(int(args.runs)):
    split_idx = split_idx_lst[run]
    loaders = build_neighbor_loaders(data, split_idx, cfg=nl_cfg)

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    p_eff, q_eff = _enforce_aug_mode(float(getattr(args, "p", 0.0)), float(getattr(args, "q", 0.0)), args.aug_mode)

    if (
        aug_tech == "rade"
        and bool(getattr(args, "ep_correction", False))
        and str(getattr(args, "ep_expectation_mode", "analytic")).lower().strip() == "empirical_ema"
    ):
        if not hasattr(model, "reset_ep_stats"):
            raise RuntimeError("ep_expectation_mode='empirical_ema' requires model.reset_ep_stats(...).")
        model.reset_ep_stats(p_eff, q_eff)

    pq_tuner: Optional[PQGradNormTunerMB] = None
    if bool(getattr(args, "pq_gradnorm", False)):
        if aug_tech != "rade":
            raise ValueError("--pq_gradnorm is only supported when --aug_tech=rade in mini-batch.")
        if str(args.aug_mode).lower().strip() == "none":
            raise ValueError("--pq_gradnorm requires --aug_mode != none.")
        if not bool(getattr(args, "ep_correction", False)):
            raise ValueError("--pq_gradnorm currently requires --ep_correction True.")
        if mb_rade_mode != "local":
            raise ValueError("--pq_gradnorm requires --mb_rade_mode=local.")
        if pq_search_method == "adam" and bool(getattr(args, "pq_compare_search_methods", False)):
            raise ValueError("--pq_compare_search_methods is not supported when --pq_search_method=adam.")

        pq_tuner = PQGradNormTunerMB(
            gnn=str(getattr(args, "gnn", "gin")),
            rade_variant=str(getattr(args, "rade_variant", "rade-of")),
            cfg=PQTunerMBConfig(
                p_max=float(getattr(args, "p_max", 0.6)),
                q_max=float(getattr(args, "q_max", 1e-3)),
                grid_size=int(getattr(args, "pq_grid_size", 11)),
                ema=float(getattr(args, "pq_ema", 0.9)),
                eps=float(getattr(args, "pq_eps", 1e-12)),
                subset_nodes=int(getattr(args, "pq_subset_nodes", 1024)),
                seed=int(getattr(args, "pq_seed", 0)),
                data_anchor=str(getattr(args, "pq_data_anchor", "clean")),
                search_method=str(getattr(args, "pq_search_method", "grid")),
                epoch_objective_mode=str(getattr(args, "pq_epoch_objective_mode", "batch_solutions")),
                batch_weighting=str(getattr(args, "pq_batch_weighting", "node_count")),
                compare_search_methods=bool(getattr(args, "pq_compare_search_methods", False)),
                powell_maxiter=int(getattr(args, "pq_powell_maxiter", 25)),
                powell_xtol=float(getattr(args, "pq_powell_xtol", 1e-3)),
                powell_ftol=float(getattr(args, "pq_powell_ftol", 1e-3)),
                newton_maxiter=int(getattr(args, "pq_newton_maxiter", 10)),
                newton_tol=float(getattr(args, "pq_newton_tol", 1e-4)),
                newton_damping=float(getattr(args, "pq_newton_damping", 1e-4)),
                max_batches_for_tuning=int(getattr(args, "pq_max_batches_for_tuning", 0)),
                adam_lr=float(getattr(args, "pq_adam_lr", 0.2)),
                adam_beta1=float(getattr(args, "pq_adam_beta1", 0.9)),
                adam_beta2=float(getattr(args, "pq_adam_beta2", 0.999)),
                adam_eps=float(getattr(args, "pq_adam_eps", 1e-8)),
            ),
        )

    best_val = float("-inf")
    best_test = float("-inf")
    patience = int(getattr(args, "patience", 100))
    epochs_no_improve = 0
    best_epoch = -1

    p_trace: List[float] = []
    q_trace: List[float] = []
    method_obj_trace = {method: [] for method in SEARCH_METHODS}
    surface_snapshots: List[Dict[str, Any]] = []

    if bool(getattr(args, "save_model", False)):
        save_model(args, model, optimizer, run)

    for epoch in range(int(args.epochs)):
        _sync_for_timing(device)
        epoch_time = time.perf_counter()
        model.train()

        p_epoch_start = float(p_eff)
        q_epoch_start = float(q_eff)
        p_trace.append(p_epoch_start)
        q_trace.append(q_epoch_start)
        for method in SEARCH_METHODS:
            method_obj_trace[method].append(float("nan"))

        epoch_loss_sum = 0.0
        epoch_seed_nodes = 0
        epoch_dropped_sum = 0.0
        epoch_added_sum = 0.0
        epoch_aug_batches = 0

        warmup = int(getattr(args, "pq_warmup_epochs", 0))
        every = int(getattr(args, "pq_update_every", 1))
        do_pq_update = (pq_tuner is not None) and ((epoch + 1) >= warmup) and (((epoch + 1) % every) == 0)
        do_pq_batchwise_adam = do_pq_update and (pq_tuner is not None) and (pq_search_method == "adam")
        do_pq_epochwise = do_pq_update and (pq_tuner is not None) and (pq_search_method != "adam")
        capture_surface_epoch = (
            do_pq_update
            and bool(getattr(args, "pq_surface_plot", False))
            and (run + 1) == max(1, int(getattr(args, "pq_surface_run", 1)))
            and ((epoch + 1) % max(1, int(getattr(args, "pq_surface_every", 1))) == 0)
        )

        tuning_batches: List[Dict[str, Any]] = []
        max_store = int(getattr(args, "pq_store_batches", 0))
        base_seed = int(args.seed) + 10_000 * int(run) + int(epoch)
        pq_epoch_seed = int(args.seed) + 100_000 * int(run) + int(epoch) + int(getattr(args, "pq_seed", 0))
        adam_batch_infos: List[Dict[str, Any]] = []
        adam_surface_snapshot: Optional[Dict[str, Any]] = None

        for batch_id, batch in enumerate(loaders["train"]):
            batch = batch.to(device)
            x_b = batch.x
            edge_index_clean_b = batch.edge_index
            y_b = batch.y
            node_ids_b = getattr(batch, "n_id", None)
            p_batch = float(p_eff)
            q_batch = float(q_eff)

            seed_n = int(getattr(batch, "batch_size", x_b.size(0)))
            if seed_n <= 0:
                continue

            y_seed = y_b[:seed_n]
            if y_seed.dim() == 2 and y_seed.size(1) == 1:
                y_seed = y_seed.squeeze(1)
            y_seed = y_seed.view(-1).to(torch.long)
            train_mask_b = torch.zeros((x_b.size(0),), device=x_b.device, dtype=torch.bool)
            train_mask_b[:seed_n] = True

            edge_index_keep = None
            edge_index_add = None
            aug_stats = None

            dropmessage_rate = float(getattr(args, "dropmessage_rate", 0.0)) if aug_tech == "dropmessage" else 0.0
            dropnode_rate = float(getattr(args, "dropnode_rate", 0.0)) if aug_tech == "dropnode" else 0.0

            do_aug = (
                aug_tech == "rade"
                and mb_rade_mode == "local"
                and str(getattr(args, "aug_mode", "none")).lower().strip() != "none"
                and (p_batch > 0.0 or q_batch > 0.0)
            )

            if do_aug:
                if str(getattr(args, "mask_sharing", "shared")).lower().strip() == "shared":
                    _, edge_index_keep, edge_index_add, aug_stats = BatchBernoulliEdgeAugmentor.augment_edge_indices(
                        edge_index_clean=edge_index_clean_b,
                        num_nodes=int(x_b.size(0)),
                        p=float(p_batch),
                        q=float(q_batch),
                        mode=str(getattr(args, "aug_mode", "both")),
                        seed=_batch_seed(base_seed, batch_id, 0),
                        device=device,
                    )
                else:
                    edge_index_keep = []
                    edge_index_add = []
                    stats_list = []
                    for layer in range(int(getattr(args, "local_layers", 1))):
                        _, edge_index_keep_l, edge_index_add_l, stats_l = BatchBernoulliEdgeAugmentor.augment_edge_indices(
                            edge_index_clean=edge_index_clean_b,
                            num_nodes=int(x_b.size(0)),
                            p=float(p_batch),
                            q=float(q_batch),
                            mode=str(getattr(args, "aug_mode", "both")),
                            seed=_batch_seed(base_seed, batch_id, layer),
                            device=device,
                        )
                        edge_index_keep.append(edge_index_keep_l)
                        edge_index_add.append(edge_index_add_l)
                        stats_list.append(stats_l)

                    aug_stats = {
                        "dropped_undir_avg": float(sum(stats["dropped_undir"] for stats in stats_list)) / float(len(stats_list)),
                        "added_undir_avg": float(sum(stats["added_undir"] for stats in stats_list)) / float(len(stats_list)),
                    }

            if aug_stats is not None:
                if "dropped_undir" in aug_stats and "added_undir" in aug_stats:
                    epoch_dropped_sum += float(aug_stats["dropped_undir"])
                    epoch_added_sum += float(aug_stats["added_undir"])
                    epoch_aug_batches += 1
                elif "dropped_undir_avg" in aug_stats and "added_undir_avg" in aug_stats:
                    epoch_dropped_sum += float(aug_stats["dropped_undir_avg"])
                    epoch_added_sum += float(aug_stats["added_undir_avg"])
                    epoch_aug_batches += 1

            optimizer.zero_grad(set_to_none=True)

            if aug_tech == "dropmessage":
                logits_b = model(
                    x_b,
                    edge_index_clean_b,
                    dropmessage_rate=dropmessage_rate,
                    node_ids=node_ids_b,
                )
            elif aug_tech == "dropnode":
                logits_b = model(
                    x_b,
                    edge_index_clean_b,
                    dropnode_rate=dropnode_rate,
                    node_ids=node_ids_b,
                )
            elif aug_tech == "rade" and do_aug:
                logits_b = model(
                    x_b,
                    edge_index_clean_b,
                    edge_index_keep=edge_index_keep,
                    edge_index_add=edge_index_add,
                    p=float(p_batch),
                    q=float(q_batch),
                    node_ids=node_ids_b,
                )
            else:
                logits_b = model(x_b, edge_index_clean_b, node_ids=node_ids_b)

            loss_b = criterion(logits_b[:seed_n], y_seed)
            loss_b.backward()
            optimizer.step()

            if (
                aug_tech == "rade"
                and do_aug
                and bool(getattr(args, "ep_correction", False))
                and str(getattr(args, "ep_expectation_mode", "analytic")).lower().strip() == "empirical_ema"
            ):
                if not hasattr(model, "update_ep_stats"):
                    raise RuntimeError("ep_expectation_mode='empirical_ema' requires model.update_ep_stats(...).")
                model.update_ep_stats(
                    edge_index_keep=edge_index_keep,
                    edge_index_add=edge_index_add,
                    p=float(p_batch),
                    q=float(q_batch),
                )

            epoch_loss_sum += float(loss_b.item()) * seed_n
            epoch_seed_nodes += seed_n

            if do_pq_batchwise_adam:
                batch_capture_surface = capture_surface_epoch and adam_surface_snapshot is None
                p_next, q_next, info_b = pq_tuner.suggest_pq_for_batch(
                    model=model,
                    x_b=x_b,
                    edge_index_b_clean=edge_index_clean_b,
                    y_b=y_b,
                    train_mask_b=train_mask_b,
                    criterion=criterion,
                    current_p=float(p_batch),
                    current_q=float(q_batch),
                    aug_mode=str(getattr(args, "aug_mode", "both")).lower().strip(),
                    epoch_seed=int(pq_epoch_seed),
                    batch_id=int(batch_id),
                    node_ids_b=node_ids_b,
                    mask_sharing=str(getattr(args, "mask_sharing", "shared")).lower().strip(),
                    num_layers=int(getattr(args, "local_layers", 1)),
                    capture_surface=batch_capture_surface,
                    surface_grid_size=int(getattr(args, "pq_surface_grid_size", 31)),
                )
                if not bool(info_b.get("skip", 0.0)):
                    p_eff, q_eff = _enforce_aug_mode(p_next, q_next, args.aug_mode)
                    adam_batch_infos.append(info_b)
                    if batch_capture_surface and ("surface_data" in info_b):
                        adam_surface_snapshot = dict(info_b["surface_data"])
                        adam_surface_snapshot["epoch"] = int(epoch + 1)
                        adam_surface_snapshot["run"] = int(run + 1)

            if do_pq_epochwise and max_store > 0 and len(tuning_batches) < max_store:
                tuning_batches.append(
                    {
                        "x": x_b.detach().cpu(),
                        "edge_index": edge_index_clean_b.detach().cpu(),
                        "y": y_b.detach().cpu(),
                        "train_mask": train_mask_b.detach().cpu(),
                        "node_ids": None if node_ids_b is None else node_ids_b.detach().cpu(),
                    }
                )

        avg_train_loss = epoch_loss_sum / max(epoch_seed_nodes, 1)
        p_eval_epoch = float(p_eff)
        q_eval_epoch = float(q_eff)

        model.eval()
        forward_eval: Dict[str, Any] = {}
        if aug_tech == "dropmessage":
            forward_eval = {"dropmessage_rate": 0.0}
        elif aug_tech == "dropnode":
            forward_eval = {"dropnode_rate": 0.0}
        elif (
            aug_tech == "rade"
            and bool(getattr(args, "ep_correction", False))
            and mb_rade_mode == "local"
            and str(getattr(args, "rade_variant", "rade-of")).lower().strip() == "rade-ofs"
            and float(q_eval_epoch) > 0.0
        ):
            forward_eval = {"p": float(p_eval_epoch), "q": float(q_eval_epoch)}

        if bool(getattr(args, "full_eval_cpu", False)) and device.type == "cuda":
            model.to(torch.device("cpu"))
            tr, va, te, va_loss, _ = evaluate_minibatch(
                model=model,
                data=data,
                loaders=loaders,
                split_idx=split_idx,
                eval_func=eval_func,
                criterion=criterion,
                device=torch.device("cpu"),
                forward_kwargs=forward_eval,
            )
            model.to(device)
            torch.cuda.empty_cache()
        else:
            tr, va, te, va_loss, _ = evaluate_minibatch(
                model=model,
                data=data,
                loaders=loaders,
                split_idx=split_idx,
                eval_func=eval_func,
                criterion=criterion,
                device=device,
                forward_kwargs=forward_eval,
            )

        result = (float(tr), float(va), float(te), float(va_loss), None)
        logger.add_result(run, result[:4])

        improved = result[1] > best_val
        if improved:
            best_val = float(result[1])
            best_test = float(result[2])
            best_epoch = int(epoch)
            epochs_no_improve = 0
            if bool(getattr(args, "save_model", False)):
                save_model(args, model, optimizer, run)
        else:
            epochs_no_improve += 1

        if do_pq_batchwise_adam and adam_batch_infos:
            def _mean_info(key: str) -> float:
                values = [float(item[key]) for item in adam_batch_infos if key in item]
                return float(sum(values) / max(len(values), 1))

            if adam_surface_snapshot is not None:
                surface_snapshots.append(adam_surface_snapshot)

            p_trace[-1] = float(p_eff)
            q_trace[-1] = float(q_eff)

            print(
                f"[PQ-GradNorm-MB-Adam] run={run + 1:02d} epoch={epoch + 1:03d} "
                f"updates={len(adam_batch_infos):03d} "
                f"p(last)={float(p_eff):.4f} q(last)={float(q_eff):.6f} "
                f"G_data_mean={_mean_info('G_data'):.3e} "
                f"G_reg_mean={_mean_info('G_reg_best'):.3e} "
                f"obj_mean={_mean_info('obj'):.2e}"
            )

            if wandb_run is not None:
                wandb.log(
                    {
                        "pq/G_data_mean": _mean_info("G_data"),
                        "pq/G_reg_best_mean": _mean_info("G_reg_best"),
                        "pq/G_reg_new_mean": _mean_info("G_reg_new"),
                        "pq/obj_mean": _mean_info("obj"),
                        "pq/obj_new_mean": _mean_info("obj_new"),
                        "pq/p_epoch_best": _mean_info("p_best"),
                        "pq/q_epoch_best": _mean_info("q_best"),
                        "pq/p_new": float(p_eff),
                        "pq/q_new": float(q_eff),
                        "pq/used_batches": float(len(adam_batch_infos)),
                        "pq/skipped_batches": 0.0,
                        "pq/search_method": "adam",
                        "pq/adam_lr": float(getattr(args, "pq_adam_lr", 0.2)),
                    },
                    step=(epoch + run * int(args.epochs)),
                )

        if do_pq_epochwise and pq_tuner is not None:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            batches_for_tuning = tuning_batches if len(tuning_batches) > 0 else loaders["train"]
            p_new, q_new, info = pq_tuner.suggest_pq_for_epoch(
                model=model,
                batches=batches_for_tuning,
                criterion=criterion,
                current_p=float(p_epoch_start),
                current_q=float(q_epoch_start),
                aug_mode=str(getattr(args, "aug_mode", "both")).lower().strip(),
                epoch_seed=int(pq_epoch_seed),
                mask_sharing=str(getattr(args, "mask_sharing", "shared")).lower().strip(),
                num_layers=int(getattr(args, "local_layers", 1)),
                capture_surface=capture_surface_epoch,
                surface_grid_size=int(getattr(args, "pq_surface_grid_size", 31)),
            )

            p_eff, q_eff = _enforce_aug_mode(p_new, q_new, args.aug_mode)

            if bool(getattr(args, "pq_compare_search_methods", False)):
                methods = [
                    method for method in str(info.get("comparison_methods", "")).split(",") if method
                ]
                for method in methods:
                    method_obj_trace[method][-1] = float(info[f"comparison_{method}_obj"])

            if capture_surface_epoch and ("surface_data" in info):
                surface_snapshot = dict(info["surface_data"])
                surface_snapshot["epoch"] = int(epoch + 1)
                surface_snapshot["run"] = int(run + 1)
                surface_snapshots.append(surface_snapshot)

            print(
                f"[PQ-GradNorm-MB] run={run + 1:02d} epoch={epoch + 1:03d} "
                f"mode={getattr(args, 'pq_epoch_objective_mode', 'batch_solutions')} "
                f"p_epoch_best={info['p_epoch_best']:.4f} q_epoch_best={info['q_epoch_best']:.6f} "
                f"p_new={info['p_new']:.4f} q_new={info['q_new']:.6f} "
                f"G_data_mean={info['G_data_mean']:.3e} G_reg_best_mean={info['G_reg_best_mean']:.3e} obj_mean={info['obj_mean']:.2e} "
                f"used_batches={int(info['used_batches'])} skipped_batches={int(info['skipped_batches'])}"
            )

            if bool(getattr(args, "pq_compare_search_methods", False)):
                methods = [
                    method for method in str(info.get("comparison_methods", "")).split(",") if method
                ]
                compare_parts = [
                    (
                        f"{method}:obj={info[f'comparison_{method}_obj']:.2e} "
                        f"(p={info[f'comparison_{method}_p_best']:.4f}, "
                        f"q={info[f'comparison_{method}_q_best']:.6f}, "
                        f"evals={int(info[f'comparison_{method}_n_eval'])})"
                    )
                    for method in methods
                ]
                print(
                    f"[PQ-Compare-MB] run={run + 1:02d} epoch={epoch + 1:03d} "
                    + " | ".join(compare_parts)
                    + " | "
                    f"better={info['comparison_better_method']}"
                )

            if wandb_run is not None:
                pq_log = {
                    "pq/G_data_mean": float(info["G_data_mean"]),
                    "pq/G_reg_best_mean": float(info["G_reg_best_mean"]),
                    "pq/obj_mean": float(info["obj_mean"]),
                    "pq/epoch_objective_mode": str(getattr(args, "pq_epoch_objective_mode", "batch_solutions")),
                    "pq/batch_weighting": str(getattr(args, "pq_batch_weighting", "node_count")),
                    "pq/p_epoch_best": float(info["p_epoch_best"]),
                    "pq/q_epoch_best": float(info["q_epoch_best"]),
                    "pq/p_new": float(info["p_new"]),
                    "pq/q_new": float(info["q_new"]),
                    "pq/used_batches": float(info["used_batches"]),
                    "pq/skipped_batches": float(info["skipped_batches"]),
                }
                if bool(getattr(args, "pq_compare_search_methods", False)):
                    methods = [
                        method for method in str(info.get("comparison_methods", "")).split(",") if method
                    ]
                    pq_log["pq_compare/enabled"] = float(info["comparison_enabled"])
                    pq_log["pq_compare/best_obj"] = float(info["comparison_best_obj"])
                    pq_log["pq_compare/obj_spread"] = float(info["comparison_obj_spread"])
                    for method in methods:
                        pq_log[f"pq_compare/{method}_obj"] = float(info[f"comparison_{method}_obj"])
                        pq_log[f"pq_compare/{method}_p_best"] = float(info[f"comparison_{method}_p_best"])
                        pq_log[f"pq_compare/{method}_q_best"] = float(info[f"comparison_{method}_q_best"])
                        pq_log[f"pq_compare/{method}_G_reg_best"] = float(info[f"comparison_{method}_G_reg_best"])
                        pq_log[f"pq_compare/{method}_n_eval"] = float(info[f"comparison_{method}_n_eval"])
                        pq_log[f"pq_compare/{method}_solver_success"] = float(
                            info[f"comparison_{method}_solver_success"]
                        )
                wandb.log(pq_log, step=(epoch + run * int(args.epochs)))

        _sync_for_timing(device)
        epoch_time = time.perf_counter() - epoch_time
        logger.add_runtime(run, epoch_time)

        if wandb_run is not None and (epoch % int(getattr(args, "wandb_log_every", 1)) == 0):
            step = epoch + run * int(args.epochs)
            log_dict = {
                "aug/tech": str(aug_tech),
                "aug/mb_rade_mode": str(mb_rade_mode),
                "run": int(run + 1),
                "epoch": int(epoch + 1),
                "train/loss_mb": float(avg_train_loss),
                "train/acc": float(result[0]),
                "valid/acc": float(result[1]),
                "test/acc": float(result[2]),
                "valid/loss": float(result[3]),
                "best/valid_acc": float(best_val),
                "best/test_acc_at_best_valid": float(best_test),
                "runtime/epoch_sec": float(epoch_time),
            }

            if aug_tech == "rade":
                log_dict["aug/p"] = float(p_eval_epoch)
                log_dict["aug/q"] = float(q_eval_epoch)
                log_dict["rade/ep_correction"] = bool(getattr(args, "ep_correction", False))
                log_dict["rade/ep_expectation_mode"] = str(getattr(args, "ep_expectation_mode", "analytic"))
                log_dict["rade/ep_emp_average_mode"] = ep_emp_average_mode
                log_dict["rade/variant"] = str(getattr(args, "rade_variant", "rade-of"))
                if epoch_aug_batches > 0:
                    log_dict["aug/dropped_undir_mean_per_batch"] = float(epoch_dropped_sum) / float(epoch_aug_batches)
                    log_dict["aug/added_undir_mean_per_batch"] = float(epoch_added_sum) / float(epoch_aug_batches)
            elif aug_tech == "dropmessage":
                log_dict["aug/dropmessage_rate"] = float(getattr(args, "dropmessage_rate", 0.0))
            elif aug_tech == "dropnode":
                log_dict["aug/dropnode_rate"] = float(getattr(args, "dropnode_rate", 0.0))

            wandb.log(log_dict, step=step)

        if epoch % int(getattr(args, "display_step", 1)) == 0:
            aug_str = ""
            if aug_tech == "rade":
                if epoch_aug_batches > 0:
                    d_mean = float(epoch_dropped_sum) / float(epoch_aug_batches)
                    a_mean = float(epoch_added_sum) / float(epoch_aug_batches)
                    aug_str = f", Drop/Add(mean per batch): -{d_mean:.2f} +{a_mean:.2f} undir"
                aug_str += (
                    f" (p={p_eval_epoch:.4f}, q={q_eval_epoch:.6f}, mb_mode={mb_rade_mode}, "
                    f"ep_correction={bool(getattr(args, 'ep_correction', False))}, "
                    f"ep_mode={getattr(args, 'ep_expectation_mode', 'analytic')}, "
                    f"ep_avg={ep_emp_average_mode}, "
                    f"variant={getattr(args, 'rade_variant', 'rade-of')})"
                )
            elif aug_tech == "dropmessage":
                aug_str = f" (dropmessage={float(getattr(args, 'dropmessage_rate', 0.0)):.3f})"
            elif aug_tech == "dropnode":
                aug_str = f" (dropnode={float(getattr(args, 'dropnode_rate', 0.0)):.3f})"

            aug_str += f", Epoch Time: {epoch_time:.4f}s"
            print(
                f"Run: {run + 1:02d}, Epoch: {epoch + 1:03d}, TrainLoss: {avg_train_loss:.4f}, "
                f"Train: {100 * result[0]:.2f}%, Valid: {100 * result[1]:.2f}%, Test: {100 * result[2]:.2f}%, "
                f"Best Valid: {100 * best_val:.2f}%, Best Test: {100 * best_test:.2f}%{aug_str}"
            )

        if patience > 0 and epochs_no_improve >= patience:
            stop_epoch = epoch + 1
            best_epoch_disp = best_epoch + 1 if best_epoch >= 0 else -1
            print(
                f"[EarlyStop] run={run + 1:02d} stopping at epoch {stop_epoch:03d} "
                f"(no val improvement for {patience} epochs). "
                f"Best val={best_val:.4f} at epoch {best_epoch_disp:03d}, test@best={best_test:.4f}."
            )
            if wandb_run is not None:
                step = epoch + run * int(args.epochs)
                wandb.log(
                    {
                        "early_stop/patience": int(patience),
                        "early_stop/stop_epoch": int(stop_epoch),
                        "early_stop/best_epoch": int(best_epoch_disp),
                        "early_stop/best_val": float(best_val),
                        "early_stop/best_test_at_best_val": float(best_test),
                    },
                    step=step,
                )
            break

    logger.print_statistics(run)
    p_histories.append(p_trace)
    q_histories.append(q_trace)
    for method in SEARCH_METHODS:
        method_obj_histories[method].append(method_obj_trace[method])
    if surface_snapshots:
        surface_snapshots_by_run[run + 1] = surface_snapshots


results = logger.print_statistics()
runtime_summary = logger.get_runtime_summary()

if getattr(args, "pq_gradnorm", False) and str(getattr(args, "aug_tech", "rade")).lower().strip() == "rade":
    dataset_tag = str(args.dataset).lower().strip()
    gnn_tag = str(getattr(args, "gnn_raw", getattr(args, "gnn", "gcn"))).lower().strip()
    variant_tag = str(getattr(args, "rade_variant", "rade-of")).lower().strip().replace("-", "")
    search_tag = str(getattr(args, "pq_search_method", "grid")).lower().strip()
    tag = f"{dataset_tag}_{gnn_tag}_{variant_tag}_{search_tag}_mb"

    save_pq_plots(p_histories, q_histories, tag=tag, out_dir="visualization", show_runs=True)
    if bool(getattr(args, "pq_compare_search_methods", False)):
        save_pq_search_comparison_plots(method_obj_histories, tag=tag, out_dir="visualization", show_runs=True)
    surface_run = max(1, int(getattr(args, "pq_surface_run", 1)))
    if bool(getattr(args, "pq_surface_plot", False)) and surface_run in surface_snapshots_by_run:
        save_pq_objective_surface_plots(
            surface_snapshots_by_run[surface_run],
            tag=tag,
            run_idx=surface_run,
            out_dir="visualization",
            create_animation=bool(getattr(args, "pq_surface_animation", False)),
        )

save_result(args, results, runtime_summary=runtime_summary)

if wandb_run is not None:
    try:
        wandb_run.summary["final_test_mean"] = float(results.mean().item())
        wandb_run.summary["final_test_std"] = float(results.std().item())
        if runtime_summary is not None:
            wandb_run.summary["runtime/profile"] = "all_mini_batch_runs"
            wandb_run.summary["runtime/epoch_sec_run_mean"] = float(runtime_summary["run_mean"])
            wandb_run.summary["runtime/epoch_sec_run_std"] = float(runtime_summary["run_std"])
            wandb_run.summary["runtime/epoch_sec_pooled_mean"] = float(runtime_summary["pooled_mean"])
    except Exception:
        pass
    wandb.finish()
