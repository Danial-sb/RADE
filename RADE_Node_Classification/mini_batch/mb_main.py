# nc_minibatch/mb_main.py
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import random
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
import wandb

from ..full_batch.logger import Logger, save_model, save_result
from ..full_batch.dataset import load_nc_dataset
from ..full_batch.data_utils import eval_acc_nc, class_rand_splits, make_random_split_idx, eval_f1_nc

from .mb_parse import parse_method, parser_add_main_args
from .mb_loaders import build_neighbor_loaders, NeighborLoaderConfig
from .mb_eval import evaluate_minibatch
from .mb_augmentation import BatchBernoulliEdgeAugmentor

from .mb_pq_gradnorm import PQGradNormTunerMB, PQTunerMBConfig


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
    if getattr(args, "unbiased", False):
        raise ValueError(f"--aug_tech={name} must be used with --unbiased False.")
    if getattr(args, "pq_gradnorm", False):
        raise ValueError(f"--aug_tech={name} must be used with --pq_gradnorm False.")
    if str(getattr(args, "aug_mode", "none")).lower().strip() != "none":
        raise ValueError(f"--aug_tech={name} requires --aug_mode none (edge augmentation disabled).")
    if getattr(args, "linear", False):
        raise ValueError(f"--aug_tech={name} is incompatible with --linear (strict linearity).")


def _batch_seed(base_seed: int, batch_id: int, layer_id: int = 0) -> int:
    # Deterministic, well-spread integers
    return int(base_seed) + 1_000_003 * int(batch_id) + 100_000_19 * int(layer_id)


# -----------------------
# Parse args
# -----------------------
parser = argparse.ArgumentParser(description="Mini-batch Training Pipeline for Node Classification")
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

# Enforce strict linearity at CLI level (before wandb overrides)
if getattr(args, "linear", False):
    args.dropout = 0.0
    if hasattr(args, "bn"):
        args.bn = False
    if hasattr(args, "ln"):
        args.ln = False

# -----------------------
# wandb init (optional)
# -----------------------
wandb_run = None
if getattr(args, "wandb", False):
    if wandb is None:
        raise ImportError("wandb is not installed. Run: pip install wandb")

    tags = [t.strip() for t in str(getattr(args, "wandb_tags", "")).split(",") if t.strip()]

    wandb_run = wandb.init(
        project=str(getattr(args, "wandb_project", "rade-nc-mb")),
        entity=getattr(args, "wandb_entity", None) or None,
        group=getattr(args, "wandb_group", None) or None,
        name=getattr(args, "wandb_name", None) or None,
        tags=tags,
        mode=str(getattr(args, "wandb_mode", "online")),
    )

    # Apply sweep overrides
    for k, v in dict(wandb.config).items():
        if hasattr(args, k):
            setattr(args, k, v)
    wandb.config.update(vars(args), allow_val_change=True)

    # Re-apply strict linear constraints after overrides
    if getattr(args, "linear", False):
        args.dropout = 0.0
        if hasattr(args, "bn"):
            args.bn = False
        if hasattr(args, "ln"):
            args.ln = False

# -----------------------
# Augmentation technique routing
# -----------------------
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

fix_seed(int(args.seed))

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device(f"cuda:{int(args.device)}") if torch.cuda.is_available() else torch.device("cpu")

# -----------------------
# Load data (KEEP ON CPU for NeighborLoader)
# -----------------------
data, base_split_idx = load_nc_dataset(
    args.data_dir,
    args.dataset,
    True,
    True,
    int(args.seed),
    float(args.train_prop),
    float(args.valid_prop),
)
data.n_id = torch.arange(data.num_nodes, dtype=torch.long)  # keep on CPU

# Basic dataset stats
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

# -----------------------
# Build split list for runs
# -----------------------
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

    split_idx = {k: v.to(torch.long).cpu() for k, v in split_idx.items()}
    split_idx_lst.append(split_idx)

# -----------------------
# Model / loss / metric
# -----------------------
model = parse_method(args, n, c, d, device)
criterion = nn.CrossEntropyLoss()
eval_func = eval_f1_nc if args.dataset.lower() == "flickr" else eval_acc_nc
logger = Logger(int(args.runs), args)

print("MODEL:", model)

# -----------------------
# NeighborLoader config
# -----------------------
# Map the CLI knobs into the NeighborLoaderConfig used by mb_loaders.py
# -----------------------
# NeighborLoader config
# -----------------------
num_workers = int(getattr(args, "mb_num_workers", 0))
batch_size_train = int(getattr(args, "batch_size", 2048))

# Default fanouts (can later be exposed as CLI)
default_num_neighbors = (15, 10, 5)
L = int(getattr(args, "local_layers", 1))
fanout = list(default_num_neighbors)
if len(fanout) < L:
    fanout += [fanout[-1]] * (L - len(fanout))
elif len(fanout) > L:
    fanout = fanout[:L]
default_num_neighbors = tuple(fanout)


nl_cfg = NeighborLoaderConfig(
    num_neighbors=default_num_neighbors,
    batch_size_train=batch_size_train,
    batch_size_eval=max(1024, batch_size_train * 2),
    shuffle_train=bool(getattr(args, "mb_shuffle", False)),
    num_workers=num_workers,
    pin_memory=bool(getattr(args, "mb_pin_memory", False)),
    persistent_workers=(num_workers > 0),
    prefetch_factor=2,  # mb_loaders.py should only pass this when num_workers > 0
)


# -----------------------
# Training loop
# -----------------------
for run in range(int(args.runs)):
    split_idx = split_idx_lst[run]

    # Build NeighborLoaders for this run/split
    loaders = build_neighbor_loaders(data, split_idx, cfg=nl_cfg)

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    # (p,q) used for THIS run
    p_eff, q_eff = _enforce_aug_mode(float(getattr(args, "p", 0.0)), float(getattr(args, "q", 0.0)), args.aug_mode)

    # PQ tuner (mini_batch)
    pq_tuner: Optional[PQGradNormTunerMB] = None
    pq_cfg: Optional[PQTunerMBConfig] = None
    if bool(getattr(args, "pq_gradnorm", False)):
        if aug_tech != "rade":
            raise ValueError("--pq_gradnorm is only supported when --aug_tech=rade in this mini_batch script.")
        if str(args.aug_mode).lower().strip() == "none":
            raise ValueError("--pq_gradnorm requires --aug_mode != none.")
        if not bool(getattr(args, "unbiased", False)):
            raise ValueError("--pq_gradnorm currently implemented for RADE/RADE-IC only (use --unbiased True).")
        if mb_rade_mode != "local":
            raise ValueError("--pq_gradnorm requires --mb_rade_mode=local (batch-local RADE semantics).")

        pq_cfg = PQTunerMBConfig(
            p_max=float(getattr(args, "p_max", 0.6)),
            q_max=float(getattr(args, "q_max", 1e-3)),
            grid_size=int(getattr(args, "pq_grid_size", 11)),
            epoch_ema=float(getattr(args, "pq_ema", 0.9)),
            eps=float(getattr(args, "pq_eps", 1e-12)),
            subset_nodes=int(getattr(args, "pq_subset_nodes", 1024)),
            seed=int(getattr(args, "pq_seed", 0)),
            # optional speed knob (not in parser by default)
            max_batches_for_tuning=int(getattr(args, "pq_max_batches_for_tuning", 0)),
        )
        pq_tuner = PQGradNormTunerMB(
            gnn=str(getattr(args, "gnn", "gin")),
            variant=str(getattr(args, "unbiased_mode", "rade")),
            cfg=pq_cfg,
        )

    best_val = float("-inf")
    best_test = float("-inf")
    patience = int(getattr(args, "patience", 100))
    epochs_no_improve = 0
    best_epoch = -1

    if bool(getattr(args, "save_model", False)):
        save_model(args, model, optimizer, run)

    for epoch in range(int(args.epochs)):
        model.train()

        # Epoch aggregates (for logging)
        epoch_loss_sum = 0.0
        epoch_seed_nodes = 0

        epoch_dropped_sum = 0.0
        epoch_added_sum = 0.0
        epoch_aug_batches = 0

        # Decide if this epoch will run PQ update; if so, decide whether we’ll reuse stored batches
        warmup = int(getattr(args, "pq_warmup_epochs", 0))
        every = int(getattr(args, "pq_update_every", 1))
        do_pq_update = (pq_tuner is not None) and ((epoch + 1) >= warmup) and (((epoch + 1) % every) == 0)

        tuning_batches: List[Dict[str, Any]] = []
        max_store = int(getattr(args, "pq_store_batches", 0))  # optional; if 0, we don’t store

        # Base seed per epoch/run (then per-batch seed derived deterministically)
        base_seed = int(args.seed) + 10_000 * int(run) + int(epoch)

        for b_id, batch in enumerate(loaders["train"]):
            batch = batch.to(device)

            x_b = batch.x
            edge_index_clean_b = batch.edge_index
            y_b = batch.y

            seed_n = int(getattr(batch, "batch_size", x_b.size(0)))
            if seed_n <= 0:
                continue

            # labels for seed nodes only
            y_seed = y_b[:seed_n]
            if y_seed.dim() == 2 and y_seed.size(1) == 1:
                y_seed = y_seed.squeeze(1)
            y_seed = y_seed.view(-1).to(torch.long)

            # -----------------------
            # Per-batch augmentation (mini_batch semantics)
            # -----------------------
            edge_index_input = edge_index_clean_b
            edge_index_keep = None
            edge_index_add = None
            aug_stats = None

            dropmessage_rate = float(getattr(args, "dropmessage_rate", 0.0)) if aug_tech == "dropmessage" else 0.0
            dropnode_rate = float(getattr(args, "dropnode_rate", 0.0)) if aug_tech == "dropnode" else 0.0

            do_aug = (
                (aug_tech == "rade")
                and (mb_rade_mode == "local")
                and (str(getattr(args, "aug_mode", "none")).lower().strip() != "none")
                and (p_eff > 0.0 or q_eff > 0.0)
            )

            if do_aug:
                if bool(getattr(args, "unbiased", False)):
                    # Unbiased RADE: pass keep/add masks into RADE convs (batch-local)
                    if str(getattr(args, "mask_sharing", "shared")).lower().strip() == "shared":
                        _, edge_index_keep, edge_index_add, aug_stats = BatchBernoulliEdgeAugmentor.augment_edge_indices(
                            edge_index_clean=edge_index_clean_b,
                            num_nodes=int(x_b.size(0)),
                            p=float(p_eff),
                            q=float(q_eff),
                            mode=str(getattr(args, "aug_mode", "both")),
                            seed=_batch_seed(base_seed, b_id, 0),
                            device=device,
                        )
                        # edge_index_input stays clean; RADE conv reads keep/add
                        edge_index_input = edge_index_clean_b
                    else:
                        L = int(getattr(args, "local_layers", 1))
                        edge_index_keep = []
                        edge_index_add = []
                        stats_list = []
                        for layer in range(L):
                            _, ei_k, ei_a, st = BatchBernoulliEdgeAugmentor.augment_edge_indices(
                                edge_index_clean=edge_index_clean_b,
                                num_nodes=int(x_b.size(0)),
                                p=float(p_eff),
                                q=float(q_eff),
                                mode=str(getattr(args, "aug_mode", "both")),
                                seed=_batch_seed(base_seed, b_id, layer),
                                device=device,
                            )
                            edge_index_keep.append(ei_k)
                            edge_index_add.append(ei_a)
                            stats_list.append(st)

                        aug_stats = {
                            "dropped_undir_avg": float(sum(s["dropped_undir"] for s in stats_list)) / float(len(stats_list)),
                            "added_undir_avg": float(sum(s["added_undir"] for s in stats_list)) / float(len(stats_list)),
                        }
                        edge_index_input = edge_index_clean_b
                else:
                    # Non-unbiased RADE: use augmented edge_index directly for standard convs
                    edge_index_aug, _, _, aug_stats = BatchBernoulliEdgeAugmentor.augment_edge_indices(
                        edge_index_clean=edge_index_clean_b,
                        num_nodes=int(x_b.size(0)),
                        p=float(p_eff),
                        q=float(q_eff),
                        mode=str(getattr(args, "aug_mode", "both")),
                        seed=_batch_seed(base_seed, b_id, 0),
                        device=device,
                    )
                    edge_index_input = edge_index_aug

            # Accumulate augmentation stats (per batch)
            if aug_stats is not None:
                if "dropped_undir" in aug_stats and "added_undir" in aug_stats:
                    epoch_dropped_sum += float(aug_stats["dropped_undir"])
                    epoch_added_sum += float(aug_stats["added_undir"])
                    epoch_aug_batches += 1
                elif "dropped_undir_avg" in aug_stats and "added_undir_avg" in aug_stats:
                    epoch_dropped_sum += float(aug_stats["dropped_undir_avg"])
                    epoch_added_sum += float(aug_stats["added_undir_avg"])
                    epoch_aug_batches += 1

            # -----------------------
            # Forward + loss on seed nodes only
            # -----------------------
            optimizer.zero_grad(set_to_none=True)

            if aug_tech == "dropmessage":
                logits_b = model(x_b, edge_index_clean_b, dropmessage_rate=float(dropmessage_rate))
            elif aug_tech == "dropnode":
                logits_b = model(x_b, edge_index_clean_b, dropnode_rate=float(dropnode_rate))
            elif bool(getattr(args, "unbiased", False)) and do_aug:
                logits_b = model(
                    x_b,
                    edge_index_clean_b,
                    edge_index_keep=edge_index_keep,
                    edge_index_add=edge_index_add,
                    p=float(p_eff),
                    q=float(q_eff),
                )
            else:
                logits_b = model(x_b, edge_index_input)

            loss_b = criterion(logits_b[:seed_n], y_seed)
            loss_b.backward()
            optimizer.step()

            epoch_loss_sum += float(loss_b.item()) * seed_n
            epoch_seed_nodes += seed_n

            # Optionally store batches for PQ tuning (to avoid a second loader pass)
            if do_pq_update and max_store > 0 and len(tuning_batches) < max_store:
                train_mask_b = torch.zeros((x_b.size(0),), device=x_b.device, dtype=torch.bool)
                train_mask_b[:seed_n] = True
                tuning_batches.append(
                    {
                        "x": x_b.detach(),
                        "edge_index": edge_index_clean_b.detach(),
                        "y": y_b.detach(),
                        "train_mask": train_mask_b,
                    }
                )

        avg_train_loss = epoch_loss_sum / max(epoch_seed_nodes, 1)

        # -----------------------
        # Evaluation (mini_batch)
        # -----------------------
        model.eval()
        fw_eval: Dict[str, Any] = {}

        if aug_tech == "dropmessage":
            fw_eval = {"dropmessage_rate": 0.0}

        elif aug_tech == "dropnode":
            fw_eval = {"dropnode_rate": 0.0}

        elif (
            bool(getattr(args, "unbiased", False))
            and (aug_tech == "rade")
            and (mb_rade_mode == "local")
            and str(getattr(args, "unbiased_mode", "rade")).lower().strip() == "rade-ic"
            and (float(q_eff) > 0.0)
        ):
            # Enable deterministic RADE-IC inference correction inside RADE convs
            fw_eval = {"p": float(p_eff), "q": float(q_eff)}


        # Optional CPU eval (slow, but avoids GPU OOM during eval)
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
                forward_kwargs=fw_eval,
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
                forward_kwargs=fw_eval,
            )

        result = (float(tr), float(va), float(te), float(va_loss), None)
        logger.add_result(run, result[:4])

        improved = (result[1] > best_val)
        if improved:
            best_val = float(result[1])
            best_test = float(result[2])
            best_epoch = int(epoch)
            epochs_no_improve = 0
            if bool(getattr(args, "save_model", False)):
                save_model(args, model, optimizer, run)
        else:
            epochs_no_improve += 1

        # -----------------------
        # PQ-GradNorm update (epoch-wise; RADE only; batch-local semantics)
        # -----------------------
        if do_pq_update:
            epoch_seed = int(args.seed) + 100_000 * int(run) + int(epoch) + int(getattr(args, "pq_seed", 0))

            # If we stored batches, tune on those; otherwise tune by re-iterating the loader (extra cost).
            batches_for_tuning = tuning_batches if (len(tuning_batches) > 0) else loaders["train"]

            p_new, q_new, info = pq_tuner.suggest_pq_for_epoch(
                model=model,
                batches=batches_for_tuning,
                criterion=criterion,
                current_p=float(p_eff),
                current_q=float(q_eff),
                aug_mode=str(getattr(args, "aug_mode", "both")).lower().strip(),
                epoch_seed=int(epoch_seed),
            )

            p_new, q_new = _enforce_aug_mode(p_new, q_new, args.aug_mode)
            p_eff, q_eff = float(p_new), float(q_new)

            print(
                f"[PQ-GradNorm-MB] run={run+1:02d} epoch={epoch+1:03d} "
                f"p_epoch_avg_best={info['p_epoch_avg_best']:.4f} q_epoch_avg_best={info['q_epoch_avg_best']:.6f} "
                f"p_new={info['p_new']:.4f} q_new={info['q_new']:.6f} "
                f"G_data_mean={info['G_data_mean']:.3e} G_reg_best_mean={info['G_reg_best_mean']:.3e} obj_mean={info['obj_mean']:.2e} "
                f"used_batches={int(info['used_batches'])} skipped_batches={int(info['skipped_batches'])}"
            )

            if wandb_run is not None:
                wandb.log(
                    {
                        "pq/p_epoch_avg_best": float(info["p_epoch_avg_best"]),
                        "pq/q_epoch_avg_best": float(info["q_epoch_avg_best"]),
                        "pq/p_new": float(info["p_new"]),
                        "pq/q_new": float(info["q_new"]),
                        "pq/G_data_mean": float(info["G_data_mean"]),
                        "pq/G_reg_best_mean": float(info["G_reg_best_mean"]),
                        "pq/obj_mean": float(info["obj_mean"]),
                        "pq/used_batches": float(info["used_batches"]),
                        "pq/skipped_batches": float(info["skipped_batches"]),
                    },
                    step=(epoch + run * int(args.epochs)),
                )

        # -----------------------
        # wandb logging
        # -----------------------
        if wandb_run is not None and (epoch % int(getattr(args, "wandb_log_every", 1)) == 0):
            step = epoch + run * int(args.epochs)

            log_dict = {
                "aug/tech": str(aug_tech),
                "aug/mb_rade_mode": str(mb_rade_mode),
                "run": int(run + 1),
                "epoch": int(epoch),
                "train/loss_mb": float(avg_train_loss),
                "train/acc": float(result[0]),
                "valid/acc": float(result[1]),
                "test/acc": float(result[2]),
                "valid/loss": float(result[3]),
                "best/valid_acc": float(best_val),
                "best/test_acc_at_best_valid": float(best_test),
            }

            if aug_tech == "rade":
                log_dict["aug/p"] = float(p_eff)
                log_dict["aug/q"] = float(q_eff)
                if epoch_aug_batches > 0:
                    log_dict["aug/dropped_undir_mean_per_batch"] = float(epoch_dropped_sum) / float(epoch_aug_batches)
                    log_dict["aug/added_undir_mean_per_batch"] = float(epoch_added_sum) / float(epoch_aug_batches)

            elif aug_tech == "dropmessage":
                log_dict["aug/dropmessage_rate"] = float(getattr(args, "dropmessage_rate", 0.0))

            elif aug_tech == "dropnode":
                log_dict["aug/dropnode_rate"] = float(getattr(args, "dropnode_rate", 0.0))

            wandb.log(log_dict, step=step)

        # -----------------------
        # Console print
        # -----------------------
        if epoch % int(getattr(args, "display_step", 1)) == 0:
            aug_str = ""
            if aug_tech == "rade":
                if epoch_aug_batches > 0:
                    d_mean = float(epoch_dropped_sum) / float(epoch_aug_batches)
                    a_mean = float(epoch_added_sum) / float(epoch_aug_batches)
                    aug_str = f", Drop/Add(mean per batch): -{d_mean:.2f} +{a_mean:.2f} undir"
                aug_str += f" (p={p_eff:.4f}, q={q_eff:.6f}, mb_mode={mb_rade_mode})"
            elif aug_tech == "dropmessage":
                aug_str = f" (dropmessage={float(getattr(args, 'dropmessage_rate', 0.0)):.3f})"
            elif aug_tech == "dropnode":
                aug_str = f" (dropnode={float(getattr(args, 'dropnode_rate', 0.0)):.3f})"

            print(
                f"Run: {run + 1:02d}, "
                f"Epoch: {epoch:03d}, "
                f"TrainLoss: {avg_train_loss:.4f}, "
                f"Train: {100 * result[0]:.2f}%, "
                f"Valid: {100 * result[1]:.2f}%, "
                f"Test: {100 * result[2]:.2f}%, "
                f"Best Valid: {100 * best_val:.2f}%, "
                f"Best Test: {100 * best_test:.2f}%"
                f"{aug_str}"
            )

        # -----------------------
        # Early stopping
        # -----------------------
        if patience > 0 and epochs_no_improve >= patience:
            stop_epoch = epoch + 1
            best_epoch_disp = best_epoch + 1 if best_epoch >= 0 else -1
            msg = (
                f"[EarlyStop] run={run + 1:02d} stopping at epoch {stop_epoch:03d} "
                f"(no val improvement for {patience} epochs). "
                f"Best val={best_val:.4f} at epoch {best_epoch_disp:03d}, "
                f"test@best={best_test:.4f}."
            )
            print(msg)

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

results = logger.print_statistics()
save_result(args, results)

if wandb_run is not None:
    try:
        wandb_run.summary["final_test_mean"] = float(results.mean().item())
        wandb_run.summary["final_test_std"] = float(results.std().item())
    except Exception:
        pass
    wandb.finish()
