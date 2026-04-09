# main.py
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
import wandb

from logger import Logger, save_model, save_result
from dataset import load_nc_dataset
from data_utils import eval_acc_nc, class_rand_splits, make_random_split_idx, eval_f1_nc
from eval import evaluate
from parse import parse_method, parser_add_main_args
from augmentation import BernoulliEdgeAugmentor
from viz_pq import save_pq_plots

from pq_gradnorm import PQGradNormTuner, PQTunerConfig


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


# -----------------------
# Parse args
# -----------------------
parser = argparse.ArgumentParser(description="Training Pipeline for Node Classification")
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
        project=str(getattr(args, "wandb_project", "rade-nc")),
        entity=getattr(args, "wandb_entity", None) or None,
        group=getattr(args, "wandb_group", None) or None,
        name=getattr(args, "wandb_name", None) or None,
        tags=tags,
        mode=str(getattr(args, "wandb_mode", "online")),
    )

    for k, v in dict(wandb.config).items():
        if hasattr(args, k):
            setattr(args, k, v)
    wandb.config.update(vars(args), allow_val_change=True)

    if getattr(args, "linear", False):
        args.dropout = 0.0
        if hasattr(args, "bn"):
            args.bn = False
        if hasattr(args, "ln"):
            args.ln = False

# -----------------------
# Linear-backbone routing
# -----------------------
gnn_raw = str(getattr(args, "gnn", "gcn")).lower().strip()
args.gnn_raw = gnn_raw  # keep for tagging/logging

if gnn_raw == "sgc":
    # SGC = linearized version of GCN stack (no nonlinearity, no aug)
    args.gnn = "gcn"
    args.linear = True
elif gnn_raw == "gin-linear":
    # GIN-Linear = linearized version of GIN stack (no nonlinearity, no aug)
    args.gnn = "gin"
    args.linear = True

# If we routed to a linear backbone, hard-disable augmentation + stochastic layers
if getattr(args, "linear", False) and gnn_raw in {"sgc", "gin-linear"}:
    args.aug_tech = "none"
    args.aug_mode = "none"
    args.unbiased = False
    args.pq_gradnorm = False

    args.dropout = 0.0
    if hasattr(args, "bn"):
        args.bn = False
    if hasattr(args, "ln"):
        args.ln = False


# -----------------------
# Augmentation technique routing
# -----------------------
aug_tech = str(getattr(args, "aug_tech", "rade")).lower().strip()
if aug_tech not in {"rade", "dropmessage", "dropnode", "none"}:
    raise ValueError(f"--aug_tech must be one of {{rade, dropmessage, dropnode, none}}. Got {aug_tech}")

def _require_no_rade_contrib(name: str) -> None:
    if getattr(args, "unbiased", False):
        raise ValueError(f"--aug_tech={name} must be used with --unbiased False.")
    if getattr(args, "pq_gradnorm", False):
        raise ValueError(f"--aug_tech={name} must be used with --pq_gradnorm False.")
    if str(getattr(args, "aug_mode", "none")).lower().strip() != "none":
        raise ValueError(f"--aug_tech={name} requires --aug_mode none (edge augmentation disabled).")
    if getattr(args, "linear", False):
        raise ValueError(f"--aug_tech={name} is incompatible with --linear (strict linearity).")

# DropMessage must NOT use RADE contributions
if aug_tech == "dropmessage":
    _require_no_rade_contrib("dropmessage")

# DropNode must NOT use RADE contributions
if aug_tech == "dropnode":
    _require_no_rade_contrib("dropnode")

if aug_tech == "none":
    args.aug_mode = "none"

print(
    f"[CONFIG] dataset={args.dataset} | backbone_raw={getattr(args,'gnn_raw',args.gnn)} -> backbone={args.gnn} | "
    f"linear={bool(getattr(args,'linear',False))} | aug_tech={args.aug_tech} aug_mode={args.aug_mode} | "
    f"unbiased={bool(getattr(args,'unbiased',False))}({getattr(args,'unbiased_mode','-')}) | "
    f"pq_gradnorm={bool(getattr(args,'pq_gradnorm',False))} | "
    f"p={float(getattr(args,'p',0.0)):.4f} q={float(getattr(args,'q',0.0)):.6f} | "
    f"mask_sharing={getattr(args,'mask_sharing','-')} "
)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")


# -----------------------
# Load data (contract)
# -----------------------
data, base_split_idx = load_nc_dataset(
    args.data_dir,
    args.dataset,
    True,
    True,
    args.seed,
    args.train_prop,
    args.valid_prop,
)

# Basic dataset stats
n = int(data.num_nodes)
e = int(data.edge_index.size(1)) // 2
d = int(data.x.size(1))
c = int(data.y.max().item()) + 1

print(f"dataset {args.dataset} | num nodes {n} | num edges {e} | num feats {d} | num classes {c}")
if wandb_run is not None:
    wandb.config.update(
        {"num_nodes": n, "num_edges": e, "num_feats": d, "num_classes": c},
        allow_val_change=True,
    )


# Cache clean edges on CPU for fast per-epoch augmentation
augmentor = None
if aug_tech == "rade":
    augmentor = BernoulliEdgeAugmentor.from_edge_index(data.edge_index, num_nodes=int(data.num_nodes))

# -----------------------
# Build split list for runs
# -----------------------
split_idx_lst = []
p_histories = []
q_histories = []
for run in range(args.runs):
    if args.rand_split:
        split_idx = make_random_split_idx(n, args.train_prop, args.valid_prop, seed=args.seed + run)
    elif args.rand_split_class:
        split_idx = class_rand_splits(
            data.y, args.label_num_per_class, valid_num=args.valid_num, test_num=args.test_num, seed=args.seed + run
        )
    else:
        split_idx = base_split_idx

    split_idx = {k: v.to(torch.long) for k, v in split_idx.items()}
    split_idx_lst.append(split_idx)

# Move data once
data = data.to(device)

def _enforce_aug_mode(p_val: float, q_val: float, aug_mode: str) -> Tuple[float, float]:
    aug_mode = str(aug_mode).lower().strip()
    if aug_mode == "drop":
        return float(p_val), 0.0
    if aug_mode == "add":
        return 0.0, float(q_val)
    if aug_mode == "none":
        return 0.0, 0.0
    return float(p_val), float(q_val)


def _sync_for_timing(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# -----------------------
# Model / loss / metric
# -----------------------
model = parse_method(args, n, c, d, device)

# IMPORTANT for RADE-GCN/RADE-GIN:
if aug_tech == "rade":
    if not hasattr(model, "set_graph"):
        raise RuntimeError("aug_tech=rade but model has no set_graph(). Add it to your model class.")
    model.set_graph(data.edge_index, int(data.num_nodes))

criterion = nn.CrossEntropyLoss()
eval_func = eval_f1_nc if args.dataset.lower() == "flickr" else eval_acc_nc
logger = Logger(args.runs, args)

print("MODEL:", model)
print("[Timing] enabled for all full-batch runs")


# -----------------------
# Training loop
# -----------------------
for run in range(args.runs):
    split_idx = split_idx_lst[run]
    train_idx = split_idx["train"].to(device)

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # (p,q) used for THIS run
    p_eff, q_eff = _enforce_aug_mode(float(args.p), float(args.q), args.aug_mode)
    # record p,q used at each epoch (values used in THAT epoch's training)
    p_trace = []
    q_trace = []

    # Create a fresh tuner per run
    pq_tuner = None
    if getattr(args, "pq_gradnorm", False):
        if str(args.aug_mode).lower().strip() == "none":
            raise ValueError("--pq_gradnorm requires --aug_mode != none.")
        if not getattr(args, "unbiased", False):
            raise ValueError("--pq_gradnorm currently implemented for RADE/RADE-IC only (use --unbiased True).")

        cfg = PQTunerConfig(
            p_max=float(getattr(args, "p_max", 0.5)),
            q_max=float(getattr(args, "q_max", 1e-3)),
            grid_size=int(getattr(args, "pq_grid_size", 5)),
            ema=float(getattr(args, "pq_ema", 0.9)),
            eps=float(getattr(args, "pq_eps", 1e-12)),
            subset_nodes=int(getattr(args, "pq_subset_nodes", 1024)),
            seed=int(getattr(args, "pq_seed", 0)),
            data_anchor=str(getattr(args, "pq_data_anchor", "clean")).lower().strip()
        )
        pq_tuner = PQGradNormTuner(
            edge_index_clean=data.edge_index,
            num_nodes=int(data.num_nodes),
            gnn=str(args.gnn),
            variant=str(args.unbiased_mode),
            cfg=cfg,
        )

    best_val = float("-inf")
    best_test = float("-inf")
    patience = int(getattr(args, "patience", 100))
    epochs_no_improve = 0
    best_epoch = -1

    if args.save_model:
        save_model(args, model, optimizer, run)

    for epoch in range(args.epochs):
        _sync_for_timing(device)
        epoch_time = time.perf_counter()

        model.train()
        optimizer.zero_grad(set_to_none=True)
        p_trace.append(float(p_eff))
        q_trace.append(float(q_eff))

        # -----------------------
        # Augment per epoch
        # -----------------------
        aug_stats = None
        edge_index_aug = data.edge_index
        edge_index_keep = None
        edge_index_add = None

        dropmessage_rate = float(getattr(args, "dropmessage_rate", 0.0)) if aug_tech == "dropmessage" else 0.0
        dropnode_rate = float(getattr(args, "dropnode_rate", 0.0)) if aug_tech == "dropnode" else 0.0

        do_aug = (
            aug_tech == "rade"
            and (str(args.aug_mode).lower().strip() != "none")
            and (p_eff > 0.0 or q_eff > 0.0)
        )

        if do_aug:
            base_seed = args.seed + 10_000 * run + epoch

            if str(getattr(args, "mask_sharing", "shared")).lower().strip() == "shared":
                _, edge_index_keep, edge_index_add, aug_stats = augmentor.augment_edge_indices(
                    p=p_eff,
                    q=q_eff,
                    mode=args.aug_mode,
                    seed=base_seed,
                    device=device,
                )
                edge_index_aug = data.edge_index
            else:
                edge_index_keep = []
                edge_index_add = []
                stats_list = []

                for layer in range(int(args.local_layers)):
                    _, ei_keep_l, ei_add_l, st = augmentor.augment_edge_indices(
                        p=p_eff,
                        q=q_eff,
                        mode=args.aug_mode,
                        seed=base_seed + 1_000_000 * layer,
                        device=device,
                    )
                    edge_index_keep.append(ei_keep_l)
                    edge_index_add.append(ei_add_l)
                    stats_list.append(st)

                aug_stats = {
                    "dropped_undir_avg": float(sum(s["dropped_undir"] for s in stats_list)) / float(len(stats_list)),
                    "added_undir_avg": float(sum(s["added_undir"] for s in stats_list)) / float(len(stats_list)),
                }
                edge_index_aug = data.edge_index

        # -----------------------
        # Forward
        # -----------------------
        if aug_tech == "dropmessage":
            logits = model(data.x, data.edge_index, dropmessage_rate=dropmessage_rate)
        elif aug_tech == "dropnode":
            logits = model(data.x, data.edge_index, dropnode_rate=dropnode_rate)
        elif aug_tech == "rade" and do_aug:
            logits = model(
                data.x,
                data.edge_index,
                edge_index_keep=edge_index_keep,
                edge_index_add=edge_index_add,
                p=p_eff,
                q=q_eff,
            )
        else:
            logits = model(data.x, data.edge_index)

        loss = criterion(logits[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()

        # -----------------------
        # Evaluation
        # -----------------------
        model.eval()
        with torch.no_grad():
            if aug_tech == "dropmessage":
                logits_eval = model(data.x, data.edge_index, dropmessage_rate=0.0)
            elif aug_tech == "dropnode":
                logits_eval = model(data.x, data.edge_index, dropnode_rate=0.0)
            elif getattr(args, "unbiased", False) and str(getattr(args, "unbiased_mode", "rade")) == "rade-ic":
                logits_eval = model(data.x, data.edge_index, p=p_eff, q=q_eff)
            else:
                logits_eval = model(data.x, data.edge_index)

        result = evaluate(
            model,
            data,
            split_idx,
            eval_func,
            criterion,
            args=args,
            result=logits_eval,
            device=device,
        )
        logger.add_result(run, result[:4])
        _sync_for_timing(device)
        epoch_time = time.perf_counter() - epoch_time
        logger.add_runtime(run, epoch_time)

        improved = (result[1] > best_val)
        if improved:
            best_val = float(result[1])
            best_test = float(result[2])
            best_epoch = int(epoch)
            epochs_no_improve = 0
            if args.save_model:
                save_model(args, model, optimizer, run)
        else:
            epochs_no_improve += 1

        # -----------------------
        # wandb logging
        # -----------------------
        if wandb_run is not None and (epoch % int(getattr(args, "wandb_log_every", 1)) == 0):
            step = epoch + run * int(args.epochs)

            log_dict = {
                "aug/tech": str(aug_tech),
                "run": int(run + 1),
                "epoch": int(epoch),
                "train/loss": float(loss.item()),
                "train/acc": float(result[0]),
                "valid/acc": float(result[1]),
                "test/acc": float(result[2]),
                "valid/loss": float(result[3].item()) if hasattr(result[3], "item") else float(result[3]),
                "best/valid_acc": float(best_val),
                "best/test_acc_at_best_valid": float(best_test),
            }

            log_dict["runtime/epoch_sec"] = float(epoch_time)

            if aug_tech == "rade":
                log_dict["aug/p"] = float(p_eff)
                log_dict["aug/q"] = float(q_eff)

                if aug_stats is not None:
                    if "dropped_undir" in aug_stats:
                        log_dict["aug/dropped_undir"] = float(aug_stats["dropped_undir"])
                    if "added_undir" in aug_stats:
                        log_dict["aug/added_undir"] = float(aug_stats["added_undir"])
                    if "dropped_undir_avg" in aug_stats:
                        log_dict["aug/dropped_undir_avg"] = float(aug_stats["dropped_undir_avg"])
                    if "added_undir_avg" in aug_stats:
                        log_dict["aug/added_undir_avg"] = float(aug_stats["added_undir_avg"])

            elif aug_tech == "dropmessage":
                log_dict["aug/dropmessage_rate"] = float(dropmessage_rate)

            elif aug_tech == "dropnode":
                log_dict["aug/dropnode_rate"] = float(dropnode_rate)

            wandb.log(log_dict, step=step)

        # -----------------------
        # Console print
        # -----------------------
        if epoch % int(args.display_step) == 0:
            aug_str = ""

            if aug_tech == "rade":
                if aug_stats is not None:
                    if "dropped_undir" in aug_stats and "added_undir" in aug_stats:
                        dU = aug_stats["dropped_undir"]
                        aU = aug_stats["added_undir"]
                        aug_str = f", Drop/Add: -{dU} +{aU} edges"
                    elif "dropped_undir_avg" in aug_stats and "added_undir_avg" in aug_stats:
                        dU = aug_stats["dropped_undir_avg"]
                        aU = aug_stats["added_undir_avg"]
                        aug_str = f", Drop/Add(avg per layer): -{dU:.1f} +{aU:.1f} edges"
                aug_str += f" (p={p_eff:.4f}, q={q_eff:.6f})"

            elif aug_tech == "dropmessage":
                aug_str = f" (dropmessage={dropmessage_rate:.3f})"

            elif aug_tech == "dropnode":
                aug_str = f" (dropnode={dropnode_rate:.3f})"

            aug_str += f", Epoch Time: {epoch_time:.4f}s"

            print(
                f"Run: {run + 1:02d}, "
                f"Epoch: {epoch:03d}, "
                f"Loss: {loss:.4f}, "
                f"Train: {100 * result[0]:.2f}%, "
                f"Valid: {100 * result[1]:.2f}%, "
                f"Test: {100 * result[2]:.2f}%, "
                f"Best Valid: {100 * best_val:.2f}%, "
                f"Best Test: {100 * best_test:.2f}%"
                f"{aug_str}"
            )

        # -----------------------
        # Early stopping trigger
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

        # -----------------------
        # PQ-GradNorm update (RADE only)
        # -----------------------
        if pq_tuner is not None:
            warmup = int(getattr(args, "pq_warmup_epochs", 0))
            every = int(getattr(args, "pq_update_every", 1))
            if (epoch + 1) >= warmup and ((epoch + 1) % every == 0):
                epoch_seed = int(args.seed) + 100000 * int(run) + int(epoch) + int(getattr(args, "pq_seed", 0))

                prev_mode_training = model.training
                model.train()

                p_new, q_new, info = pq_tuner.suggest_pq(
                    model=model,
                    data=data,
                    train_idx=train_idx,
                    criterion=criterion,
                    current_p=p_eff,
                    current_q=q_eff,
                    aug_mode=str(args.aug_mode).lower().strip(),
                    epoch_seed=epoch_seed,
                    mask_sharing=str(getattr(args, "mask_sharing", "shared")).lower().strip(),
                    num_layers=int(getattr(args, "local_layers", 1)),
                )


                if not prev_mode_training:
                    model.eval()

                p_new, q_new = _enforce_aug_mode(p_new, q_new, args.aug_mode)
                p_eff, q_eff = p_new, q_new

                print(
                    f"[PQ-GradNorm] run={run+1:02d} epoch={epoch+1:03d} "
                    f"G_data={info['G_data']:.3e} G_reg_best={info['G_reg_best']:.3e} "
                    f"p(best->new)={info['p_best']:.4f}->{info['p_new']:.4f} "
                    f"q(best->new)={info['q_best']:.6f}->{info['q_new']:.6f} obj={info['obj']:.2e}"
                )

                if wandb_run is not None:
                    wandb.log(
                        {
                            "pq/G_data": float(info["G_data"]),
                            "pq/G_reg_best": float(info["G_reg_best"]),
                            "pq/obj": float(info["obj"]),
                            "pq/p_best": float(info["p_best"]),
                            "pq/q_best": float(info["q_best"]),
                            "pq/p_new": float(info["p_new"]),
                            "pq/q_new": float(info["q_new"]),
                        },
                        step=(epoch + run * int(args.epochs)),
                    )

    logger.print_statistics(run)
    p_histories.append(p_trace)
    q_histories.append(q_trace)


results = logger.print_statistics()
runtime_summary = logger.get_runtime_summary()
# Save GradNorm p/q schedules (mean±std over runs) when pq_gradnorm is enabled
if getattr(args, "pq_gradnorm", False) and str(getattr(args, "aug_tech", "rade")).lower().strip() == "rade":
    dataset_tag = str(args.dataset).lower().strip()
    gnn_tag = str(getattr(args, "gnn_raw", getattr(args, "gnn", "gcn"))).lower().strip()
    mode_tag = str(getattr(args, "unbiased_mode", "rade")).lower().strip()  # "rade" or "rade-ic"
    mode_tag = "radeic" if mode_tag == "rade-ic" else mode_tag              # filename-friendly

    tag = f"{dataset_tag}_{gnn_tag}_{mode_tag}"  # e.g., "cora_gcn_rade" or "cora_gcn_radeic"

    save_pq_plots(
        p_histories,
        q_histories,
        tag=tag,
        out_dir="visualization",
        show_runs=True,
    )


save_result(args, results, runtime_summary=runtime_summary)

if wandb_run is not None:
    try:
        wandb_run.summary["final_test_mean"] = float(results.mean().item())
        wandb_run.summary["final_test_std"] = float(results.std().item())
        if runtime_summary is not None:
            wandb_run.summary["runtime/profile"] = "all_full_batch_runs"
            wandb_run.summary["runtime/epoch_sec_run_mean"] = float(runtime_summary["run_mean"])
            wandb_run.summary["runtime/epoch_sec_run_std"] = float(runtime_summary["run_std"])
            wandb_run.summary["runtime/epoch_sec_pooled_mean"] = float(runtime_summary["pooled_mean"])
    except Exception:
        pass
    wandb.finish()
