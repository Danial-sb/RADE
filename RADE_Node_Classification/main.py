# main.py
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
import wandb

from logger import Logger, save_model, save_result
from dataset import load_nc_dataset
from data_utils import eval_acc_nc, class_rand_splits, make_random_split_idx
from eval import evaluate
from parse import parse_method, parser_add_main_args
from augmentation import BernoulliEdgeAugmentor

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
        # config=vars(args),  # logs all CLI hyperparameters
    )

    for k, v in dict(wandb.config).items():
        if hasattr(args, k):
            setattr(args, k, v)

    # Log the final resolved args (after sweep overrides)
    wandb.config.update(vars(args), allow_val_change=True)

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
c = int(data.y.max().item()) + 1  # safe with Data-only loader

print(f"dataset {args.dataset} | num nodes {n} | num edges {e} | num feats {d} | num classes {c}")
if wandb_run is not None:
    wandb.config.update(
        {"num_nodes": n, "num_edges": e, "num_feats": d, "num_classes": c},
        allow_val_change=True,
    )


# Cache clean edges on CPU for fast per-epoch augmentation
augmentor = BernoulliEdgeAugmentor.from_edge_index(data.edge_index, num_nodes=int(data.num_nodes))


# -----------------------
# Build split list for runs
# -----------------------
split_idx_lst = []
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

# -----------------------
# Model / loss / metric
# -----------------------
model = parse_method(args, n, c, d, device)

# IMPORTANT for RADE-GCN/RADE-GIN:
# set the clean graph cache ONCE (your model must implement set_graph).
if getattr(args, "unbiased", False):
    if not hasattr(model, "set_graph"):
        raise RuntimeError("args.unbiased=True but model has no set_graph(). Add it to your model class.")
    model.set_graph(data.edge_index, int(data.num_nodes))


criterion = nn.CrossEntropyLoss()
eval_func = eval_acc_nc
logger = Logger(args.runs, args)

print("MODEL:", model)


def _enforce_aug_mode(p_val: float, q_val: float, aug_mode: str) -> Tuple[float, float]:
    aug_mode = str(aug_mode).lower().strip()
    if aug_mode == "drop":
        return float(p_val), 0.0
    if aug_mode == "add":
        return 0.0, float(q_val)
    if aug_mode == "none":
        return 0.0, 0.0
    return float(p_val), float(q_val)


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

    # Create a fresh tuner per run (prevents EMA/state leakage across runs)
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

    if args.save_model:
        save_model(args, model, optimizer, run)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # -----------------------
        # Augment per epoch
        # -----------------------
        aug_stats = None
        edge_index_aug = data.edge_index
        edge_index_keep = None
        edge_index_add = None

        do_aug = (str(args.aug_mode).lower().strip() != "none") and (p_eff > 0.0 or q_eff > 0.0)

        if do_aug:
            if getattr(args, "unbiased", False):
                base_seed = args.seed + 10_000 * run + epoch

                if str(getattr(args, "mask_sharing", "shared")).lower().strip() == "shared":
                    # One keep/add mask shared across all layers
                    _, edge_index_keep, edge_index_add, aug_stats = augmentor.augment_edge_indices(
                        p=p_eff,
                        q=q_eff,
                        mode=args.aug_mode,
                        seed=base_seed,
                        device=device,
                    )
                    edge_index_aug = data.edge_index  # clean is used in RADE forward
                else:
                    # Independent keep/add per layer
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

            else:
                # Baseline augmentation: just use edge_index_aug
                edge_index_aug, aug_stats = augmentor.augment_edge_index(
                    p=p_eff,
                    q=q_eff,
                    mode=args.aug_mode,
                    seed=args.seed + 10_000 * run + epoch,
                    device=device,
                    return_stats=True,
                )

        # -----------------------
        # Forward (baseline vs RADE)
        # -----------------------
        if getattr(args, "unbiased", False) and do_aug:
            logits = model(
                data.x,
                data.edge_index,  # clean reference graph
                edge_index_keep=edge_index_keep,
                edge_index_add=edge_index_add,
                p=p_eff,
                q=q_eff,
            )
        else:
            logits = model(data.x, edge_index_aug)

        loss = criterion(logits[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()

        # -----------------------
        # Evaluation
        #   - RADE: clean inference
        #   - RADE-IC: clean + inference correction (uses p_eff,q_eff)
        # -----------------------
        model.eval()
        with torch.no_grad():
            if getattr(args, "unbiased", False) and str(getattr(args, "unbiased_mode", "rade")) == "rade-ic":
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

        if result[1] > best_val:
            best_val = result[1]
            best_test = result[2]
            if args.save_model:
                save_model(args, model, optimizer, run)

        # -----------------------
        # wandb logging (per-epoch)
        # -----------------------
        if wandb_run is not None and (epoch % int(getattr(args, "wandb_log_every", 1)) == 0):
            step = epoch + run * int(args.epochs)

            log_dict = {
                "run": int(run + 1),
                "epoch": int(epoch),
                "train/loss": float(loss.item()),
                "train/acc": float(result[0]),
                "valid/acc": float(result[1]),
                "test/acc": float(result[2]),
                "valid/loss": float(result[3].item()) if hasattr(result[3], "item") else float(result[3]),
                "aug/p": float(p_eff),
                "aug/q": float(q_eff),
                "best/valid_acc": float(best_val),
                "best/test_acc_at_best_valid": float(best_test),
            }

            if aug_stats is not None:
                # shared-mask stats
                if "dropped_undir" in aug_stats:
                    log_dict["aug/dropped_undir"] = float(aug_stats["dropped_undir"])
                if "added_undir" in aug_stats:
                    log_dict["aug/added_undir"] = float(aug_stats["added_undir"])
                # layerwise-mask averages
                if "dropped_undir_avg" in aug_stats:
                    log_dict["aug/dropped_undir_avg"] = float(aug_stats["dropped_undir_avg"])
                if "added_undir_avg" in aug_stats:
                    log_dict["aug/added_undir_avg"] = float(aug_stats["added_undir_avg"])

            wandb.log(log_dict, step=step)


        if epoch % int(args.display_step) == 0:
            aug_str = ""
            if aug_stats is not None:
                if "dropped_undir" in aug_stats and "added_undir" in aug_stats:
                    dU = aug_stats["dropped_undir"]
                    aU = aug_stats["added_undir"]
                    aug_str = f", Drop/Add: -{dU} +{aU} edges"
                elif "dropped_undir_avg" in aug_stats and "added_undir_avg" in aug_stats:
                    dU = aug_stats["dropped_undir_avg"]
                    aU = aug_stats["added_undir_avg"]
                    aug_str = f", Drop/Add(avg per layer): -{dU:.1f} +{aU:.1f} edges"

            print(
                f"Run: {run + 1:02d}, "
                f"Epoch: {epoch:03d}, "
                f"Loss: {loss:.4f}, "
                f"Train: {100 * result[0]:.2f}%, "
                f"Valid: {100 * result[1]:.2f}%, "
                f"Test: {100 * result[2]:.2f}%, "
                f"Best Valid: {100 * best_val:.2f}%, "
                f"Best Test: {100 * best_test:.2f}%"
                f"{aug_str} "
                f"(p={p_eff:.4f}, q={q_eff:.6f})"
            )

        # -----------------------
        # Epoch-wise GradNorm update of (p,q) for next epoch
        # -----------------------
        if pq_tuner is not None:
            warmup = int(getattr(args, "pq_warmup_epochs", 0))
            every = int(getattr(args, "pq_update_every", 1))
            if (epoch + 1) >= warmup and ((epoch + 1) % every == 0):
                # Deterministic-but-epoch-varying seed for subset selection and tuner RNG isolation
                epoch_seed = int(args.seed) + 100000 * int(run) + int(epoch) + int(getattr(args, "pq_seed", 0))

                # IMPORTANT: tune using training-mode gradients (but BN buffers are frozen inside tuner)
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
                )

                # restore mode
                if not prev_mode_training:
                    model.eval()

                # enforce aug_mode again (robustness)
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

results = logger.print_statistics()

# -----------------------
# Save results
# -----------------------
save_result(args, results)

if wandb_run is not None:
    # log multi-run summary (if runs > 1)
    try:
        wandb_run.summary["final_test_mean"] = float(results.mean().item())
        wandb_run.summary["final_test_std"] = float(results.std().item())
    except Exception:
        pass
    wandb.finish()

