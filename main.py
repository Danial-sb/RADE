# main.py
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import random
import numpy as np
import torch
import torch.nn as nn

from logger import Logger, save_model, save_result
from dataset import load_nc_dataset
from data_utils import eval_acc_nc, class_rand_splits, make_random_split_idx
from eval import evaluate
from parse import parse_method, parser_add_main_args
from augmentation import BernoulliEdgeAugmentor


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

p_eff, q_eff = args.p, args.q
if args.aug_mode == "drop":
    q_eff = 0.0
elif args.aug_mode == "add":
    p_eff = 0.0
elif args.aug_mode == "none":
    p_eff = 0.0
    q_eff = 0.0

# -----------------------
# Training loop
# -----------------------
for run in range(args.runs):
    split_idx = split_idx_lst[run]
    train_idx = split_idx["train"].to(device)

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("-inf")
    best_test = float("-inf")

    if args.save_model:
        save_model(args, model, optimizer, run)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        # -----------------------
        # Augment per epoch
        # -----------------------
        aug_stats = None
        edge_index_aug = data.edge_index
        edge_index_keep = None
        edge_index_add = None

        do_aug = (args.aug_mode != "none") and (p_eff > 0.0 or q_eff > 0.0)
        if do_aug:
            if getattr(args, "unbiased", False):
                # RADE path needs keep/add split
                base_seed = args.seed + 10_000 * run + epoch

                if getattr(args, "rade_scope", "all") == "last":
                    # Only one keep/add mask is needed (last layer only).
                    _, edge_index_keep, edge_index_add, aug_stats = augmentor.augment_edge_indices(
                        p=p_eff,
                        q=q_eff,
                        mode=args.aug_mode,
                        seed=base_seed,
                        device=device,
                    )
                    # edge_index_aug not used for RADE forward; keep it clean
                    edge_index_aug = data.edge_index

                else:
                    # rade_scope == "all"
                    if getattr(args, "mask_sharing", "shared") == "shared":
                        edge_index_aug, edge_index_keep, edge_index_add, aug_stats = augmentor.augment_edge_indices(
                            p=p_eff,
                            q=q_eff,
                            mode=args.aug_mode,
                            seed=base_seed,
                            device=device,
                        )
                    else:
                        # Independent mask per layer
                        edge_index_keep = []
                        edge_index_add = []
                        stats_list = []

                        for layer in range(args.local_layers):
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
                # Baseline augmentation: just use augmented edge_index
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
            # Baseline uses the augmented graph when do_aug=True
            logits = model(data.x, edge_index_aug)

        loss = criterion(logits[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()

        # -----------------------
        # Evaluation (always clean)
        # -----------------------
        result = evaluate(model, data, split_idx, eval_func, criterion, args, device=device)
        logger.add_result(run, result[:4])

        if result[1] > best_val:
            best_val = result[1]
            best_test = result[2]
            if args.save_model:
                save_model(args, model, optimizer, run)

        if epoch % args.display_step == 0:
            aug_str = ""
            if aug_stats is not None:
                # If your stats dict uses different keys, adjust here.
                # For my earlier recommended stats:
                #   dropped_undir / added_undir
                # For the keep/add version above, I used:
                #   num_keep_undir / num_add_undir
                if "dropped_undir" in aug_stats and "added_undir" in aug_stats:
                    dU = aug_stats["dropped_undir"]
                    aU = aug_stats["added_undir"]
                    aug_str = f", Drop/Add: -{dU} +{aU} edges"
                elif "num_keep_undir" in aug_stats and "num_add_undir" in aug_stats:
                    aU = aug_stats["num_add_undir"]
                    aug_str = f", Added: +{aU} edges"

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
                f"{aug_str}"
            )

    logger.print_statistics(run)

results = logger.print_statistics()

# -----------------------
# Save results
# -----------------------
save_result(args, results)
