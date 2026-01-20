# graph_classification/main_gc.py
from __future__ import annotations

import argparse
import os
import random
from typing import Dict, Tuple, Optional, List, Union

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset

try:
    from torch_geometric.data import Batch as PyGBatch
except Exception:
    PyGBatch = None

from parse_gc import parser_add_gc_args, parse_method_gc
from dataset_gc import load_gc_dataset
from eval_gc import evaluate_splits, masked_bce_with_logits
from augmentation_gc import BernoulliEdgeAugmentor
from pq_gradnorm_gc import PQGradNormTunerGC, PQTunerGCConfig
import wandb


EdgeIndexObj = Union[torch.Tensor, List[torch.Tensor]]


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_deterministic_debug_mode("error")


def _make_loaders(
    dataset,
    split_idx: Dict[str, torch.Tensor],
    *,
    batch_size: int,
    num_workers: int,
) -> Dict[str, DataLoader]:
    def subset(name: str) -> Subset:
        idx = split_idx[name].tolist()
        return Subset(dataset, idx)

    return {
        "train": DataLoader(subset("train"), batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "valid": DataLoader(subset("valid"), batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(subset("test"), batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }


def _save_checkpoint(args, model, optimizer, run: int, epoch: int, best_valid: float) -> None:
    out_dir = os.path.join("models_gc", args.dataset)
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
        if use_edge_attr and hasattr(batch, "edge_attr") and (batch.edge_attr is not None):
            if not torch.is_floating_point(batch.edge_attr):
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
    unbiased: bool,
    mask_sharing: str,
    local_layers: int,
    safety_max_additions: int = 2_000_000,
) -> Tuple[object, dict, Optional[EdgeIndexObj], Optional[EdgeIndexObj]]:
    if PyGBatch is None:
        raise RuntimeError("torch_geometric.data.Batch is required for batch augmentation.")

    mode = str(mode).lower().strip()
    mask_sharing = str(mask_sharing).lower().strip()

    if mode == "none" or ((p <= 0.0) and (q <= 0.0)):
        stats = {"graphs": int(getattr(batch, "num_graphs", 0)), "p": float(p), "q": float(q), "mode": mode}
        return batch.to(device), stats, None, None

    data_list = batch.to_data_list()

    dropped_total = 0
    added_total = 0
    graphs_in_batch = int(getattr(batch, "num_graphs", len(data_list)))

    if bool(unbiased) and (mask_sharing == "layerwise"):
        keep_layers: List[List[torch.Tensor]] = [[] for _ in range(int(local_layers))]
        add_layers: List[List[torch.Tensor]] = [[] for _ in range(int(local_layers))]
        dropped_layers = [0 for _ in range(int(local_layers))]
        added_layers = [0 for _ in range(int(local_layers))]
    else:
        keep_list: List[torch.Tensor] = []
        add_list: List[torch.Tensor] = []

    node_offset = 0

    for gi, data in enumerate(data_list):
        ei_clean = data.edge_index

        ea_clean = None
        if use_edge_attr and hasattr(data, "edge_attr") and (data.edge_attr is not None):
            ea_clean = data.edge_attr

        aug = BernoulliEdgeAugmentor.from_graph(ei_clean, int(data.num_nodes), edge_attr=ea_clean)

        if bool(unbiased):
            data.edge_index = ei_clean.to(torch.long)
            if use_edge_attr and ea_clean is not None:
                data.edge_attr = ea_clean

            if mask_sharing == "shared":
                _, ei_keep, ei_add, _, _, _, st = aug.augment_edge_indices_with_attr(
                    p=float(p),
                    q=float(q),
                    mode=mode,
                    seed=int(seed) + 97 * gi,
                    device=torch.device("cpu"),
                    safety_max_additions=int(safety_max_additions),
                )
                ei_keep = ei_keep.to(torch.long)
                ei_add = ei_add.to(torch.long)

                if ei_keep.numel() > 0:
                    keep_list.append(ei_keep + int(node_offset))
                if ei_add.numel() > 0:
                    add_list.append(ei_add + int(node_offset))

                dropped_total += int(st.get("dropped_undir", 0))
                added_total += int(st.get("added_undir", 0))

            else:
                for layer in range(int(local_layers)):
                    _, ei_keep_l, ei_add_l, _, _, _, st_l = aug.augment_edge_indices_with_attr(
                        p=float(p),
                        q=float(q),
                        mode=mode,
                        seed=int(seed) + 97 * gi + 1_000_000 * int(layer),
                        device=torch.device("cpu"),
                        safety_max_additions=int(safety_max_additions),
                    )
                    ei_keep_l = ei_keep_l.to(torch.long)
                    ei_add_l = ei_add_l.to(torch.long)

                    if ei_keep_l.numel() > 0:
                        keep_layers[layer].append(ei_keep_l + int(node_offset))
                    if ei_add_l.numel() > 0:
                        add_layers[layer].append(ei_add_l + int(node_offset))

                    dropped_layers[layer] += int(st_l.get("dropped_undir", 0))
                    added_layers[layer] += int(st_l.get("added_undir", 0))

        else:
            ei_aug, _, _, ea_aug, _, _, st = aug.augment_edge_indices_with_attr(
                p=float(p),
                q=float(q),
                mode=mode,
                seed=int(seed) + 97 * gi,
                device=torch.device("cpu"),
                safety_max_additions=int(safety_max_additions),
            )
            data.edge_index = ei_aug.to(torch.long)
            if use_edge_attr and (ea_aug is not None):
                data.edge_attr = ea_aug

            dropped_total += int(st.get("dropped_undir", 0))
            added_total += int(st.get("added_undir", 0))

        node_offset += int(data.num_nodes)

    batch_out = PyGBatch.from_data_list(data_list).to(device)

    keep_obj: Optional[EdgeIndexObj] = None
    add_obj: Optional[EdgeIndexObj] = None

    if bool(unbiased):
        if mask_sharing == "shared":
            keep_obj = _cat_edge_indices(keep_list).to(device)
            add_obj = _cat_edge_indices(add_list).to(device)
            stats = {
                "graphs": int(graphs_in_batch),
                "dropped_undir": int(dropped_total),
                "added_undir": int(added_total),
                "p": float(p),
                "q": float(q),
                "mode": mode,
                "unbiased": True,
                "mask_sharing": "shared",
            }
        else:
            keep_obj = [_cat_edge_indices(keep_layers[l]).to(device) for l in range(int(local_layers))]
            add_obj = [_cat_edge_indices(add_layers[l]).to(device) for l in range(int(local_layers))]

            dropped_avg = float(sum(dropped_layers)) / float(max(int(local_layers), 1))
            added_avg = float(sum(added_layers)) / float(max(int(local_layers), 1))

            stats = {
                "graphs": int(graphs_in_batch),
                "dropped_undir_avg": float(dropped_avg),
                "added_undir_avg": float(added_avg),
                "p": float(p),
                "q": float(q),
                "mode": mode,
                "unbiased": True,
                "mask_sharing": "layerwise",
            }
    else:
        stats = {
            "graphs": int(graphs_in_batch),
            "dropped_undir": int(dropped_total),
            "added_undir": int(added_total),
            "p": float(p),
            "q": float(q),
            "mode": mode,
            "unbiased": False,
            "mask_sharing": "na",
        }

    return batch_out, stats, keep_obj, add_obj


def _mean(xs) -> float:
    if not xs:
        return 0.0
    return float(sum(xs) / len(xs))


def print_training_banner(args, model) -> None:
    aug_tech = str(getattr(args, "aug_tech", "rade")).lower().strip()
    aug_mode = str(getattr(args, "aug_mode", "none")).lower().strip()
    p0 = float(getattr(args, "p", 0.0))
    q0 = float(getattr(args, "q", 0.0))

    unbiased = bool(getattr(args, "unbiased", False))
    unbiased_mode = str(getattr(args, "unbiased_mode", "rade")).lower().strip() if unbiased else "off"
    pq_on = bool(getattr(args, "pq_gradnorm", False))
    mask_sharing = str(getattr(args, "mask_sharing", "shared")).lower().strip()
    if not (unbiased and aug_tech == "rade" and aug_mode != "none" and ((p0 > 0.0) or (q0 > 0.0))):
        mask_sharing = "na"

    gnn = getattr(getattr(model, "cfg", None), "gnn", None) or getattr(args, "gnn", "unknown")

    extra = ""
    if aug_tech == "dropmessage":
        extra = f" | dropmessage_rate={float(getattr(args, 'dropmessage_rate', 0.0)):.3f}"
    elif aug_tech == "dropnode":
        extra = f" | dropnode_rate={float(getattr(args, 'dropnode_rate', 0.0)):.3f}"

    print(
        f"[GC] tech={aug_tech} model={model.__class__.__name__} gnn={gnn} | "
        f"aug_mode={aug_mode}(p={p0:.4g}, q={q0:.4g}) | "
        f"unbiased={'ON' if unbiased else 'OFF'}({unbiased_mode}) | "
        f"mask={mask_sharing} | "
        f"gradnorm={'ON' if pq_on else 'OFF'}"
        f"{extra}"
    )
    print(model, flush=True)

def count_trainable_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

def main():
    parser = argparse.ArgumentParser()
    parser_add_gc_args(parser)
    args = parser.parse_args()

    # Defensive linear enforcement
    if getattr(args, "linear", False):
        args.dropout = 0.0
        if hasattr(args, "bn"):
            args.bn = False

    # -----------------------
    # Augmentation technique routing (NC style)
    # -----------------------
    aug_tech = str(getattr(args, "aug_tech", "rade")).lower().strip()
    if aug_tech not in {"rade", "dropmessage", "dropnode", "none"}:
        raise ValueError(f"--aug_tech must be one of {{rade, dropmessage, dropnode, none}}. Got {aug_tech}")

    def _force_disable_edge_aug() -> None:
        if str(getattr(args, "aug_mode", "none")).lower().strip() != "none":
            args.aug_mode = "none"
        if hasattr(args, "p"):
            args.p = 0.0
        if hasattr(args, "q"):
            args.q = 0.0

    if aug_tech in {"dropmessage", "dropnode"}:
        if bool(getattr(args, "unbiased", False)):
            raise ValueError(f"--aug_tech={aug_tech} must be used with --unbiased False.")
        if bool(getattr(args, "pq_gradnorm", False)):
            raise ValueError(f"--aug_tech={aug_tech} must be used with --pq_gradnorm False.")
        if bool(getattr(args, "linear", False)):
            raise ValueError(f"--aug_tech={aug_tech} is incompatible with --linear (strict linearity).")
        _force_disable_edge_aug()

    elif aug_tech == "none":
        args.unbiased = False
        args.pq_gradnorm = False
        _force_disable_edge_aug()

    # device
    if args.cpu or (not torch.cuda.is_available()):
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}")

    # wandb
    use_wandb = bool(getattr(args, "wandb", False)) and (wandb is not None) and (args.wandb_mode != "disabled")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_name,
            tags=[t.strip() for t in args.wandb_tags.split(",") if t.strip()],
            mode=args.wandb_mode,
            config=vars(args),
        )

    # dataset + split + meta
    dataset, split_idx, meta = load_gc_dataset(
        root=args.data_dir,
        name=args.dataset,
        seed=args.seed,
        train_prop=args.train_prop,
        valid_prop=args.valid_prop,
    )
    loaders = _make_loaders(dataset, split_idx, batch_size=args.batch_size, num_workers=args.num_workers)

    in_dim = int(meta["in_dim"])
    out_dim = int(meta["out_dim"])
    loss_type = meta["loss"] if args.loss == "auto" else args.loss
    metric = meta["metric"] if args.metric == "auto" else args.metric

    # criterion
    if loss_type == "ce":
        criterion = nn.CrossEntropyLoss()
    elif loss_type == "bce":
        criterion = nn.BCEWithLogitsLoss(reduction="none")
    else:
        raise ValueError(f"Unsupported loss={loss_type}")

    # augmentation config (RADE only)
    aug_mode = str(getattr(args, "aug_mode", "none")).lower().strip()
    safety_max_additions = int(getattr(args, "safety_max_additions", 2_000_000))
    mask_sharing = str(getattr(args, "mask_sharing", "shared")).lower().strip()
    if mask_sharing not in {"shared", "layerwise"}:
        raise ValueError(f"mask_sharing must be 'shared' or 'layerwise'. Got {mask_sharing}")

    p_init = float(getattr(args, "p", 0.0))
    q_init = float(getattr(args, "q", 0.0))
    unbiased_mode = str(getattr(args, "unbiased_mode", "rade")).lower().strip()

    # PQ-GradNorm settings (RADE only)
    pq_enabled = bool(getattr(args, "pq_gradnorm", False))
    if pq_enabled and aug_tech != "rade":
        raise ValueError("--pq_gradnorm is only valid with --aug_tech=rade.")
    pq_warmup = int(getattr(args, "pq_warmup_epochs", 0))
    pq_every = int(getattr(args, "pq_update_every", 1))
    pq_grid = int(getattr(args, "pq_grid_size", 11))
    pq_p_max = float(getattr(args, "pq_p_max", 0.8))
    pq_q_max = float(getattr(args, "pq_q_max", 1e-3))
    pq_ema = float(getattr(args, "pq_ema", 0.9))

    all_best_tests = []

    for run in range(int(args.runs)):
        fix_seed(int(args.seed) + run)

        model = parse_method_gc(args, in_channels=in_dim, out_channels=out_dim, device=device)
        n_params = count_trainable_params(model)
        print(f"[GC] Trainable params: {n_params:,}")
        print_training_banner(args, model)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        use_ogb_encoders = bool(getattr(args, "use_ogb_encoders", False))
        use_edge_attr = bool(getattr(args, "use_edge_attr", False))

        local_layers = int(getattr(getattr(model, "cfg", None), "local_layers", getattr(args, "local_layers", 3)))

        pq_tuner = None
        if pq_enabled:
            pq_cfg = PQTunerGCConfig(
                p_max=pq_p_max,
                q_max=pq_q_max,
                grid_size=pq_grid,
                eps=1e-12,
                seed=int(args.seed),
            )
            pq_tuner = PQGradNormTunerGC(
                gnn=str(args.gnn),
                variant=unbiased_mode,
                pooling=str(args.pooling),
                cfg=pq_cfg,
            )

        p_cur = float(p_init)
        q_cur = float(q_init)

        best_valid = -1e18
        best_test = -1e18
        best_epoch = -1

        patience = int(getattr(args, "patience", 100))
        epochs_no_improve = 0

        for epoch in range(1, int(args.epochs) + 1):
            model.train()
            loss_sum = 0.0
            n_graphs = 0

            # enforce aug_mode (only relevant for RADE)
            p = float(p_cur)
            q = float(q_cur)
            if aug_tech != "rade":
                p, q = 0.0, 0.0
            else:
                if aug_mode == "drop":
                    q = 0.0
                elif aug_mode == "add":
                    p = 0.0
                elif aug_mode == "none":
                    p, q = 0.0, 0.0

            dropped_epoch_acc = 0.0
            added_epoch_acc = 0.0
            seen_graphs_epoch = 0
            stats_are_layer_avg = (bool(getattr(args, "unbiased", False)) and (mask_sharing == "layerwise"))

            dropmessage_rate = float(getattr(args, "dropmessage_rate", 0.0)) if aug_tech == "dropmessage" else 0.0
            dropnode_rate = float(getattr(args, "dropnode_rate", 0.0)) if aug_tech == "dropnode" else 0.0

            for step, batch in enumerate(loaders["train"]):
                optimizer.zero_grad(set_to_none=True)

                _maybe_cast_batch_for_linear(batch, use_ogb_encoders=use_ogb_encoders, use_edge_attr=use_edge_attr)

                edge_index_keep_obj: Optional[EdgeIndexObj] = None
                edge_index_add_obj: Optional[EdgeIndexObj] = None

                do_aug = (aug_tech == "rade") and (aug_mode != "none") and ((p > 0.0) or (q > 0.0))
                if do_aug:
                    seed_step = int(args.seed) + 10_000 * run + 1_000 * epoch + step
                    batch, st, edge_index_keep_obj, edge_index_add_obj = _augment_batch_graphwise(
                        batch,
                        p=p,
                        q=q,
                        mode=aug_mode,
                        seed=seed_step,
                        device=device,
                        use_edge_attr=use_edge_attr,
                        unbiased=bool(getattr(args, "unbiased", False)),
                        mask_sharing=mask_sharing,
                        local_layers=local_layers,
                        safety_max_additions=safety_max_additions,
                    )

                    if "dropped_undir" in st:
                        dropped_epoch_acc += float(st["dropped_undir"])
                        added_epoch_acc += float(st["added_undir"])
                        stats_are_layer_avg = False
                    elif "dropped_undir_avg" in st:
                        dropped_epoch_acc += float(st["dropped_undir_avg"])
                        added_epoch_acc += float(st["added_undir_avg"])
                        stats_are_layer_avg = True

                    seen_graphs_epoch += int(st.get("graphs", 0))
                else:
                    batch = batch.to(device)

                _maybe_cast_batch_for_linear(batch, use_ogb_encoders=use_ogb_encoders, use_edge_attr=use_edge_attr)
                edge_attr = batch.edge_attr if (use_edge_attr and hasattr(batch, "edge_attr")) else None

                edge_index_keep_pass = None
                edge_index_add_pass = None
                if bool(getattr(args, "unbiased", False)) and do_aug:
                    edge_index_keep_pass = edge_index_keep_obj
                    edge_index_add_pass = edge_index_add_obj

                # Forward routing
                if aug_tech == "dropmessage":
                    out = model(
                        batch.x,
                        batch.edge_index,
                        batch.batch,
                        edge_attr=edge_attr,
                        dropmessage_rate=float(dropmessage_rate),
                        p=0.0,
                        q=0.0,
                    )
                elif aug_tech == "dropnode":
                    out = model(
                        batch.x,
                        batch.edge_index,
                        batch.batch,
                        edge_attr=edge_attr,
                        dropnode_rate=float(dropnode_rate),
                        p=0.0,
                        q=0.0,
                    )
                else:
                    out = model(
                        batch.x,
                        batch.edge_index,
                        batch.batch,
                        edge_attr=edge_attr,
                        edge_index_keep=edge_index_keep_pass,
                        edge_index_add=edge_index_add_pass,
                        p=float(p),
                        q=float(q),
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

                bsz = int(batch.num_graphs) if hasattr(batch, "num_graphs") else int(y.size(0))
                loss_sum += float(loss.item()) * bsz
                n_graphs += bsz

            train_loss = loss_sum / max(n_graphs, 1)

            # For eval: keep p/q only for RADE-IC correction when relevant, otherwise 0 is fine.
            p_eval, q_eval = 0.0, 0.0
            if (aug_tech == "rade") and bool(getattr(args, "unbiased", False)) and (unbiased_mode == "rade-ic"):
                p_eval, q_eval = float(p), float(q)

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

            improved = (float(valid_score) > float(best_valid))
            if improved:
                best_valid = float(valid_score)
                best_test = float(test_score)
                best_epoch = int(epoch)
                epochs_no_improve = 0
                if args.save_model:
                    _save_checkpoint(args, model, optimizer, run=run + 1, epoch=epoch, best_valid=best_valid)
            else:
                epochs_no_improve += 1

            # -----------------------
            # print / wandb
            # -----------------------
            pq_info = None  # will be set later (after early stop check) if PQ updates this epoch

            if (epoch % int(args.display_step)) == 0:
                msg = (
                    f"[GC] tech={aug_tech} run={run+1:02d} epoch={epoch:03d} "
                    f"loss={train_loss:.4f} "
                    f"train_{metric}={train_score:.4f} "
                    f"valid_{metric}={valid_score:.4f} "
                    f"test_{metric}={test_score:.4f} "
                    f"best_valid={best_valid:.4f} best_test@best={best_test:.4f} (ep={best_epoch})"
                )

                if aug_tech == "rade" and aug_mode != "none" and ((p > 0.0) or (q > 0.0)):
                    if stats_are_layer_avg:
                        msg += (
                            f" | drop/add(avg per layer, undir): -{dropped_epoch_acc:.1f} +{added_epoch_acc:.1f} "
                            f"(p={p:.4g}, q={q:.4g})"
                        )
                    else:
                        msg += (
                            f" | drop/add(undir): -{int(dropped_epoch_acc)} +{int(added_epoch_acc)} "
                            f"(p={p:.4g}, q={q:.4g})"
                        )
                elif aug_tech == "dropmessage":
                    msg += f" | dropmessage={dropmessage_rate:.3f}"
                elif aug_tech == "dropnode":
                    msg += f" | dropnode={dropnode_rate:.3f}"

                print(msg)

            if use_wandb and (epoch % int(args.wandb_log_every) == 0):
                log_obj = {
                    "epoch": epoch,
                    "run": run,
                    "aug/tech": str(aug_tech),
                    "train/loss": train_loss,
                    "valid/loss": valid_loss,
                    f"train/{metric}": train_score,
                    f"valid/{metric}": valid_score,
                    f"test/{metric}": test_score,
                    f"best/valid_{metric}": best_valid,
                    f"best/test_at_best_valid_{metric}": best_test,
                    "best/epoch": best_epoch,
                }

                if aug_tech == "rade":
                    log_obj.update(
                        {
                            "aug/mode": aug_mode,
                            "aug/p_epoch": float(p),
                            "aug/q_epoch": float(q),
                            "aug/mask_sharing": str(mask_sharing),
                        }
                    )
                    if aug_mode != "none" and ((p > 0.0) or (q > 0.0)):
                        log_obj["aug/graphs_epoch"] = int(seen_graphs_epoch)
                        if stats_are_layer_avg:
                            log_obj["aug/dropped_undir_avg_epoch"] = float(dropped_epoch_acc)
                            log_obj["aug/added_undir_avg_epoch"] = float(added_epoch_acc)
                        else:
                            log_obj["aug/dropped_undir_epoch"] = int(dropped_epoch_acc)
                            log_obj["aug/added_undir_epoch"] = int(added_epoch_acc)

                elif aug_tech == "dropmessage":
                    log_obj["aug/dropmessage_rate"] = float(dropmessage_rate)
                elif aug_tech == "dropnode":
                    log_obj["aug/dropnode_rate"] = float(dropnode_rate)

                wandb.log(log_obj)

            # -----------------------
            # Early stopping trigger (NC-style)
            # -----------------------
            if patience > 0 and epochs_no_improve >= patience:
                stop_epoch = int(epoch)
                best_epoch_disp = int(best_epoch) if best_epoch >= 0 else -1
                msg = (
                    f"[EarlyStop] run={run + 1:02d} stopping at epoch {stop_epoch:03d} "
                    f"(no val improvement for {patience} epochs). "
                    f"Best val={best_valid:.4f} at epoch {best_epoch_disp:03d}, "
                    f"test@best={best_test:.4f}."
                )
                print(msg)

                if use_wandb:
                    wandb.log(
                        {
                            "early_stop/patience": int(patience),
                            "early_stop/stop_epoch": int(stop_epoch),
                            "early_stop/best_epoch": int(best_epoch_disp),
                            "early_stop/best_val": float(best_valid),
                            "early_stop/best_test_at_best_val": float(best_test),
                            "run": int(run),
                        }
                    )
                break

            # -----------------------
            # PQ-GradNorm update (RADE only)  [after early stop check]
            # -----------------------
            if pq_enabled and (epoch >= pq_warmup) and (pq_every > 0) and ((epoch % pq_every) == 0):
                p_bests, q_bests = [], []
                Gd_list, Gr_list, obj_list = [], [], []

                for b, batch_tune in enumerate(loaders["train"]):
                    batch_tune = batch_tune.to(device)
                    _maybe_cast_batch_for_linear(
                        batch_tune, use_ogb_encoders=use_ogb_encoders, use_edge_attr=use_edge_attr
                    )

                    p_b, q_b, info = pq_tuner.suggest_pq_batch(
                        model=model,
                        batch=batch_tune,
                        loss_type=str(loss_type),
                        aug_mode=str(aug_mode),
                        p_max=float(pq_p_max),
                        q_max=float(pq_q_max),
                        seed=int(args.seed) + 100_000 * run + 10_000 * epoch + b,
                    )
                    p_bests.append(float(p_b))
                    q_bests.append(float(q_b))
                    Gd_list.append(float(info["G_data"]))
                    Gr_list.append(float(info["G_reg_best"]))
                    obj_list.append(float(info["obj"]))

                p_hat = _mean(p_bests)
                q_hat = _mean(q_bests)

                p_next = float(pq_ema) * float(p_cur) + (1.0 - float(pq_ema)) * float(p_hat)
                q_next = float(pq_ema) * float(q_cur) + (1.0 - float(pq_ema)) * float(q_hat)

                if aug_mode == "drop":
                    q_next = 0.0
                elif aug_mode == "add":
                    p_next = 0.0
                elif aug_mode == "none":
                    p_next, q_next = 0.0, 0.0

                p_next = float(max(0.0, min(p_next, float(pq_p_max))))
                q_next = float(max(0.0, min(q_next, float(pq_q_max))))

                pq_info = {
                    "G_data_avg": _mean(Gd_list),
                    "G_reg_best_avg": _mean(Gr_list),
                    "obj_avg": _mean(obj_list),
                    "p_hat": float(p_hat),
                    "q_hat": float(q_hat),
                    "p_next": float(p_next),
                    "q_next": float(q_next),
                }

                p_cur, q_cur = p_next, q_next

                # Optional: print PQ summary when display_step hits
                if (epoch % int(args.display_step)) == 0:
                    print(
                        f"[PQ-GradNorm-GC] epoch={epoch:03d} "
                        f"G_data(avg)={pq_info['G_data_avg']:.3e} "
                        f"G_reg_best(avg)={pq_info['G_reg_best_avg']:.3e} "
                        f"p(avgBest->new)={pq_info['p_hat']:.4f}->{pq_info['p_next']:.4f} "
                        f"q(avgBest->new)={pq_info['q_hat']:.6f}->{pq_info['q_next']:.6f} "
                        f"obj(avg)={pq_info['obj_avg']:.3e}"
                    )

                if use_wandb and (epoch % int(args.wandb_log_every) == 0):
                    wandb.log(
                        {
                            "pq/G_data_avg": float(pq_info["G_data_avg"]),
                            "pq/G_reg_best_avg": float(pq_info["G_reg_best_avg"]),
                            "pq/obj_avg": float(pq_info["obj_avg"]),
                            "pq/p_hat": float(pq_info["p_hat"]),
                            "pq/q_hat": float(pq_info["q_hat"]),
                            "pq/p_next": float(pq_info["p_next"]),
                            "pq/q_next": float(pq_info["q_next"]),
                            "run": int(run),
                            "epoch": int(epoch),
                        }
                    )

        all_best_tests.append(best_test)

    mean_best = float(np.mean(all_best_tests)) if all_best_tests else 0.0
    std_best = float(np.std(all_best_tests)) if all_best_tests else 0.0

    mean_best_pct = 100.0 * mean_best
    std_best_pct = 100.0 * std_best

    print(
        f"[GC] done: best_test@best_valid_{metric} = "
        f"{mean_best_pct:.2f} ± {std_best_pct:.2f} (%) over {args.runs} runs"
    )

    if use_wandb:
        # Store percentage in summary (recommended if you want W&B to show % numbers)
        wandb.summary[f"final/mean_best_test_at_best_valid_{metric}"] = mean_best_pct
        wandb.summary[f"final/std_best_test_at_best_valid_{metric}"] = std_best_pct

        # Optional: also store raw (0-1) for later calculations
        wandb.summary[f"final/raw_mean_best_test_at_best_valid_{metric}"] = mean_best
        wandb.summary[f"final/raw_std_best_test_at_best_valid_{metric}"] = std_best

        wandb.finish()


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
