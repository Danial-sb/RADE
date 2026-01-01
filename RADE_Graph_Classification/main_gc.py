# graph_classification/main_gc.py
from __future__ import annotations

import argparse
import os
import random
from typing import Dict, Tuple
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


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    """
    Keep OGB categorical tensors as-is (Atom/Bond encoders expect LongTensor).
    For TU-style float models (Linear encoder), ensure float.
    """
    if not use_ogb_encoders:
        if batch.x is not None and not torch.is_floating_point(batch.x):
            batch.x = batch.x.float()
        if use_edge_attr and hasattr(batch, "edge_attr") and (batch.edge_attr is not None):
            if not torch.is_floating_point(batch.edge_attr):
                batch.edge_attr = batch.edge_attr.float()


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
    safety_max_additions: int = 2_000_000,
) -> Tuple[object, dict]:
    """
    Apply sampling per graph in the batch.

    If unbiased=False:
      - overwrite data.edge_index with augmented topology (baseline perturbation)
      - overwrite data.edge_attr accordingly (if enabled)

    If unbiased=True:
      - keep data.edge_index CLEAN
      - attach:
          data.edge_index_keep  (directed kept edges)
          data.edge_index_add   (directed added edges)
        so RADE convs can consume masks and be expectation-preserving.
    """
    if PyGBatch is None:
        raise RuntimeError("torch_geometric.data.Batch is required for batch augmentation.")

    mode = str(mode).lower().strip()
    if mode == "none" or ((p <= 0.0) and (q <= 0.0)):
        return batch.to(device), {"dropped_undir": 0, "added_undir": 0, "graphs": int(getattr(batch, "num_graphs", 0))}

    data_list = batch.to_data_list()

    dropped_total = 0
    added_total = 0

    for gi, data in enumerate(data_list):
        # CLEAN tensors (keep as-is for unbiased mode)
        ei_clean = data.edge_index

        ea_clean = None
        if use_edge_attr and hasattr(data, "edge_attr") and (data.edge_attr is not None):
            ea_clean = data.edge_attr

        aug = BernoulliEdgeAugmentor.from_graph(ei_clean, int(data.num_nodes), edge_attr=ea_clean)

        ei_aug, ei_keep, ei_add, ea_aug, ea_keep, ea_add, st = aug.augment_edge_indices_with_attr(
            p=float(p),
            q=float(q),
            mode=mode,
            seed=int(seed) + 97 * gi,
            device=torch.device("cpu"),  # build on CPU; move whole batch at end
            safety_max_additions=int(safety_max_additions),
        )

        if bool(unbiased):
            # keep CLEAN topology for forward; attach masks for RADE convs
            data.edge_index = ei_clean.to(torch.long)  # keep clean
            data.edge_index_keep = ei_keep.to(torch.long)
            data.edge_index_add = ei_add.to(torch.long)

            # keep clean edge_attr aligned with clean edge_index
            if use_edge_attr and ea_clean is not None:
                data.edge_attr = ea_clean
        else:
            # baseline perturbation: overwrite topology (and attrs) to augmented graph
            data.edge_index = ei_aug.to(torch.long)
            if use_edge_attr and (ea_aug is not None):
                data.edge_attr = ea_aug

        dropped_total += int(st.get("dropped_undir", 0))
        added_total += int(st.get("added_undir", 0))

    batch_out = PyGBatch.from_data_list(data_list).to(device)
    stats = {
        "graphs": int(getattr(batch_out, "num_graphs", len(data_list))),
        "dropped_undir": int(dropped_total),
        "added_undir": int(added_total),
        "p": float(p),
        "q": float(q),
        "mode": mode,
        "unbiased": bool(unbiased),
    }
    return batch_out, stats



def _mean(xs) -> float:
    if not xs:
        return 0.0
    return float(sum(xs) / len(xs))

def print_training_banner(args, model) -> None:
    aug_mode = str(getattr(args, "aug_mode", "none")).lower().strip()
    p0 = float(getattr(args, "p", 0.0))
    q0 = float(getattr(args, "q", 0.0))
    aug_on = (aug_mode != "none") and ((p0 > 0.0) or (q0 > 0.0))

    unbiased = bool(getattr(args, "unbiased", False))
    unbiased_mode = str(getattr(args, "unbiased_mode", "rade")).lower().strip() if unbiased else "off"
    pq_on = bool(getattr(args, "pq_gradnorm", False))

    gnn = getattr(getattr(model, "cfg", None), "gnn", None) or getattr(args, "gnn", "unknown")

    print(
        f"[GC] model={model.__class__.__name__} gnn={gnn} | "
        f"aug={'ON' if aug_on else 'OFF'}(mode={aug_mode}, p={p0:.4g}, q={q0:.4g}) | "
        f"unbiased={'ON' if unbiased else 'OFF'}({unbiased_mode}) | "
        f"gradnorm={'ON' if pq_on else 'OFF'}"
    )
    print(model, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser_add_gc_args(parser)
    args = parser.parse_args()

    # device
    if args.cpu or (not torch.cuda.is_available()):
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}")

    # wandb
    use_wandb = bool(args.wandb) and (wandb is not None) and (args.wandb_mode != "disabled")
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
        criterion = nn.BCEWithLogitsLoss(reduction="none")  # training uses masked_bce_with_logits
    else:
        raise ValueError(f"Unsupported loss={loss_type}")

    # augmentation config (safe defaults if parse_gc not updated yet)
    aug_mode = str(getattr(args, "aug_mode", "none")).lower().strip()
    safety_max_additions = int(getattr(args, "safety_max_additions", 2_000_000))

    # p/q initialization (support both new and legacy flag names)
    p_init = float(getattr(args, "p_init", getattr(args, "p", 0.0)))
    q_init = float(getattr(args, "q_init", getattr(args, "q", 0.0)))

    # RADE variant label (used for PQ tuner; defaults to 'rade')
    unbiased_mode = str(getattr(args, "unbiased_mode", "rade")).lower().strip()

    # PQ-GradNorm settings (safe defaults if parse_gc not updated yet)
    pq_enabled = bool(getattr(args, "pq_gradnorm", False))
    pq_warmup = int(getattr(args, "pq_warmup_epochs", 0))
    pq_every = int(getattr(args, "pq_update_every", 1))
    pq_grid = int(getattr(args, "pq_grid_size", 11))
    pq_p_max = float(getattr(args, "pq_p_max", 0.8))
    pq_q_max = float(getattr(args, "pq_q_max", 1e-3))
    pq_ema = float(getattr(args, "pq_ema", 0.9))

    if pq_enabled and (PQGradNormTunerGC is None or PQTunerGCConfig is None):
        raise RuntimeError("pq_gradnorm requested but pq_gradnorm_gc.py is not importable.")

    all_best_tests = []

    for run in range(int(args.runs)):
        fix_seed(int(args.seed) + run)

        model = parse_method_gc(args, in_channels=in_dim, out_channels=out_dim, device=device)
        print_training_banner(args, model)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # model behavior flags
        use_ogb_encoders = bool(getattr(args, "use_ogb_encoders", False))
        use_edge_attr = bool(getattr(args, "use_edge_attr", False))

        # PQ tuner (run-level)
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
                variant=unbiased_mode,   # 'rade' or 'rade-ic'
                pooling=str(args.pooling),
                cfg=pq_cfg,
            )

        # epoch-wise p/q state (used for augmentation THIS epoch; updated for NEXT epoch)
        p_cur = float(p_init)
        q_cur = float(q_init)

        best_valid = -1e18
        best_test = -1e18
        best_epoch = -1

        for epoch in range(1, int(args.epochs) + 1):
            model.train()
            loss_sum = 0.0
            n_graphs = 0

            # enforce aug_mode
            p = float(p_cur)
            q = float(q_cur)
            if aug_mode == "drop":
                q = 0.0
            elif aug_mode == "add":
                p = 0.0
            elif aug_mode == "none":
                p, q = 0.0, 0.0

            dropped_epoch = 0
            added_epoch = 0
            seen_graphs_epoch = 0

            for step, batch in enumerate(loaders["train"]):
                optimizer.zero_grad(set_to_none=True)

                # keep batch on CPU for augmentation; move to device after
                _maybe_cast_batch_for_linear(batch, use_ogb_encoders=use_ogb_encoders, use_edge_attr=use_edge_attr)

                # Apply augmentation (training only)
                if aug_mode != "none" and ((p > 0.0) or (q > 0.0)):
                    seed_step = int(args.seed) + 10_000 * run + 1_000 * epoch + step
                    batch, st = _augment_batch_graphwise(
                        batch,
                        p=p,
                        q=q,
                        mode=aug_mode,
                        seed=seed_step,
                        device=device,
                        use_edge_attr=use_edge_attr,
                        unbiased=bool(args.unbiased),
                        safety_max_additions=safety_max_additions,
                    )
                    dropped_epoch += int(st.get("dropped_undir", 0))
                    added_epoch += int(st.get("added_undir", 0))
                    seen_graphs_epoch += int(st.get("graphs", 0))
                else:
                    batch = batch.to(device)

                _maybe_cast_batch_for_linear(batch, use_ogb_encoders=use_ogb_encoders, use_edge_attr=use_edge_attr)

                edge_attr = batch.edge_attr if (use_edge_attr and hasattr(batch, "edge_attr")) else None

                # If unbiased: pass keep/add masks so RADE path runs (expectation-preserving)
                edge_index_keep = getattr(batch, "edge_index_keep", None)
                edge_index_add = getattr(batch, "edge_index_add", None)

                out = model(
                    batch.x,
                    batch.edge_index,  # CLEAN for unbiased=True; AUG for unbiased=False baseline
                    batch.batch,
                    edge_attr=edge_attr,
                    edge_index_keep=edge_index_keep,
                    edge_index_add=edge_index_add,
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

            train_score, valid_score, test_score, valid_loss = evaluate_splits(
                model,
                loaders,
                args.dataset,
                criterion,
                metric=metric,
                use_edge_attr=use_edge_attr,
                device=device,
                p=float(p),
                q=float(q),
            )

            if valid_score > best_valid:
                best_valid = float(valid_score)
                best_test = float(test_score)
                best_epoch = int(epoch)
                if args.save_model:
                    _save_checkpoint(args, model, optimizer, run=run + 1, epoch=epoch, best_valid=best_valid)

            # -------------------------
            # PQ-GradNorm update (end of epoch): batch-wise solve, epoch average -> (p_cur,q_cur) for NEXT epoch
            # -------------------------
            pq_info = None
            if pq_enabled and (epoch >= pq_warmup) and (pq_every > 0) and ((epoch % pq_every) == 0):
                p_bests, q_bests = [], []
                Gd_list, Gr_list, obj_list = [], [], []

                for b, batch_tune in enumerate(loaders["train"]):
                    batch_tune = batch_tune.to(device)
                    _maybe_cast_batch_for_linear(batch_tune, use_ogb_encoders=use_ogb_encoders, use_edge_attr=use_edge_attr)

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

                # enforce aug_mode
                if aug_mode == "drop":
                    q_next = 0.0
                elif aug_mode == "add":
                    p_next = 0.0
                elif aug_mode == "none":
                    p_next, q_next = 0.0, 0.0

                # clamp
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

            # -------------------------
            # print / wandb
            # -------------------------
            if (epoch % int(args.display_step)) == 0:
                msg = (
                    f"[GC] run={run+1:02d} epoch={epoch:03d} "
                    f"loss={train_loss:.4f} "
                    f"train_{metric}={train_score:.4f} "
                    f"valid_{metric}={valid_score:.4f} "
                    f"test_{metric}={test_score:.4f} "
                    f"best_valid={best_valid:.4f} best_test@best={best_test:.4f} (ep={best_epoch})"
                )
                if aug_mode != "none" and ((p > 0.0) or (q > 0.0)):
                    msg += f" | drop/add(undir): -{dropped_epoch} +{added_epoch} (p={p:.4g}, q={q:.4g})"
                if pq_info is not None:
                    msg += (
                        f"\n[PQ-GradNorm-GC] epoch={epoch:03d} "
                        f"G_data(avg)={pq_info['G_data_avg']:.3e} "
                        f"G_reg_best(avg)={pq_info['G_reg_best_avg']:.3e} "
                        f"p(avgBest->new)={pq_info['p_hat']:.4f}->{pq_info['p_next']:.4f} "
                        f"q(avgBest->new)={pq_info['q_hat']:.6f}->{pq_info['q_next']:.6f} "
                        f"obj(avg)={pq_info['obj_avg']:.3e}"
                    )
                print(msg)

            if use_wandb and (epoch % int(args.wandb_log_every) == 0):
                log_obj = {
                    "epoch": epoch,
                    "run": run,
                    "train/loss": train_loss,
                    "valid/loss": valid_loss,
                    f"train/{metric}": train_score,
                    f"valid/{metric}": valid_score,
                    f"test/{metric}": test_score,
                    f"best/valid_{metric}": best_valid,
                    f"best/test_at_best_valid_{metric}": best_test,
                    "best/epoch": best_epoch,
                    "aug/mode": aug_mode,
                    "aug/p_epoch": float(p),
                    "aug/q_epoch": float(q),
                }
                if aug_mode != "none" and ((p > 0.0) or (q > 0.0)):
                    log_obj.update(
                        {
                            "aug/dropped_undir_epoch": int(dropped_epoch),
                            "aug/added_undir_epoch": int(added_epoch),
                            "aug/graphs_epoch": int(seen_graphs_epoch),
                        }
                    )
                if pq_info is not None:
                    log_obj.update(
                        {
                            "pq/G_data_avg": float(pq_info["G_data_avg"]),
                            "pq/G_reg_best_avg": float(pq_info["G_reg_best_avg"]),
                            "pq/obj_avg": float(pq_info["obj_avg"]),
                            "pq/p_hat": float(pq_info["p_hat"]),
                            "pq/q_hat": float(pq_info["q_hat"]),
                            "pq/p_next": float(pq_info["p_next"]),
                            "pq/q_next": float(pq_info["q_next"]),
                        }
                    )
                wandb.log(log_obj)

        all_best_tests.append(best_test)

    mean_best = float(np.mean(all_best_tests)) if all_best_tests else 0.0
    std_best = float(np.std(all_best_tests)) if all_best_tests else 0.0
    print(f"[GC] done: best_test@best_valid_{metric} = {mean_best:.4f} ± {std_best:.4f} over {args.runs} runs")

    if use_wandb:
        wandb.summary[f"final/mean_best_test_at_best_valid_{metric}"] = mean_best
        wandb.summary[f"final/std_best_test_at_best_valid_{metric}"] = std_best
        wandb.finish()


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
