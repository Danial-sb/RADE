# graph_classification/dataset_gc.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import (
    coalesce,
    remove_self_loops,
    to_undirected,
    degree,
)

try:
    from ogb.graphproppred import PygGraphPropPredDataset
except Exception:
    PygGraphPropPredDataset = None


# -------------------------
# Helpers
# -------------------------

def _canon_name(name: str) -> str:
    n = str(name).lower().strip()
    aliases = {
        "mutag": "mutag",
        "proteins": "proteins",
        "imdb-b": "imdb-b",
        "imdb-binary": "imdb-b",
        "imdb-m": "imdb-m",
        "imdb-multi": "imdb-m",
        "imdb-mult": "imdb-m",
        # OGBG names
        "ogbg-molhiv": "ogbg-molhiv",
        "molhiv": "ogbg-molhiv",
        "ogbg-molpcba": "ogbg-molpcba",
        "molpcba": "ogbg-molpcba",
    }
    return aliases.get(n, n)


def _tu_official_name(canon: str) -> str:
    if canon == "mutag":
        return "MUTAG"
    if canon == "proteins":
        return "PROTEINS"
    if canon == "imdb-b":
        return "IMDB-BINARY"
    if canon == "imdb-m":
        return "IMDB-MULTI"
    raise ValueError(f"Not a TU dataset canon name: {canon}")


def _random_split_graphs(
    num_graphs: int,
    train_prop: float,
    valid_prop: float,
    seed: int,
) -> Dict[str, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    perm = torch.randperm(num_graphs, generator=g)

    n_train = int(round(train_prop * num_graphs))
    n_valid = int(round(valid_prop * num_graphs))

    train_idx = perm[:n_train].to(torch.long)
    valid_idx = perm[n_train:n_train + n_valid].to(torch.long)
    test_idx = perm[n_train + n_valid:].to(torch.long)

    return {"train": train_idx, "valid": valid_idx, "test": test_idx}


def _sanitize_graph(
    data: Data,
    *,
    make_undirected_edges: bool,
    remove_loops: bool = True,
) -> Data:
    edge_index = data.edge_index.to(torch.long)
    edge_attr = getattr(data, "edge_attr", None)

    if remove_loops:
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)

    # For molecules, do NOT call to_undirected unless you explicitly want it.
    if make_undirected_edges:
        if edge_attr is None:
            edge_index = to_undirected(edge_index, num_nodes=int(data.num_nodes))
        else:
            edge_index, edge_attr = to_undirected(edge_index, edge_attr, num_nodes=int(data.num_nodes))

    if edge_attr is None:
        edge_index = coalesce(edge_index, num_nodes=int(data.num_nodes))
    else:
        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes=int(data.num_nodes))
        data.edge_attr = edge_attr

    data.edge_index = edge_index
    return data


def _needs_degree_features(dataset: TUDataset) -> bool:
    for i in range(min(len(dataset), 50)):
        if dataset[i].x is None:
            return True
    return False


def _compute_max_degree_tu(dataset: TUDataset, make_undirected_edges: bool) -> int:
    max_deg = 0
    for i in range(len(dataset)):
        d = dataset[i]
        d = _sanitize_graph(d, make_undirected_edges=make_undirected_edges)
        row = d.edge_index[0]
        deg = degree(row, num_nodes=int(d.num_nodes)).to(torch.long)
        max_deg = max(max_deg, int(deg.max().item()) if deg.numel() > 0 else 0)
    return int(max_deg)


@dataclass
class TUPreprocessConfig:
    make_undirected_edges: bool = True
    normalize_features: bool = False
    degree_onehot: bool = True
    degree_cap: Optional[int] = None


class _TUTransform:
    def __init__(self, cfg: TUPreprocessConfig, max_degree: int):
        self.cfg = cfg
        self.max_degree = int(max_degree)
        self.norm = T.NormalizeFeatures() if cfg.normalize_features else None

    def __call__(self, data: Data) -> Data:
        data = _sanitize_graph(data, make_undirected_edges=self.cfg.make_undirected_edges)

        if data.y is not None:
            data.y = data.y.view(-1).to(torch.long)

        if data.x is None:
            row = data.edge_index[0]
            deg = degree(row, num_nodes=int(data.num_nodes)).to(torch.long)

            cap = self.cfg.degree_cap
            if cap is not None:
                deg = torch.clamp(deg, max=int(cap))
                K = int(cap)
            else:
                K = int(self.max_degree)

            if self.cfg.degree_onehot:
                data.x = F.one_hot(deg, num_classes=K + 1).to(torch.float32)
            else:
                data.x = deg.view(-1, 1).to(torch.float32)
        else:
            if not torch.is_floating_point(data.x):
                data.x = data.x.to(torch.float32)

        if self.norm is not None:
            data = self.norm(data)

        return data


# -------------------------
# Public loader
# -------------------------

def load_gc_dataset(
    root: str,
    name: str,
    *,
    seed: int = 0,
    train_prop: float = 0.8,
    valid_prop: float = 0.1,
    tu_cfg: Optional[TUPreprocessConfig] = None,
) -> Tuple[Any, Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Returns:
      dataset: PyG dataset
      split_idx: dict with {'train','valid','test'} graph indices (LongTensor, CPU)
      meta: dict with:
        - in_dim, out_dim, loss, metric, evaluator_name
        - plus dataset/task fields
    """
    canon = _canon_name(name)

    # -------------------------
    # TU datasets (random split)
    # -------------------------
    if canon in {"mutag", "proteins", "imdb-b", "imdb-m"}:
        tu_name = _tu_official_name(canon)
        root_dir = os.path.join(root, "TU")
        ds = TUDataset(root=root_dir, name=tu_name)

        cfg = tu_cfg if tu_cfg is not None else TUPreprocessConfig(
            make_undirected_edges=True,
            normalize_features=False,
            degree_onehot=True,
            degree_cap=None,
        )

        needs_deg = _needs_degree_features(ds)
        max_deg_used = 0
        if needs_deg:
            max_deg = _compute_max_degree_tu(ds, make_undirected_edges=cfg.make_undirected_edges)
            max_deg_used = int(cfg.degree_cap) if cfg.degree_cap is not None else int(max_deg)

        # IMPORTANT: always transform (sanitize edges + ensure float x)
        ds.transform = _TUTransform(cfg=cfg, max_degree=max_deg_used)

        split_idx = _random_split_graphs(len(ds), train_prop=train_prop, valid_prop=valid_prop, seed=seed)

        # infer in_dim after transform
        d0 = ds[int(split_idx["train"][0])]
        in_dim = int(d0.x.size(-1))

        meta = {
            "dataset_family": "tu",
            "dataset_name": tu_name,
            "task_type": "multiclass",
            "metric": "acc",
            "loss": "ce",
            "evaluator_name": None,
            "in_dim": in_dim,
            "out_dim": int(ds.num_classes),
            "num_classes": int(ds.num_classes),
            "y_format": "long_class_index",
        }
        return ds, split_idx, meta

    # -------------------------
    # OGBG datasets (official split)
    # -------------------------
    if canon in {"ogbg-molhiv", "ogbg-molpcba"}:
        if PygGraphPropPredDataset is None:
            raise ImportError("OGB is not available. Install 'ogb' to use ogbg-molhiv/molpcba.")

        root_dir = os.path.join(root, "ogbg")
        ds = PygGraphPropPredDataset(name=canon, root=root_dir)

        def ogb_transform(data: Data) -> Data:
            data = _sanitize_graph(data, make_undirected_edges=False, remove_loops=True)
            if data.y is not None:
                data.y = data.y.to(torch.float32)
            return data

        ds.transform = ogb_transform

        split = ds.get_idx_split()
        split_idx = {k: torch.as_tensor(v, dtype=torch.long) for k, v in split.items()}

        d0 = ds[int(split_idx["train"][0])]
        in_dim = int(d0.x.size(-1))
        out_dim = int(ds.num_tasks)

        if canon == "ogbg-molhiv":
            meta = {
                "dataset_family": "ogbg",
                "dataset_name": canon,
                "task_type": "binary",
                "metric": "rocauc",
                "loss": "bce",
                "evaluator_name": canon,
                "in_dim": in_dim,
                "out_dim": out_dim,  # typically 1
                "num_tasks": out_dim,
                "y_format": "float_with_possible_missing",
            }
        else:
            meta = {
                "dataset_family": "ogbg",
                "dataset_name": canon,
                "task_type": "multilabel",
                "metric": "ap",
                "loss": "bce",
                "evaluator_name": canon,
                "in_dim": in_dim,
                "out_dim": out_dim,  # 128
                "num_tasks": out_dim,
                "y_format": "float_with_nan_or_minus1_missing",
            }

        return ds, split_idx, meta

    raise ValueError(
        f"Unsupported dataset: {name}. "
        f"Supported: mutag, proteins, imdb-b, imdb-m, ogbg-molhiv, ogbg-molpcba."
    )
