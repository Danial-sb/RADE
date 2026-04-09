# dataset.py
import os
from typing import Dict, Tuple

import argparse
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, Flickr
from torch_geometric.utils import coalesce, to_undirected, remove_self_loops

from ogb.nodeproppred import NodePropPredDataset


def _sanitize_edge_index(edge_index: torch.Tensor,
                         num_nodes: int,
                         make_undirected: bool = True,
                         remove_loops: bool = True) -> torch.Tensor:
    edge_index = edge_index.to(torch.long)
    if remove_loops:
        edge_index, _ = remove_self_loops(edge_index)
    if make_undirected:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index = coalesce(edge_index, num_nodes=num_nodes)
    return edge_index


def _mask_to_idx(mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask)[0].to(torch.long)


def random_split_idx(num_nodes: int,
                     train_prop: float = 0.6,
                     valid_prop: float = 0.2,
                     seed: int = 0) -> Dict[str, torch.Tensor]:
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=g)
    n_train = int(round(train_prop * num_nodes))
    n_valid = int(round(valid_prop * num_nodes))
    train_idx = perm[:n_train]
    valid_idx = perm[n_train:n_train + n_valid]
    test_idx = perm[n_train + n_valid:]
    return {"train": train_idx, "valid": valid_idx, "test": test_idx}


def load_nc_dataset(root: str,
                    name: str,
                    normalize_features: bool = True,
                    make_undirected_edges: bool = True,
                    seed: int = 0,
                    train_prop: float = 0.6,
                    valid_prop: float = 0.2) -> Tuple[Data, Dict[str, torch.Tensor]]:
    """
    name in:
      - 'cora', 'citeseer', 'pubmed'
      - 'computer'   (Amazon Computers)
      - 'cs'         (Coauthor CS)
      - 'physics'    (Coauthor Physics)
      - 'flickr'     (PyG Flickr built-in split masks)
      - 'ogbn-arxiv'
    Returns:
      data: PyG Data with x, edge_index, y
      split_idx: dict with train/valid/test indices
    """
    name = name.lower().strip()

    transform = T.NormalizeFeatures() if normalize_features else None

    # -------------------------
    # Planetoid (public split)
    # -------------------------
    if name in {"cora", "citeseer", "pubmed"}:
        ds = Planetoid(root=os.path.join(root, "Planetoid"),
                       name=name.capitalize(),
                       transform=transform)
        data = ds[0]
        data.edge_index = _sanitize_edge_index(
            data.edge_index, data.num_nodes, make_undirected_edges
        )
        data.y = data.y.to(torch.long)

        split_idx = {
            "train": _mask_to_idx(data.train_mask),
            "valid": _mask_to_idx(data.val_mask),
            "test": _mask_to_idx(data.test_mask),
        }
        return data, split_idx

    # -------------------------
    # Flickr
    # -------------------------
    if name == "flickr":
        ds = Flickr(root=os.path.join(root, "Flickr"),
                    transform=transform)
        data = ds[0]

        data.edge_index = _sanitize_edge_index(
            data.edge_index, data.num_nodes, make_undirected_edges
        )

        # Safety: ensure y is shape [N] long
        if data.y.dim() == 2 and data.y.size(1) == 1:
            data.y = data.y.squeeze(1)
        data.y = data.y.to(torch.long)

        split_idx = {
            "train": _mask_to_idx(data.train_mask),
            "valid": _mask_to_idx(data.val_mask),
            "test": _mask_to_idx(data.test_mask),
        }
        return data, split_idx

    # -------------------------
    # Amazon Computers (random split)
    # -------------------------
    if name == "computer":
        ds = Amazon(root=os.path.join(root, "Amazon"),
                    name="Computers",
                    transform=transform)
        data = ds[0]
        data.edge_index = _sanitize_edge_index(
            data.edge_index, data.num_nodes, make_undirected_edges
        )
        data.y = data.y.to(torch.long)

        split_idx = random_split_idx(data.num_nodes, train_prop, valid_prop, seed)
        return data, split_idx

    # -------------------------
    # Coauthor CS / Physics (random split)
    # -------------------------
    if name in {"cs", "physics"}:
        ds = Coauthor(root=os.path.join(root, "Coauthor"),
                      name=("CS" if name == "cs" else "Physics"),
                      transform=transform)
        data = ds[0]
        data.edge_index = _sanitize_edge_index(
            data.edge_index, data.num_nodes, make_undirected_edges
        )
        data.y = data.y.to(torch.long)

        split_idx = random_split_idx(data.num_nodes, train_prop, valid_prop, seed)
        return data, split_idx

    # -------------------------
    # OGBN-Arxiv (official split)
    # -------------------------
    if name == "ogbn-arxiv":
        ogb = NodePropPredDataset(name="ogbn-arxiv", root=os.path.join(root, "ogb"))
        graph = ogb.graph
        edge_index = torch.as_tensor(graph["edge_index"], dtype=torch.long)
        num_nodes = int(graph["num_nodes"])
        x = torch.as_tensor(graph["node_feat"])

        edge_index = _sanitize_edge_index(edge_index, num_nodes, make_undirected_edges)

        y = torch.as_tensor(ogb.labels)
        # Typically [N, 1] class indices
        if y.dim() == 2 and y.size(1) == 1:
            y = y.squeeze(1)
        y = y.to(torch.long)

        data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)

        split = ogb.get_idx_split()
        split_idx = {k: torch.as_tensor(v, dtype=torch.long) for k, v in split.items()}
        return data, split_idx

    raise ValueError(f"Unsupported dataset name: {name}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        default="flickr",
        choices=["cora", "citeseer", "pubmed", "computer", "cs", "physics", "ogbn-arxiv", "flickr"],
        help="Dataset name",
    )
    parser.add_argument("--root", type=str, default="data", help="Dataset root folder")
    parser.add_argument("--seed", type=int, default=0, help="Seed for random splits (computer/cs/physics)")
    args = parser.parse_args()

    data, split_idx = load_nc_dataset(root=args.root, name=args.name, seed=args.seed)

    print("Dataset:", args.name)
    print("Nodes:", data.num_nodes)
    print("Edges:", data.edge_index.size(1))
    print("x shape:", tuple(data.x.shape))
    print("y shape:", tuple(data.y.shape))
    print("Split sizes:",
          {k: int(v.numel()) for k, v in split_idx.items()})

    # show first few labels in train split
    train_idx = split_idx["train"]
    print("First 10 train node ids:", train_idx[:10].tolist())
    print("First 10 train labels:", data.y[train_idx[:10]].tolist())

