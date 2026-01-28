# import argparse
# import itertools
# from typing import List, Tuple
#
# import torch
# import torch.nn as nn
# from models import MPNNs
#
# from rade_convs import GraphCache, RADEGCNConv, RADEGINConv
# # Maybe do it only on real datasets.
#
# # ----------------------------
# # Helpers: undirected <-> directed
# # ----------------------------
# def undir_to_dir(edges_undir: List[Tuple[int, int]]) -> torch.Tensor:
#     """Expand undirected edges into both directed directions."""
#     row, col = [], []
#     for u, v in edges_undir:
#         row += [u, v]
#         col += [v, u]
#     if len(row) == 0:
#         return torch.empty((2, 0), dtype=torch.long)
#     return torch.tensor([row, col], dtype=torch.long)
#
#
# def edge_list_undir_complete(n: int) -> List[Tuple[int, int]]:
#     return [(i, j) for i in range(n) for j in range(i + 1, n)]
#
#
# def complement_undir(n: int, edges_undir: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
#     E = set(edges_undir)
#     return [e for e in edge_list_undir_complete(n) if e not in E]
#
#
# # ----------------------------
# # Exact expectation on toy graph by enumeration
# # ----------------------------
# def exact_expectation_toy(
#     gnn: str,
#     p: float,
#     q: float,
#     correct_self_loop: bool,
#     device: str = "cpu",
# ) -> None:
#     """
#     Enumerate ALL keep/add masks on a tiny graph and compute exact expectations:
#       E[biased output] vs clean
#       E[unbiased output] vs clean
#     """
#
#     # Toy graph: 4-node chain 0-1-2-3 (undirected)
#     n = 4
#     edges_undir = [(0, 1), (1, 2), (2, 3)]
#     nonedges_undir = complement_undir(n, edges_undir)
#
#     edge_index_clean = undir_to_dir(edges_undir).to(device)
#
#     # Fixed features (deterministic)
#     torch.manual_seed(0)
#     x = torch.randn(n, 5, device=device)
#
#     # Build cache from CLEAN graph (no self-loops inside cache)
#     cache = GraphCache.from_edge_index(edge_index_clean, num_nodes=n)
#
#     # Build three modules with identical parameters:
#     #   clean: uses clean graph
#     #   biased: uses augmented graph directly (no RADE corrections)
#     #   unbiased: uses RADE keep/add corrections
#     if gnn == "gin":
#         mlp = nn.Identity()
#         conv_clean = RADEGINConv(mlp, eps=0.0).to(device)
#         conv_biased = RADEGINConv(mlp, eps=0.0).to(device)
#         conv_unbiased = RADEGINConv(mlp, eps=0.0).to(device)
#
#         conv_clean.set_graph(cache)
#         conv_biased.set_graph(cache)
#         conv_unbiased.set_graph(cache)
#
#     elif gnn == "gcn":
#         in_dim = x.size(1)
#         out_dim = x.size(1)
#
#         conv_clean = RADEGCNConv(in_dim, out_dim, bias=False, correct_self_loop=correct_self_loop).to(device)
#         conv_biased = RADEGCNConv(in_dim, out_dim, bias=False, correct_self_loop=correct_self_loop).to(device)
#         conv_unbiased = RADEGCNConv(in_dim, out_dim, bias=False, correct_self_loop=correct_self_loop).to(device)
#
#         conv_clean.set_graph(cache)
#         conv_biased.set_graph(cache)
#         conv_unbiased.set_graph(cache)
#
#         # Make linear weights identical and deterministic
#         torch.manual_seed(0)
#         W = torch.randn(out_dim, in_dim, device=device)
#         with torch.no_grad():
#             conv_clean.lin.weight.copy_(W)
#             conv_biased.lin.weight.copy_(W)
#             conv_unbiased.lin.weight.copy_(W)
#
#     else:
#         raise ValueError("gnn must be 'gin' or 'gcn'")
#
#     # Clean output (deterministic)
#     with torch.no_grad():
#         y_clean = conv_clean(x, edge_index_clean, edge_index_keep=None)
#
#     # Enumerate all keep-subsets and add-subsets on undirected edges
#     m = len(edges_undir)
#     M = len(nonedges_undir)
#
#     y_biased_exp = torch.zeros_like(y_clean)
#     y_unbiased_exp = torch.zeros_like(y_clean)
#
#     # keep mask over m edges; add mask over M non-edges
#     for keep_bits in range(1 << m):
#         keep_set = [edges_undir[i] for i in range(m) if (keep_bits >> i) & 1]
#         num_keep = len(keep_set)
#         num_drop = m - num_keep
#         prob_keep = (1.0 - p) ** num_keep * (p ** num_drop)
#
#         edge_index_keep = undir_to_dir(keep_set).to(device)
#
#         for add_bits in range(1 << M):
#             add_set = [nonedges_undir[i] for i in range(M) if (add_bits >> i) & 1]
#             num_add = len(add_set)
#             num_noadd = M - num_add
#             prob_add = (q ** num_add) * ((1.0 - q) ** num_noadd)
#
#             prob = prob_keep * prob_add
#             if prob == 0.0:
#                 continue
#
#             edge_index_add = undir_to_dir(add_set).to(device)
#             edge_index_aug = torch.cat([edge_index_keep, edge_index_add], dim=1)
#
#             with torch.no_grad():
#                 # Biased baseline: run on augmented graph directly
#                 y_biased = conv_biased(x, edge_index_aug, edge_index_keep=None)
#
#                 # Unbiased RADE: run RADE correction on keep/add
#                 y_unbiased = conv_unbiased(
#                     x,
#                     edge_index_clean,
#                     edge_index_keep=edge_index_keep,
#                     edge_index_add=edge_index_add,
#                     p=p,
#                     q=q,
#                 )
#
#             y_biased_exp += prob * y_biased
#             y_unbiased_exp += prob * y_unbiased
#
#     # Report shifts
#     def report(name: str, y_exp: torch.Tensor) -> None:
#         diff = y_exp - y_clean
#         print(f"{name}:")
#         print(f"  max|E[y]-y_clean| = {diff.abs().max().item():.6e}")
#         print(f"  mean|E[y]-y_clean| = {diff.abs().mean().item():.6e}")
#         print(f"  ||E[y]-y_clean||_F = {diff.norm().item():.6e}")
#
#     print("\n=== TOY (EXACT EXPECTATION) ===")
#     print(f"gnn={gnn} | p={p} | q={q} | correct_self_loop={correct_self_loop}")
#     report("Biased (aug graph directly)", y_biased_exp)
#     report("Unbiased (RADE keep/add)", y_unbiased_exp)
#     print("Clean is the reference.\n")
#
#
# # ----------------------------
# # Monte Carlo expectation on real dataset
# # ----------------------------
# def mc_expectation_dataset(
#     dataset: str,
#     gnn: str,
#     p: float,
#     q: float,
#     num_samples: int,
#     subset_nodes: int,
#     correct_self_loop: bool,
#     device: str,
# ) -> None:
#     """
#     Monte Carlo estimate on a real dataset:
#       E[biased output] and E[unbiased output] vs clean
#     Uses BernoulliEdgeAugmentor.
#     """
#     from dataset import load_nc_dataset
#     from augmentation import BernoulliEdgeAugmentor
#
#     # Load data (same contract as main.py)
#     data, _ = load_nc_dataset(
#         root="data",
#         name=dataset,
#         normalize_features=True,
#         make_undirected_edges=True,
#         seed=0,
#         train_prop=0.6,
#         valid_prop=0.2,
#     )
#
#     data = data.to(device)
#     n = int(data.num_nodes)
#
#     # Subset nodes for speed (optional but recommended for MC)
#     if subset_nodes > 0 and subset_nodes < n:
#         torch.manual_seed(0)
#         idx = torch.randperm(n, device=device)[:subset_nodes]
#         idx = idx.sort().values
#     else:
#         idx = torch.arange(n, device=device)
#
#     x = data.x
#     edge_index_clean = data.edge_index
#
#     # Cache clean edges on CPU/whatever device you want (augmentor can return tensors on device)
#     augmentor = BernoulliEdgeAugmentor.from_edge_index(edge_index_clean, num_nodes=int(data.num_nodes))
#
#     # Build cache for RADE convs
#     cache = GraphCache.from_edge_index(edge_index_clean, num_nodes=n)
#
#     # Build modules (shared weights)
#     if gnn == "gin":
#         mlp = nn.Identity()
#         conv_clean = RADEGINConv(mlp, eps=0.0).to(device)
#         conv_biased = RADEGINConv(mlp, eps=0.0).to(device)
#         conv_unbiased = RADEGINConv(mlp, eps=0.0).to(device)
#
#         conv_clean.set_graph(cache)
#         conv_biased.set_graph(cache)
#         conv_unbiased.set_graph(cache)
#
#     elif gnn == "gcn":
#         in_dim = x.size(1)
#         out_dim = x.size(1)
#
#         conv_clean = RADEGCNConv(in_dim, out_dim, bias=False, correct_self_loop=correct_self_loop).to(device)
#         conv_biased = RADEGCNConv(in_dim, out_dim, bias=False, correct_self_loop=correct_self_loop).to(device)
#         conv_unbiased = RADEGCNConv(in_dim, out_dim, bias=False, correct_self_loop=correct_self_loop).to(device)
#
#         conv_clean.set_graph(cache)
#         conv_biased.set_graph(cache)
#         conv_unbiased.set_graph(cache)
#
#         torch.manual_seed(0)
#         W = torch.randn(out_dim, in_dim, device=device)
#         with torch.no_grad():
#             conv_clean.lin.weight.copy_(W)
#             conv_biased.lin.weight.copy_(W)
#             conv_unbiased.lin.weight.copy_(W)
#
#     else:
#         raise ValueError("gnn must be 'gin' or 'gcn'")
#
#     with torch.no_grad():
#         y_clean = conv_clean(x, edge_index_clean, edge_index_keep=None)[idx]
#
#     y_biased_sum = torch.zeros_like(y_clean)
#     y_unbiased_sum = torch.zeros_like(y_clean)
#
#     for s in range(num_samples):
#         seed = 12345 + s
#
#         # We need keep/add for RADE, and edge_index_aug for biased
#         edge_index_aug, edge_index_keep, edge_index_add, _ = augmentor.augment_edge_indices(
#             p=p, q=q, mode="both", seed=seed, device=device
#         )
#
#         with torch.no_grad():
#             y_biased = conv_biased(x, edge_index_aug, edge_index_keep=None)[idx]
#             y_unbiased = conv_unbiased(
#                 x,
#                 edge_index_clean,
#                 edge_index_keep=edge_index_keep,
#                 edge_index_add=edge_index_add,
#                 p=p,
#                 q=q,
#             )[idx]
#
#         y_biased_sum += y_biased
#         y_unbiased_sum += y_unbiased
#
#     y_biased_exp = y_biased_sum / float(num_samples)
#     y_unbiased_exp = y_unbiased_sum / float(num_samples)
#
#     def report(name: str, y_exp: torch.Tensor) -> None:
#         diff = y_exp - y_clean
#         print(f"{name}:")
#         print(f"  max|E[y]-y_clean| = {diff.abs().max().item():.6e}")
#         print(f"  mean|E[y]-y_clean| = {diff.abs().mean().item():.6e}")
#         print(f"  ||E[y]-y_clean||_F = {diff.norm().item():.6e}")
#
#     print("\n=== DATASET (MONTE CARLO EXPECTATION) ===")
#     print(f"dataset={dataset} | gnn={gnn} | p={p} | q={q} | samples={num_samples} | subset_nodes={idx.numel()}")
#     print(f"correct_self_loop={correct_self_loop} | device={device}")
#     report("Biased (aug graph directly)", y_biased_exp)
#     report("Unbiased (RADE keep/add)", y_unbiased_exp)
#     print("Clean is the reference.\n")
#
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--mode", type=str, default="toy", choices=["toy", "dataset"])
#     parser.add_argument("--gnn", type=str, default="gin", choices=["gin", "gcn"])
#     parser.add_argument("--p", type=float, default=0.2)
#     parser.add_argument("--q", type=float, default=0.2)
#     parser.add_argument("--correct_self_loop", type=bool, default=True)
#
#     # dataset mode args
#     parser.add_argument("--dataset", type=str, default="cora")
#     parser.add_argument("--samples", type=int, default=500)
#     parser.add_argument("--subset_nodes", type=int, default=512)
#     parser.add_argument("--device", type=str, default="cpu")
#
#     args = parser.parse_args()
#
#     if args.mode == "toy":
#         exact_expectation_toy(
#             gnn=args.gnn,
#             p=args.p,
#             q=args.q,
#             correct_self_loop=args.correct_self_loop,
#             device=args.device,
#         )
#     else:
#         mc_expectation_dataset(
#             dataset=args.dataset,
#             gnn=args.gnn,
#             p=args.p,
#             q=args.q,
#             num_samples=args.samples,
#             subset_nodes=args.subset_nodes,
#             correct_self_loop=args.correct_self_loop,
#             device=args.device,
#         )
#
#
# if __name__ == "__main__":
#     main()

# check_unbiasedness_2layer.py
# import argparse
# from typing import List, Tuple
#
# import torch
# import torch.nn as nn
#
# from rade_convs import GraphCache, RADEGCNConv, RADEGINConv
#
#
# # ----------------------------
# # Helpers: undirected <-> directed
# # ----------------------------
# def undir_to_dir(edges_undir: List[Tuple[int, int]]) -> torch.Tensor:
#     """Expand undirected edges into both directed directions."""
#     row, col = [], []
#     for u, v in edges_undir:
#         row += [u, v]
#         col += [v, u]
#     if len(row) == 0:
#         return torch.empty((2, 0), dtype=torch.long)
#     return torch.tensor([row, col], dtype=torch.long)
#
#
# def edge_list_undir_complete(n: int) -> List[Tuple[int, int]]:
#     return [(i, j) for i in range(n) for j in range(i + 1, n)]
#
#
# def complement_undir(n: int, edges_undir: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
#     E = set(edges_undir)
#     return [e for e in edge_list_undir_complete(n) if e not in E]
#
#
# def enumerate_masks(
#     edges_undir: List[Tuple[int, int]],
#     nonedges_undir: List[Tuple[int, int]],
#     p: float,
#     q: float,
#     device: str,
# ):
#     """
#     Enumerate all (keep_set, add_set) with probability under independent Bernoulli masks:
#       keep each clean edge with prob (1-p), drop with prob p
#       add each non-edge with prob q, not-add with prob (1-q)
#     Returns iterable of (edge_index_keep_dir, edge_index_add_dir, edge_index_aug_dir, prob).
#     """
#     m = len(edges_undir)
#     M = len(nonedges_undir)
#
#     for keep_bits in range(1 << m):
#         keep_set = [edges_undir[i] for i in range(m) if (keep_bits >> i) & 1]
#         num_keep = len(keep_set)
#         num_drop = m - num_keep
#         prob_keep = (1.0 - p) ** num_keep * (p ** num_drop)
#
#         edge_index_keep = undir_to_dir(keep_set).to(device)
#
#         for add_bits in range(1 << M):
#             add_set = [nonedges_undir[i] for i in range(M) if (add_bits >> i) & 1]
#             num_add = len(add_set)
#             num_noadd = M - num_add
#             prob_add = (q ** num_add) * ((1.0 - q) ** num_noadd)
#
#             prob = prob_keep * prob_add
#             if prob == 0.0:
#                 continue
#
#             edge_index_add = undir_to_dir(add_set).to(device)
#             edge_index_aug = torch.cat([edge_index_keep, edge_index_add], dim=1)
#
#             yield edge_index_keep, edge_index_add, edge_index_aug, prob
#
#
# # ----------------------------
# # Exact expectation on toy graph (2 layers, linear)
# # ----------------------------
# def exact_expectation_toy_2layer(
#     gnn: str,
#     p: float,
#     q: float,
#     correct_self_loop: bool,
#     share_mask: bool,
#     device: str = "cpu",
# ) -> None:
#     """
#     Enumerate ALL keep/add masks on a tiny graph and compute exact expectations for a 2-layer stack.
#
#     - No nonlinearities: GIN uses Identity MLP, eps=0; GCN uses bias=False and RADEGCNConv.
#     - If share_mask=True: same (keep/add) mask is used in both layers.
#       If share_mask=False: independent masks per layer (recommended for testing composition).
#     """
#
#     # Toy graph: 4-node chain 0-1-2-3 (undirected)
#     n = 4
#     edges_undir = [(0, 1), (1, 2), (2, 3)]
#     nonedges_undir = complement_undir(n, edges_undir)
#
#     edge_index_clean = undir_to_dir(edges_undir).to(device)
#
#     # Fixed features (deterministic)
#     torch.manual_seed(0)
#     x0 = torch.randn(n, 5, device=device)
#
#     # Cache from CLEAN graph
#     cache = GraphCache.from_edge_index(edge_index_clean, num_nodes=n)
#
#     # Build 2-layer stacks (clean / biased / unbiased) with identical params
#     if gnn == "gin":
#         mlp = nn.Identity()
#
#         clean1 = RADEGINConv(mlp, eps=0.0).to(device)
#         clean2 = RADEGINConv(mlp, eps=0.0).to(device)
#         biased1 = RADEGINConv(mlp, eps=0.0).to(device)
#         biased2 = RADEGINConv(mlp, eps=0.0).to(device)
#         unb1 = RADEGINConv(mlp, eps=0.0).to(device)
#         unb2 = RADEGINConv(mlp, eps=0.0).to(device)
#
#         for m in [clean1, clean2, biased1, biased2, unb1, unb2]:
#             m.set_graph(cache)
#
#     elif gnn == "gcn":
#         in_dim = x0.size(1)
#         out_dim = x0.size(1)
#
#         clean1 = RADEGCNConv(in_dim, out_dim, bias=False, correct_self_loop=correct_self_loop).to(device)
#         clean2 = RADEGCNConv(in_dim, out_dim, bias=False, correct_self_loop=correct_self_loop).to(device)
#         biased1 = RADEGCNConv(in_dim, out_dim, bias=False, correct_self_loop=correct_self_loop).to(device)
#         biased2 = RADEGCNConv(in_dim, out_dim, bias=False, correct_self_loop=correct_self_loop).to(device)
#         unb1 = RADEGCNConv(in_dim, out_dim, bias=False, correct_self_loop=correct_self_loop).to(device)
#         unb2 = RADEGCNConv(in_dim, out_dim, bias=False, correct_self_loop=correct_self_loop).to(device)
#
#         for m in [clean1, clean2, biased1, biased2, unb1, unb2]:
#             m.set_graph(cache)
#
#         # Deterministic identical weights for corresponding layers
#         torch.manual_seed(0)
#         W1 = torch.randn(out_dim, in_dim, device=device)
#         W2 = torch.randn(out_dim, in_dim, device=device)
#         with torch.no_grad():
#             clean1.lin.weight.copy_(W1)
#             biased1.lin.weight.copy_(W1)
#             unb1.lin.weight.copy_(W1)
#
#             clean2.lin.weight.copy_(W2)
#             biased2.lin.weight.copy_(W2)
#             unb2.lin.weight.copy_(W2)
#
#     else:
#         raise ValueError("gnn must be 'gin' or 'gcn'")
#
#     # Clean 2-layer output (deterministic)
#     with torch.no_grad():
#         h1_clean = clean1(x0, edge_index_clean, edge_index_keep=None)
#         y_clean = clean2(h1_clean, edge_index_clean, edge_index_keep=None)
#
#     y_biased_exp = torch.zeros_like(y_clean)
#     y_unbiased_exp = torch.zeros_like(y_clean)
#
#     masks = list(enumerate_masks(edges_undir, nonedges_undir, p=p, q=q, device=device))
#
#     if share_mask:
#         # Same mask in both layers: enumerate mask once.
#         for edge_keep, edge_add, edge_aug, prob in masks:
#             with torch.no_grad():
#                 # biased: run on augmented edges in both layers
#                 h1_b = biased1(x0, edge_aug, edge_index_keep=None)
#                 y_b = biased2(h1_b, edge_aug, edge_index_keep=None)
#
#                 # RADE: same keep/add in both layers
#                 h1_u = unb1(
#                     x0,
#                     edge_index_clean,
#                     edge_index_keep=edge_keep,
#                     edge_index_add=edge_add,
#                     p=p,
#                     q=q,
#                 )
#                 y_u = unb2(
#                     h1_u,
#                     edge_index_clean,
#                     edge_index_keep=edge_keep,
#                     edge_index_add=edge_add,
#                     p=p,
#                     q=q,
#                 )
#
#             y_biased_exp += prob * y_b
#             y_unbiased_exp += prob * y_u
#
#     else:
#         # Independent masks per layer: enumerate mask1 x mask2.
#         for edge_keep1, edge_add1, edge_aug1, prob1 in masks:
#             for edge_keep2, edge_add2, edge_aug2, prob2 in masks:
#                 prob = prob1 * prob2
#                 if prob == 0.0:
#                     continue
#
#                 with torch.no_grad():
#                     # biased baseline: aug1 then aug2
#                     h1_b = biased1(x0, edge_aug1, edge_index_keep=None)
#                     y_b = biased2(h1_b, edge_aug2, edge_index_keep=None)
#
#                     # RADE: keep/add1 then keep/add2
#                     h1_u = unb1(
#                         x0,
#                         edge_index_clean,
#                         edge_index_keep=edge_keep1,
#                         edge_index_add=edge_add1,
#                         p=p,
#                         q=q,
#                     )
#                     y_u = unb2(
#                         h1_u,
#                         edge_index_clean,
#                         edge_index_keep=edge_keep2,
#                         edge_index_add=edge_add2,
#                         p=p,
#                         q=q,
#                     )
#
#                 y_biased_exp += prob * y_b
#                 y_unbiased_exp += prob * y_u
#
#     def report(name: str, y_exp: torch.Tensor) -> None:
#         diff = y_exp - y_clean
#         print(f"{name}:")
#         print(f"  max|E[y]-y_clean| = {diff.abs().max().item():.6e}")
#         print(f"  mean|E[y]-y_clean| = {diff.abs().mean().item():.6e}")
#         print(f"  ||E[y]-y_clean||_F = {diff.norm().item():.6e}")
#
#     print("\n=== TOY (EXACT EXPECTATION, 2 LAYERS, LINEAR) ===")
#     print(f"gnn={gnn} | p={p} | q={q} | correct_self_loop={correct_self_loop} | share_mask={share_mask} | device={device}")
#     report("Biased (aug graph directly)", y_biased_exp)
#     report("Unbiased (RADE keep/add)", y_unbiased_exp)
#     print("Clean is the reference.\n")
#
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--gnn", type=str, default="gin", choices=["gin", "gcn"])
#     parser.add_argument("--p", type=float, default=0.3)
#     parser.add_argument("--q", type=float, default=0.2)
#     parser.add_argument("--correct_self_loop", type=lambda x: x.lower() == "true", default=True)
#     parser.add_argument("--share_mask", type=bool, default=False)
#     parser.add_argument("--device", type=str, default="cpu")
#     args = parser.parse_args()
#
#     exact_expectation_toy_2layer(
#         gnn=args.gnn,
#         p=args.p,
#         q=args.q,
#         correct_self_loop=args.correct_self_loop,
#         share_mask=args.share_mask,
#         device=args.device,
#     )
#
#
# if __name__ == "__main__":
#     main()


# diagnostics_unbiased_model.py
# Compare "biased" vs "RADE (unbiased aggregation)" for FULL MPNNs model.
#
# Usage examples:
#   python diagnostics_unbiased_model.py --mode toy --gnn gin --p 0.3 --q 0.2
#   python diagnostics_unbiased_model.py --mode toy --gnn gcn --p 0.3 --q 0.2 --correct_self_loop True
#
#   python diagnostics_unbiased_model.py --mode dataset --dataset cora --gnn gin --p 0.1 --q 0.000143 --samples 200 --subset_nodes 512 --device cpu
#   python diagnostics_unbiased_model.py --mode dataset --dataset cora --gnn gcn --p 0.1 --q 0.000143 --samples 200 --subset_nodes 512 --device cuda:0
#
# Notes:
# - Set dropout=0 to avoid mixing dropout randomness into the expectation check.
# - For GCN, RADE uses approximations for E[Gamma_ij], so you should not expect near-zero residual.
# - For the FULL MPNNs (multiple layers + ReLU), exact equality is NOT expected because E[ReLU(Z)] != ReLU(E[Z]).
#   The goal is: RADE should REDUCE the shift relative to biased augmentation.

# import argparse
# import random
# from typing import List, Tuple
#
# import numpy as np
# import torch
#
# from rade_convs import RADEGCNConv  # to toggle correct_self_loop inside the model
# from models import MPNNs  # model.py must expose MPNNs
#
#
# # ----------------------------
# # Reproducibility
# # ----------------------------
# def fix_seed(seed: int = 0) -> None:
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#
#
# # ----------------------------
# # Helpers: undirected <-> directed
# # ----------------------------
# def undir_to_dir(edges_undir: List[Tuple[int, int]]) -> torch.Tensor:
#     """Expand undirected edges into both directed directions."""
#     row, col = [], []
#     for u, v in edges_undir:
#         row += [u, v]
#         col += [v, u]
#     if len(row) == 0:
#         return torch.empty((2, 0), dtype=torch.long)
#     return torch.tensor([row, col], dtype=torch.long)
#
#
# def edge_list_undir_complete(n: int) -> List[Tuple[int, int]]:
#     return [(i, j) for i in range(n) for j in range(i + 1, n)]
#
#
# def complement_undir(n: int, edges_undir: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
#     E = set(edges_undir)
#     return [e for e in edge_list_undir_complete(n) if e not in E]
#
#
# # ----------------------------
# # Build ONE model used for clean / biased / RADE
# # ----------------------------
# def build_model(
#     gnn: str,
#     in_dim: int,
#     hidden_dim: int,
#     out_dim: int,
#     local_layers: int,
#     dropout: float,
#     correct_self_loop: bool,
#     edge_index_clean: torch.Tensor,
#     num_nodes: int,
#     device: str,
# ) -> MPNNs:
#     model = MPNNs(
#         in_channels=in_dim,
#         hidden_channels=hidden_dim,
#         out_channels=out_dim,
#         local_layers=local_layers,
#         dropout=dropout,
#         pre_linear=False,
#         res=False,
#         ln=False,
#         bn=False,
#         jk=False,
#         gnn=gnn,
#         unbiased=True,  # IMPORTANT: RADE convs inside
#     ).to(device)
#
#     # Set clean graph cache for RADE convs
#     model.set_graph(edge_index_clean.to(device), num_nodes=num_nodes)
#
#     # Toggle correct_self_loop inside each RADEGCNConv (if gnn == "gcn")
#     for conv in getattr(model, "local_convs", []):
#         if isinstance(conv, RADEGCNConv):
#             conv.correct_self_loop = bool(correct_self_loop)
#
#     # Keep training=True so RADE path can trigger (model checks self.training)
#     # With dropout=0 and no BN/LN, this is deterministic.
#     model.train()
#     return model
#
#
# # ----------------------------
# # Reporting
# # ----------------------------
# def report_shift(name: str, y_exp: torch.Tensor, y_clean: torch.Tensor) -> None:
#     diff = y_exp - y_clean
#     print(f"{name}:")
#     print(f"  max|E[y]-y_clean| = {diff.abs().max().item():.6e}")
#     print(f"  mean|E[y]-y_clean| = {diff.abs().mean().item():.6e}")
#     print(f"  ||E[y]-y_clean||_F = {diff.norm().item():.6e}")
#
#
# # ----------------------------
# # EXACT expectation on tiny toy graph (enumerate all masks)
# # ----------------------------
# def exact_expectation_toy(
#     gnn: str,
#     p: float,
#     q: float,
#     correct_self_loop: bool,
#     hidden_dim: int,
#     out_dim: int,
#     local_layers: int,
#     dropout: float,
#     device: str,
# ) -> None:
#     # Toy graph: 4-node chain 0-1-2-3 (undirected)
#     n = 4
#     edges_undir = [(0, 1), (1, 2), (2, 3)]
#     nonedges_undir = complement_undir(n, edges_undir)
#
#     edge_index_clean = undir_to_dir(edges_undir).to(device)
#
#     # Fixed features
#     fix_seed(0)
#     x = torch.randn(n, 5, device=device)
#
#     # One shared model
#     model = build_model(
#         gnn=gnn,
#         in_dim=x.size(1),
#         hidden_dim=hidden_dim,
#         out_dim=out_dim,
#         local_layers=local_layers,
#         dropout=dropout,
#         correct_self_loop=correct_self_loop,
#         edge_index_clean=edge_index_clean,
#         num_nodes=n,
#         device=device,
#     )
#
#     with torch.no_grad():
#         y_clean = model(x, edge_index_clean)
#
#     m = len(edges_undir)
#     M = len(nonedges_undir)
#
#     y_biased_exp = torch.zeros_like(y_clean)
#     y_unbiased_exp = torch.zeros_like(y_clean)
#
#     # Enumerate keep mask on m edges, add mask on M non-edges
#     for keep_bits in range(1 << m):
#         keep_set = [edges_undir[i] for i in range(m) if (keep_bits >> i) & 1]
#         num_keep = len(keep_set)
#         num_drop = m - num_keep
#         prob_keep = (1.0 - p) ** num_keep * (p ** num_drop)
#
#         edge_index_keep = undir_to_dir(keep_set).to(device)
#
#         for add_bits in range(1 << M):
#             add_set = [nonedges_undir[i] for i in range(M) if (add_bits >> i) & 1]
#             num_add = len(add_set)
#             num_noadd = M - num_add
#             prob_add = (q ** num_add) * ((1.0 - q) ** num_noadd)
#
#             prob = prob_keep * prob_add
#             if prob == 0.0:
#                 continue
#
#             edge_index_add = undir_to_dir(add_set).to(device)
#             edge_index_aug = torch.cat([edge_index_keep, edge_index_add], dim=1)
#
#             with torch.no_grad():
#                 # Biased baseline: run on augmented graph directly (no keep/add provided)
#                 y_biased = model(x, edge_index_aug)
#
#                 # RADE: run keep/add correction relative to clean
#                 y_unbiased = model(
#                     x,
#                     edge_index_clean,
#                     edge_index_keep=edge_index_keep,
#                     edge_index_add=edge_index_add,
#                     p=p,
#                     q=q,
#                 )
#
#             y_biased_exp += prob * y_biased
#             y_unbiased_exp += prob * y_unbiased
#
#     print("\n=== TOY (EXACT EXPECTATION) ===")
#     print(
#         f"gnn={gnn} | p={p} | q={q} | correct_self_loop={correct_self_loop} | "
#         f"layers={local_layers} | hidden={hidden_dim} | out={out_dim} | dropout={dropout} | device={device}"
#     )
#     report_shift("Biased (aug graph directly)", y_biased_exp, y_clean)
#     report_shift("Unbiased (RADE keep/add)", y_unbiased_exp, y_clean)
#     print("Clean is the reference.\n")
#
#
# # ----------------------------
# # Monte Carlo expectation on real dataset
# # ----------------------------
# def mc_expectation_dataset(
#     dataset: str,
#     gnn: str,
#     p: float,
#     q: float,
#     num_samples: int,
#     subset_nodes: int,
#     correct_self_loop: bool,
#     hidden_dim: int,
#     local_layers: int,
#     dropout: float,
#     device: str,
# ) -> None:
#     # Import project utilities
#     from dataset import load_nc_dataset
#     from augmentation import BernoulliEdgeAugmentor
#
#     # Load data (same contract as main.py)
#     data, _ = load_nc_dataset(
#         root="data",
#         name=dataset,
#         normalize_features=True,
#         make_undirected_edges=True,
#         seed=0,
#         train_prop=0.6,
#         valid_prop=0.2,
#     )
#
#     data = data.to(device)
#     n = int(data.num_nodes)
#     x = data.x
#     edge_index_clean = data.edge_index
#     c = int(data.y.max().item()) + 1
#
#     # Node subset only affects reporting; RADE computations still see the whole graph.
#     if 0 < subset_nodes < n:
#         fix_seed(0)
#         idx = torch.randperm(n, device=device)[:subset_nodes].sort().values
#     else:
#         idx = torch.arange(n, device=device)
#
#     # Augmentor
#     augmentor = BernoulliEdgeAugmentor.from_edge_index(edge_index_clean, num_nodes=n)
#
#     # One shared model
#     model = build_model(
#         gnn=gnn,
#         in_dim=x.size(1),
#         hidden_dim=hidden_dim,
#         out_dim=c,  # logits dimension = num classes
#         local_layers=local_layers,
#         dropout=dropout,
#         correct_self_loop=correct_self_loop,
#         edge_index_clean=edge_index_clean,
#         num_nodes=n,
#         device=device,
#     )
#
#     with torch.no_grad():
#         y_clean = model(x, edge_index_clean)[idx]
#
#     y_biased_sum = torch.zeros_like(y_clean)
#     y_unbiased_sum = torch.zeros_like(y_clean)
#
#     for s in range(num_samples):
#         seed = 12345 + s
#
#         # keep/add for RADE, edge_index_aug for biased
#         edge_index_aug, edge_index_keep, edge_index_add, _ = augmentor.augment_edge_indices(
#             p=p, q=q, mode="both", seed=seed, device=device
#         )
#
#         with torch.no_grad():
#             y_biased = model(x, edge_index_aug)[idx]
#             y_unbiased = model(
#                 x,
#                 edge_index_clean,
#                 edge_index_keep=edge_index_keep,
#                 edge_index_add=edge_index_add,
#                 p=p,
#                 q=q,
#             )[idx]
#
#         y_biased_sum += y_biased
#         y_unbiased_sum += y_unbiased
#
#     y_biased_exp = y_biased_sum / float(num_samples)
#     y_unbiased_exp = y_unbiased_sum / float(num_samples)
#
#     print("\n=== DATASET (MONTE CARLO EXPECTATION) ===")
#     print(
#         f"dataset={dataset} | gnn={gnn} | p={p} | q={q} | samples={num_samples} | subset_nodes={idx.numel()} | "
#         f"correct_self_loop={correct_self_loop} | layers={local_layers} | hidden={hidden_dim} | dropout={dropout} | device={device}"
#     )
#     report_shift("Biased (aug graph directly)", y_biased_exp, y_clean)
#     report_shift("Unbiased (RADE keep/add)", y_unbiased_exp, y_clean)
#     print("Clean is the reference.\n")
#
#
# def str2bool(v: str) -> bool:
#     if isinstance(v, bool):
#         return v
#     v = v.lower().strip()
#     if v in {"1", "true", "t", "yes", "y"}:
#         return True
#     if v in {"0", "false", "f", "no", "n"}:
#         return False
#     raise argparse.ArgumentTypeError("Expected a boolean value (true/false).")
#
#
# def main() -> None:
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--mode", type=str, default="toy", choices=["toy", "dataset"])
#     parser.add_argument("--gnn", type=str, default="gin", choices=["gin", "gcn"])
#     parser.add_argument("--p", type=float, default=0.3)
#     parser.add_argument("--q", type=float, default=0.2)
#
#     parser.add_argument("--correct_self_loop", type=str2bool, default=True)
#
#     # Model size knobs
#     parser.add_argument("--hidden_dim", type=int, default=16)
#     parser.add_argument("--out_dim", type=int, default=8)       # used only in toy
#     parser.add_argument("--local_layers", type=int, default=2)
#     parser.add_argument("--dropout", type=float, default=0.0)   # IMPORTANT for expectation checking
#
#     # Dataset mode args
#     parser.add_argument("--dataset", type=str, default="toy")
#     parser.add_argument("--samples", type=int, default=200)
#     parser.add_argument("--subset_nodes", type=int, default=512)
#     parser.add_argument("--device", type=str, default="cpu")
#     parser.add_argument(
#         "--mask_sharing",
#         type=str,
#         default="layerwise",
#         choices=["shared", "layerwise"],
#         help="shared = one mask reused across layers; layerwise = independent mask per layer",
#     )
#
#     args = parser.parse_args()
#
#     if args.mode == "toy":
#         exact_expectation_toy(
#             gnn=args.gnn,
#             p=args.p,
#             q=args.q,
#             correct_self_loop=args.correct_self_loop,
#             hidden_dim=args.hidden_dim,
#             out_dim=args.out_dim,
#             local_layers=args.local_layers,
#             dropout=args.dropout,
#             device=args.device,
#         )
#     else:
#         mc_expectation_dataset(
#             dataset=args.dataset,
#             gnn=args.gnn,
#             p=args.p,
#             q=args.q,
#             num_samples=args.samples,
#             subset_nodes=args.subset_nodes,
#             correct_self_loop=args.correct_self_loop,
#             hidden_dim=args.hidden_dim,
#             local_layers=args.local_layers,
#             dropout=args.dropout,
#             device=args.device,
#         )
#
#
# if __name__ == "__main__":
#     main()


# unbiasedness_layerwise_check.py
# Copy-paste this file into project root (same level as dataset.py, augmentation.py, models.py, rade_convs.py)
#
# Examples:
#   (1) Toy, exact, shared mask across layers (fast)
#   python unbiasedness_layerwise_check.py --mode toy --gnn gcn --local_layers 2 --mask_sharing shared --linear_check true
#
#   (2) Toy, exact, layer-wise independent masks (still feasible for local_layers=2/3 on the 4-node toy)
#   python unbiasedness_layerwise_check.py --mode toy --gnn gcn --local_layers 2 --mask_sharing layerwise --linear_check true
#
#   (3) Dataset, Monte Carlo, layer-wise masks, unbiased vs clean
#   python unbiasedness_layerwise_check.py --mode dataset --dataset cora --gnn gcn --local_layers 3 \
#       --mask_sharing layerwise --samples 200 --subset_nodes 512 --linear_check false
#
# Notes:
# - For a strict expectation-preserving check, use: --linear_check true
#   (it removes outer ReLU/dropout by bypassing model.forward and applying only linear steps).
# - For GIN, strict unbiasedness will generally NOT hold if GIN MLP contains ReLU inside the conv.
#   So for strict checks, prefer --gnn gcn.

import argparse
import itertools
import random
from typing import List, Tuple, Optional, Union

import numpy as np
import torch
from torch_geometric.utils import coalesce

from rade_convs import RADEGCNConv  # to toggle correct_self_loop inside the model
from models import MPNNs  # model.py must expose MPNNs


# ----------------------------
# Reproducibility
# ----------------------------
def fix_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    v = v.lower().strip()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value (true/false).")


# ----------------------------
# Helpers: undirected <-> directed (toy)
# ----------------------------
def undir_to_dir(edges_undir: List[Tuple[int, int]]) -> torch.Tensor:
    """Expand undirected edges into both directed directions."""
    row, col = [], []
    for u, v in edges_undir:
        row += [u, v]
        col += [v, u]
    if len(row) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([row, col], dtype=torch.long)


def edge_list_undir_complete(n: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def complement_undir(n: int, edges_undir: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    E = set(edges_undir)
    return [e for e in edge_list_undir_complete(n) if e not in E]


# ----------------------------
# Reporting
# ----------------------------
def report_shift(name: str, y_exp: torch.Tensor, y_clean: torch.Tensor) -> None:
    diff = y_exp - y_clean
    print(f"{name}:")
    print(f"  max|E[y]-y_clean| = {diff.abs().max().item():.6e}")
    print(f"  mean|E[y]-y_clean| = {diff.abs().mean().item():.6e}")
    print(f"  ||E[y]-y_clean||_F = {diff.norm().item():.6e}")


# ----------------------------
# Build ONE model used for clean / biased / RADE
# ----------------------------
def build_model(
    *,
    gnn: str,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    local_layers: int,
    dropout: float,
    correct_self_loop: bool,
    edge_index_clean: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> MPNNs:
    model = MPNNs(
        in_channels=in_dim,
        hidden_channels=hidden_dim,
        out_channels=out_dim,
        local_layers=local_layers,
        dropout=dropout,
        pre_linear=False,
        res=False,
        ln=False,
        bn=False,
        jk=False,
        gnn=gnn,
        unbiased=True,  # IMPORTANT: RADE convs inside
    ).to(device)

    # Set clean graph cache for RADE convs (required)
    model.set_graph(edge_index_clean.to(device), num_nodes=num_nodes)

    # Toggle correct_self_loop inside each RADEGCNConv (only relevant for gcn)
    for conv in getattr(model, "local_convs", []):
        if isinstance(conv, RADEGCNConv):
            conv.correct_self_loop = bool(correct_self_loop)

    # Keep training=True so model's RADE gating (self.training) can trigger if you use model.forward
    model.train()
    return model


# ----------------------------
# Linear forward passes (no outer nonlinearity/dropout)
# These bypass model.forward so you can do strict expectation checks
# ----------------------------
EdgeIdx = torch.Tensor
MaybeEdgeList = Union[EdgeIdx, List[EdgeIdx], Tuple[EdgeIdx, ...]]


def _as_list(x: Optional[MaybeEdgeList]) -> Optional[List[EdgeIdx]]:
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


@torch.no_grad()
def forward_clean_linear(model: MPNNs, x: torch.Tensor, edge_index: EdgeIdx) -> torch.Tensor:
    h = x
    for conv in model.local_convs:
        h = conv(h, edge_index)  # plain message passing on provided graph
    return model.pred_local(h)


@torch.no_grad()
def forward_biased_layerwise_linear(model: MPNNs, x: torch.Tensor, edge_index_list: List[EdgeIdx]) -> torch.Tensor:
    h = x
    L = len(model.local_convs)
    if len(edge_index_list) != L:
        raise ValueError(f"edge_index_list length={len(edge_index_list)} must equal num_layers={L}")
    for l, conv in enumerate(model.local_convs):
        h = conv(h, edge_index_list[l])
    return model.pred_local(h)


@torch.no_grad()
def forward_rade_linear(
    model: MPNNs,
    x: torch.Tensor,
    edge_index_clean: EdgeIdx,
    *,
    edge_index_keep: Optional[MaybeEdgeList],
    edge_index_add: Optional[MaybeEdgeList],
    p: float,
    q: float,
) -> torch.Tensor:
    """
    RADE forward WITHOUT outer nonlinearities:
      layer l uses keep/add for that layer if lists are provided; otherwise shared keep/add.
    """
    h = x
    L = len(model.local_convs)

    keep_list = _as_list(edge_index_keep)
    add_list = _as_list(edge_index_add)

    # If shared (tensor), lists will have length 1; treat as broadcast.
    for l, conv in enumerate(model.local_convs):
        keep_l = None
        add_l = None
        if keep_list is not None:
            keep_l = keep_list[l] if len(keep_list) == L else keep_list[0]
        if add_list is not None:
            add_l = add_list[l] if len(add_list) == L else add_list[0]

        h = conv(
            h,
            edge_index_clean,
            edge_index_keep=keep_l,
            edge_index_add=add_l,
            p=p,
            q=q,
        )

    return model.pred_local(h)


# ----------------------------
# EXACT expectation on tiny toy graph (enumerate all masks)
# ----------------------------
def exact_expectation_toy(
    *,
    gnn: str,
    p: float,
    q: float,
    correct_self_loop: bool,
    hidden_dim: int,
    out_dim: int,
    local_layers: int,
    dropout: float,
    device: torch.device,
    mask_sharing: str,
    biased_layerwise: bool,
    linear_check: bool,
) -> None:
    # Toy graph: 4-node chain 0-1-2-3 (undirected)
    n = 4
    edges_undir = [(0, 1), (1, 2), (2, 3)]
    nonedges_undir = complement_undir(n, edges_undir)

    edge_index_clean = undir_to_dir(edges_undir).to(device)

    # Fixed features
    fix_seed(0)
    x = torch.randn(n, 5, device=device)

    # One shared model
    model = build_model(
        gnn=gnn,
        in_dim=x.size(1),
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        local_layers=local_layers,
        dropout=dropout,
        correct_self_loop=correct_self_loop,
        edge_index_clean=edge_index_clean,
        num_nodes=n,
        device=device,
    )

    # Clean reference
    if linear_check:
        y_clean = forward_clean_linear(model, x, edge_index_clean)
    else:
        y_clean = model(x, edge_index_clean)  # includes outer relu/dropout in MPNNs

    m = len(edges_undir)
    M = len(nonedges_undir)
    L = local_layers

    y_biased_exp = torch.zeros_like(y_clean)
    y_unbiased_exp = torch.zeros_like(y_clean)

    # Precompute all keep/add configurations for speed
    keep_cfgs = []
    for keep_bits in range(1 << m):
        keep_set = [edges_undir[i] for i in range(m) if (keep_bits >> i) & 1]
        num_keep = len(keep_set)
        num_drop = m - num_keep
        prob_keep = (1.0 - p) ** num_keep * (p ** num_drop)
        ei_keep = undir_to_dir(keep_set).to(device)
        keep_cfgs.append((ei_keep, prob_keep))

    add_cfgs = []
    for add_bits in range(1 << M):
        add_set = [nonedges_undir[i] for i in range(M) if (add_bits >> i) & 1]
        num_add = len(add_set)
        num_noadd = M - num_add
        prob_add = (q ** num_add) * ((1.0 - q) ** num_noadd)
        ei_add = undir_to_dir(add_set).to(device)
        add_cfgs.append((ei_add, prob_add))

    if mask_sharing == "shared":
        # One keep/add mask reused across layers
        for ei_keep, pk in keep_cfgs:
            for ei_add, pa in add_cfgs:
                prob = pk * pa
                if prob == 0.0:
                    continue

                # biased graph
                ei_aug = coalesce(torch.cat([ei_keep, ei_add], dim=1), num_nodes=n)

                if linear_check:
                    # biased baseline: same augmented graph at every layer unless biased_layerwise=True (but in shared mode it is the same anyway)
                    if biased_layerwise:
                        aug_list = [ei_aug] * L
                        y_biased = forward_biased_layerwise_linear(model, x, aug_list)
                    else:
                        y_biased = forward_clean_linear(model, x, ei_aug)

                    # RADE (shared keep/add broadcast)
                    y_unbiased = forward_rade_linear(
                        model,
                        x,
                        edge_index_clean,
                        edge_index_keep=ei_keep,
                        edge_index_add=ei_add,
                        p=p,
                        q=q,
                    )
                else:
                    y_biased = model(x, ei_aug)
                    y_unbiased = model(
                        x,
                        edge_index_clean,
                        edge_index_keep=ei_keep,
                        edge_index_add=ei_add,
                        p=p,
                        q=q,
                    )

                y_biased_exp += prob * y_biased
                y_unbiased_exp += prob * y_unbiased

    else:
        # Layer-wise: independent keep/add per layer.
        # Exact expectation enumerates the product space across layers.
        keep_space = list(range(len(keep_cfgs)))
        add_space = list(range(len(add_cfgs)))

        for keep_idx_vec in itertools.product(keep_space, repeat=L):
            keep_list = []
            prob_keep = 1.0
            for idx in keep_idx_vec:
                ei_keep, pk = keep_cfgs[idx]
                keep_list.append(ei_keep)
                prob_keep *= pk
            if prob_keep == 0.0:
                continue

            for add_idx_vec in itertools.product(add_space, repeat=L):
                add_list = []
                prob_add = 1.0
                for idx in add_idx_vec:
                    ei_add, pa = add_cfgs[idx]
                    add_list.append(ei_add)
                    prob_add *= pa
                prob = prob_keep * prob_add
                if prob == 0.0:
                    continue

                if linear_check:
                    # Build augmented edge_index per layer
                    aug_list = [
                        coalesce(torch.cat([keep_list[l], add_list[l]], dim=1), num_nodes=n)
                        for l in range(L)
                    ]

                    if biased_layerwise:
                        y_biased = forward_biased_layerwise_linear(model, x, aug_list)
                    else:
                        # simplest biased baseline: use layer 0 augmented graph for all layers (not truly layer-wise)
                        y_biased = forward_clean_linear(model, x, aug_list[0])

                    # True layer-wise RADE
                    y_unbiased = forward_rade_linear(
                        model,
                        x,
                        edge_index_clean,
                        edge_index_keep=keep_list,
                        edge_index_add=add_list,
                        p=p,
                        q=q,
                    )
                else:
                    # Nonlinear model.forward cannot express "different edge_index per layer" for biased baseline
                    # unless you rewrite the model. So we keep the biased baseline as "use layer 0 aug graph".
                    ei_aug0 = coalesce(torch.cat([keep_list[0], add_list[0]], dim=1), num_nodes=n)
                    y_biased = model(x, ei_aug0)
                    y_unbiased = model(
                        x,
                        edge_index_clean,
                        edge_index_keep=keep_list,
                        edge_index_add=add_list,
                        p=p,
                        q=q,
                    )

                y_biased_exp += prob * y_biased
                y_unbiased_exp += prob * y_unbiased

    print("\n=== TOY (EXACT EXPECTATION) ===")
    print(
        f"gnn={gnn} | p={p} | q={q} | correct_self_loop={correct_self_loop} | "
        f"layers={local_layers} | hidden={hidden_dim} | out={out_dim} | dropout={dropout} | device={device.type} | "
        f"mask_sharing={mask_sharing} | biased_layerwise={biased_layerwise} | linear_check={linear_check}"
    )
    report_shift("Biased (baseline)", y_biased_exp, y_clean)
    report_shift("Unbiased (RADE)", y_unbiased_exp, y_clean)
    print("Clean is the reference.\n")


# ----------------------------
# Monte Carlo expectation on real dataset
# ----------------------------
def mc_expectation_dataset(
    *,
    dataset: str,
    gnn: str,
    p: float,
    q: float,
    num_samples: int,
    subset_nodes: int,
    correct_self_loop: bool,
    hidden_dim: int,
    local_layers: int,
    dropout: float,
    device: torch.device,
    mask_sharing: str,
    biased_layerwise: bool,
    linear_check: bool,
) -> None:
    from dataset import load_nc_dataset
    from augmentation import BernoulliEdgeAugmentor

    data, _ = load_nc_dataset(
        root="data",
        name=dataset,
        normalize_features=True,
        make_undirected_edges=True,
        seed=0,
        train_prop=0.6,
        valid_prop=0.2,
    )
    data = data.to(device)
    n = int(data.num_nodes)
    x = data.x
    edge_index_clean = data.edge_index
    c = int(data.y.max().item()) + 1

    # Subset nodes only for reporting speed
    if 0 < subset_nodes < n:
        fix_seed(0)
        idx = torch.randperm(n, device=device)[:subset_nodes].sort().values
    else:
        idx = torch.arange(n, device=device)

    augmentor = BernoulliEdgeAugmentor.from_edge_index(edge_index_clean, num_nodes=n)

    model = build_model(
        gnn=gnn,
        in_dim=x.size(1),
        hidden_dim=hidden_dim,
        out_dim=c,
        local_layers=local_layers,
        dropout=dropout,
        correct_self_loop=correct_self_loop,
        edge_index_clean=edge_index_clean,
        num_nodes=n,
        device=device,
    )

    # Clean reference
    if linear_check:
        y_clean = forward_clean_linear(model, x, edge_index_clean)[idx]
    else:
        y_clean = model(x, edge_index_clean)[idx]

    y_biased_sum = torch.zeros_like(y_clean)
    y_unbiased_sum = torch.zeros_like(y_clean)

    for s in range(num_samples):
        seed = 12345 + s

        if mask_sharing == "shared":
            edge_index_aug, edge_index_keep, edge_index_add, _ = augmentor.augment_edge_indices(
                p=p, q=q, mode="both", seed=seed, device=device
            )

            if linear_check:
                if biased_layerwise:
                    aug_list = [edge_index_aug] * local_layers
                    y_biased = forward_biased_layerwise_linear(model, x, aug_list)[idx]
                else:
                    y_biased = forward_clean_linear(model, x, edge_index_aug)[idx]

                y_unbiased = forward_rade_linear(
                    model,
                    x,
                    edge_index_clean,
                    edge_index_keep=edge_index_keep,
                    edge_index_add=edge_index_add,
                    p=p,
                    q=q,
                )[idx]
            else:
                y_biased = model(x, edge_index_aug)[idx]
                y_unbiased = model(
                    x,
                    edge_index_clean,
                    edge_index_keep=edge_index_keep,
                    edge_index_add=edge_index_add,
                    p=p,
                    q=q,
                )[idx]

        else:
            # layer-wise masks
            keep_list = []
            add_list = []
            aug_list = []

            for l in range(local_layers):
                _, ei_keep_l, ei_add_l, _ = augmentor.augment_edge_indices(
                    p=p, q=q, mode="both", seed=seed + 1_000_000 * l, device=device
                )
                keep_list.append(ei_keep_l)
                add_list.append(ei_add_l)
                aug_list.append(coalesce(torch.cat([ei_keep_l, ei_add_l], dim=1), num_nodes=n))

            if linear_check:
                if biased_layerwise:
                    y_biased = forward_biased_layerwise_linear(model, x, aug_list)[idx]
                else:
                    # simplest biased baseline (not truly layer-wise)
                    y_biased = forward_clean_linear(model, x, aug_list[0])[idx]

                y_unbiased = forward_rade_linear(
                    model,
                    x,
                    edge_index_clean,
                    edge_index_keep=keep_list,
                    edge_index_add=add_list,
                    p=p,
                    q=q,
                )[idx]
            else:
                # Nonlinear model.forward cannot do a different edge_index per layer for biased baseline (without rewriting MPNNs).
                y_biased = model(x, aug_list[0])[idx]
                y_unbiased = model(
                    x,
                    edge_index_clean,
                    edge_index_keep=keep_list,
                    edge_index_add=add_list,
                    p=p,
                    q=q,
                )[idx]

        y_biased_sum += y_biased
        y_unbiased_sum += y_unbiased

    y_biased_exp = y_biased_sum / float(num_samples)
    y_unbiased_exp = y_unbiased_sum / float(num_samples)

    print("\n=== DATASET (MONTE CARLO EXPECTATION) ===")
    print(
        f"dataset={dataset} | gnn={gnn} | p={p} | q={q} | samples={num_samples} | subset_nodes={idx.numel()} | "
        f"correct_self_loop={correct_self_loop} | layers={local_layers} | hidden={hidden_dim} | dropout={dropout} | device={device.type} | "
        f"mask_sharing={mask_sharing} | biased_layerwise={biased_layerwise} | linear_check={linear_check}"
    )
    report_shift("Biased (baseline)", y_biased_exp, y_clean)
    report_shift("Unbiased (RADE)", y_unbiased_exp, y_clean)
    print("Clean is the reference.\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="toy", choices=["toy", "dataset"])
    parser.add_argument("--dataset", type=str, default="cora")

    parser.add_argument("--gnn", type=str, default="gcn", choices=["gin", "gcn"])
    parser.add_argument("--p", type=float, default=0.3)
    parser.add_argument("--q", type=float, default=0.2)
    parser.add_argument("--correct_self_loop", type=str2bool, default=True)

    # Model knobs
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--out_dim", type=int, default=8)     # used only in toy
    parser.add_argument("--local_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Mask behavior
    parser.add_argument("--mask_sharing", type=str, default="layerwise", choices=["shared", "layerwise"])
    parser.add_argument(
        "--biased_layerwise",
        type=str2bool,
        default=True,
        help="If true, biased baseline also uses layer-wise augmented graphs (only supported in linear_check mode).",
    )

    # Strict expectation check
    parser.add_argument(
        "--linear_check",
        type=str2bool,
        default=True,
        help="If true, removes outer nonlinearities by using linear forward helpers (recommended for expectation tests).",
    )

    # Dataset MC args
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--subset_nodes", type=int, default=512)

    # Device
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    device = torch.device(args.device)

    if args.gnn == "gin" and args.linear_check:
        print(
            "WARNING: You are using gnn=gin with linear_check=true.\n"
            "If GIN MLP contains ReLU inside the conv (common), exact expectation preservation generally will NOT hold.\n"
            "For strict checks, prefer gnn=gcn.\n"
        )

    if args.mode == "toy":
        exact_expectation_toy(
            gnn=args.gnn,
            p=args.p,
            q=args.q,
            correct_self_loop=args.correct_self_loop,
            hidden_dim=args.hidden_dim,
            out_dim=args.out_dim,
            local_layers=args.local_layers,
            dropout=args.dropout,
            device=device,
            mask_sharing=args.mask_sharing,
            biased_layerwise=args.biased_layerwise,
            linear_check=args.linear_check,
        )
    else:
        mc_expectation_dataset(
            dataset=args.dataset,
            gnn=args.gnn,
            p=args.p,
            q=args.q,
            num_samples=args.samples,
            subset_nodes=args.subset_nodes,
            correct_self_loop=args.correct_self_loop,
            hidden_dim=args.hidden_dim,
            local_layers=args.local_layers,
            dropout=args.dropout,
            device=device,
            mask_sharing=args.mask_sharing,
            biased_layerwise=args.biased_layerwise,
            linear_check=args.linear_check,
        )


if __name__ == "__main__":
    main()
