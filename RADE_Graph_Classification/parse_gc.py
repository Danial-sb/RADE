# graph_classification/parse_gc.py
from __future__ import annotations
import argparse

try:
    from .models_gc import GraphMPNN, GraphMPNNConfig
except Exception:
    from models_gc import GraphMPNN, GraphMPNNConfig

def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got '{v}'")

def parse_method_gc(args, in_channels: int, out_channels: int, device):
    cfg = GraphMPNNConfig(
        gnn=args.gnn,
        pooling=args.pooling,
        local_layers=args.local_layers,
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
        bn=bool(getattr(args, "bn", False)),
        use_ogb_encoders=bool(getattr(args, "use_ogb_encoders", False)),
        use_edge_attr=bool(getattr(args, "use_edge_attr", False)),
        pre_linear=bool(getattr(args, "pre_linear", False)),
        unbiased=bool(getattr(args, "unbiased", False)),
        unbiased_mode=str(getattr(args, "unbiased_mode", "rade")),
        correct_self_loop=True,
        linear=bool(getattr(args, "linear", False)),
        aug_tech=str(getattr(args, "aug_tech", "rade")),
    )
    model = GraphMPNN(in_channels=in_channels, out_channels=out_channels, cfg=cfg).to(device)
    return model


def parser_add_gc_args(parser):
    # -----------------
    # dataset
    # -----------------
    parser.add_argument(
        "--dataset",
        type=str,
        default="peptides-func",
        choices=["mutag", "proteins", "imdb-b", "imdb-m", "ogbg-molhiv", "ogbg-molpcba", "peptides-func"],
    )
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--seed", type=int, default=42)

    # TU split ratios (OGB uses official split)
    parser.add_argument("--train_prop", type=float, default=0.8)
    parser.add_argument("--valid_prop", type=float, default=0.1)

    # -----------------
    # system
    # -----------------
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")

    # -----------------
    # training
    # -----------------
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience in epochs based on "
                                                                  "validation accuracy. Set <=0 to disable "
                                                                  "early stopping.",)

    # -----------------
    # model
    # -----------------
    parser.add_argument("--gnn", type=str, default="gcn", choices=["gcn", "gin"])
    parser.add_argument("--pooling", type=str, default="sum", choices=["mean", "sum"])
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--local_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bn", action="store_true", help="Enable BatchNorm1d after each conv.")
    parser.add_argument(
        "--linear",
        type=str2bool, default=False,
        help="Use the linear version of the model (no activations, no BN, no dropout; GIN uses linear MLP).",
    )

    parser.add_argument(
        "--pre_linear",
        action="store_true",
        help="Apply a pre-linear projection from input dim to hidden dim before message passing (NC-style).",
    )

    parser.add_argument("--use_ogb_encoders", type=str2bool, default=False)
    parser.add_argument("--use_edge_attr", action="store_true")

    # -----------------------
    # Augmentation selector (NC style)
    # -----------------------
    parser.add_argument(
        "--aug_tech",
        type=str,
        default="rade",
        choices=["rade", "dropmessage", "dropnode", "none"],
        help="Which augmentation family to use. "
             "'rade' uses edge drop/add pipeline. "
             "'dropmessage' uses message dropout (no RADE/unbiased/gradnorm). "
             "'dropnode' uses node-wise feature masking (no RADE/unbiased/gradnorm). "
             "'none' is clean training.",
    )

    # DropMessage hyperparameter
    parser.add_argument(
        "--dropmessage_rate",
        type=float,
        default=0.0,
        help="DropMessage rate (message dropout probability). Used only when --aug_tech=dropmessage.",
    )

    # DropNode hyperparameter (node-wise feature masking + rescale)
    parser.add_argument(
        "--dropnode_rate",
        type=float,
        default=0.0,
        help="DropNode rate (node dropout probability). Used only when --aug_tech=dropnode.",
    )

    # -----------------
    # RADE augmentation
    # -----------------
    parser.add_argument(
        "--aug_mode",
        type=str,
        default="drop",
        choices=["none", "drop", "add", "both"],
        help="Graph topology augmentation mode on training batches.",
    )
    parser.add_argument("--p", type=float, default=0.2, help="Edge drop probability for clean edges (0<=p<1).")
    parser.add_argument("--q", type=float, default=0.01, help="Non-edge add probability over clean complement (0<=q<=1).")
    parser.add_argument(
        "--safety_max_additions",
        type=int,
        default=2_000_000,
        help="Safety cap on expected number of added undirected non-edges per graph.",
    )

    # RADE controls (GC)
    parser.add_argument("--unbiased", type=str2bool, default=False,
                        help="Use expectation-preserving (unbiased) RADE aggregation.")
    parser.add_argument("--unbiased_mode", type=str, default="rade", choices=["rade", "rade-ic"])
    parser.add_argument(
        "--mask_sharing",
        type=str,
        default="layerwise",
        choices=["shared", "layerwise"],
        help="RADE only: 'shared' uses one keep/add mask across all layers; "
             "'layerwise' samples independent keep/add per layer per epoch.",
    )

    # PQ-GradNorm (graph classification) -- RADE only
    parser.add_argument("--pq_gradnorm", type=str2bool, default=False,
                        help="Enable PQ-GradNorm tuning for graph classification (RADE only).")
    parser.add_argument("--pq_warmup_epochs", type=int, default=10,
                        help="Warmup epochs before PQ tuning starts.")
    parser.add_argument("--pq_update_every", type=int, default=1,
                        help="Run PQ tuning every K epochs.")
    parser.add_argument("--pq_grid_size", type=int, default=11,
                        help="Grid resolution for p and q.")
    parser.add_argument("--pq_p_max", type=float, default=0.5,
                        help="Maximum p searched by PQ tuner.")
    parser.add_argument("--pq_q_max", type=float, default=1e-3,
                        help="Maximum q searched by PQ tuner.")
    parser.add_argument("--pq_ema", type=float, default=0.9,
                        help="Epoch-level EMA for (p,q) after averaging batch-wise solutions. Set 0 to disable.")

    # -----------------
    # optimization
    # -----------------
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # -----------------
    # loss / metric override (auto is resolved in main)
    # -----------------
    parser.add_argument("--metric", type=str, default="auto",
                        choices=["auto", "acc", "rocauc", "ap"])

    parser.add_argument("--loss", type=str, default="auto",
                        choices=["auto", "ce", "bce"])

    # -----------------
    # wandb
    # -----------------
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="rade-gc")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_log_every", type=int, default=1)

    # -----------------
    # misc
    # -----------------
    parser.add_argument("--display_step", type=int, default=1)
    parser.add_argument("--save_model", action="store_true")
