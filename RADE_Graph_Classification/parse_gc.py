# graph_classification/parse_gc.py
from __future__ import annotations

try:
    from .models_gc import GraphMPNN, GraphMPNNConfig
except Exception:
    from models_gc import GraphMPNN, GraphMPNNConfig


def parse_method_gc(args, in_channels: int, out_channels: int, device):
    cfg = GraphMPNNConfig(
        gnn=args.gnn,
        pooling=args.pooling,
        local_layers=args.local_layers,
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
        bn=bool(getattr(args, "bn", False)),  # <--- NEW
        use_ogb_encoders=args.use_ogb_encoders,
        use_edge_attr=args.use_edge_attr,
        unbiased=bool(args.unbiased),
        unbiased_mode=str(args.unbiased_mode),
        correct_self_loop=True,
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
        default="mutag",
        choices=["mutag", "proteins", "imdb-b", "imdb-m", "ogbg-molhiv", "ogbg-molpcba"],
    )
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--seed", type=int, default=42)

    # TU split ratios (OGB uses official split)
    parser.add_argument("--train_prop", type=float, default=0.8)
    parser.add_argument("--valid_prop", type=float, default=0.1)

    # -----------------
    # system
    # -----------------
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")

    # -----------------
    # training
    # -----------------
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)

    # -----------------
    # model
    # -----------------
    parser.add_argument("--gnn", type=str, default="gin", choices=["gcn", "gin"])
    parser.add_argument("--pooling", type=str, default="sum", choices=["mean", "sum"])
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--local_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--bn", action="store_true", help="Enable BatchNorm1d after each conv.")  # <--- NEW

    parser.add_argument("--use_ogb_encoders", action="store_true")
    parser.add_argument("--use_edge_attr", action="store_true")

    # -----------------
    # augmentation (training-time only; applied in main_gc.py)
    # -----------------
    parser.add_argument(
        "--aug_mode",
        type=str,
        default="both",
        choices=["none", "drop", "add", "both"],
        help="Graph topology augmentation mode on training batches.",
    )
    parser.add_argument("--p", type=float, default=0.02, help="Edge drop probability for clean edges (0<=p<1).")
    parser.add_argument("--q", type=float, default=0.01, help="Non-edge add probability over clean complement (0<=q<=1).")
    parser.add_argument(
        "--safety_max_additions",
        type=int,
        default=2_000_000,
        help="Safety cap on expected number of added undirected non-edges per graph.",
    )

    # RADE controls (GC)
    parser.add_argument("--unbiased", action="store_true")
    parser.add_argument("--unbiased_mode", type=str, default="rade-ic", choices=["rade", "rade-ic"])

    # PQ-GradNorm (graph classification)
    parser.add_argument("--pq_gradnorm", action="store_true",
                        help="Enable PQ-GradNorm tuning for graph classification.")
    parser.add_argument("--pq_warmup_epochs", type=int, default=0,
                        help="Warmup epochs before PQ tuning starts.")
    parser.add_argument("--pq_update_every", type=int, default=1,
                        help="Run PQ tuning every K epochs.")
    parser.add_argument("--pq_grid_size", type=int, default=11,
                        help="Grid resolution for p and q.")
    parser.add_argument("--pq_p_max", type=float, default=0.8,
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
    parser.add_argument("--loss", type=str, default="auto", choices=["auto", "ce", "bce"])
    parser.add_argument("--metric", type=str, default="auto", choices=["auto", "acc", "rocauc", "ap"])

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
