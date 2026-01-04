# parse.py (revised for nc_datasets_simple.py compatibility)

from models import MPNNs


def parse_method(args, num_nodes, num_classes, in_dim, device):
    # num_nodes is unused by MPNNs, kept for API consistency
    model = MPNNs(
        in_channels=in_dim,
        hidden_channels=args.hidden_channels,
        out_channels=num_classes,
        local_layers=args.local_layers,
        dropout=args.dropout,
        pre_linear=args.pre_linear,
        res=args.res,
        ln=args.ln,
        bn=args.bn,
        jk=args.jk,
        gnn=args.gnn,
        unbiased=args.unbiased,
        unbiased_mode=args.unbiased_mode,
        linear=bool(getattr(args, "linear", False)),
        aug_tech=str(getattr(args, "aug_tech", "rade")),
    ).to(device)
    return model


def parser_add_main_args(parser):
    # dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="citeseer",
        choices=["cora", "citeseer", "pubmed", "computer", "cs", "physics", "flickr", "ogbn-arxiv"],
        help="Dataset name (must match nc_datasets_simple.py)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/",
        help="Root directory passed to load_nc_dataset(root=...)",
    )

    # system
    parser.add_argument("--device", type=int, default=0, help="GPU id (default: 0)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    # training schedule
    parser.add_argument("--epochs", type=int, default=500, help="number of training epochs")
    parser.add_argument("--runs", type=int, default=5, help="number of distinct runs")

    # split control (used only for datasets that are random-split in the loader)
    parser.add_argument("--train_prop", type=float, default=0.6, help="training proportion (random-split datasets)")
    parser.add_argument("--valid_prop", type=float, default=0.2, help="validation proportion (random-split datasets)")

    # Optional overrides (ONLY if your main script implements them explicitly)
    parser.add_argument("--rand_split", action="store_true",
                        help="Override loader split with a fresh random split (requires main-script support).")
    parser.add_argument("--rand_split_class", action="store_true",
                        help="Override loader split with class-balanced split (requires main-script support).")
    parser.add_argument("--label_num_per_class", type=int, default=20)
    parser.add_argument("--valid_num", type=int, default=500)
    parser.add_argument("--test_num", type=int, default=1000)

    # metric (for these datasets, accuracy is the expected default)
    parser.add_argument("--metric", type=str, default="acc", choices=["acc"], help="evaluation metric")

    # model
    parser.add_argument("--model", type=str, default="MPNN", choices=["MPNN"])
    parser.add_argument("--gnn", type=str, default="gin", choices=["gcn", "gin"])
    parser.add_argument("--hidden_channels", type=int, default=512)
    parser.add_argument("--local_layers", type=int, default=2)
    parser.add_argument("--pre_linear", action="store_true")
    parser.add_argument("--res", action="store_true")
    parser.add_argument("--ln", action="store_true")
    parser.add_argument("--bn", action="store_true")
    parser.add_argument("--jk", action="store_true")
    parser.add_argument("--linear", action="store_true", help="Use strictly linear model: disables ReLU/Dropout/BN/LN,"
                                                              "and makes GIN MLP linear.")

    # optimization
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    # early stopping
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience in epochs based on "
                                                                  "validation accuracy. Set <=0 to disable "
                                                                  "early stopping.",)

    # -----------------------
    # Augmentation selector
    # -----------------------
    parser.add_argument(
        "--aug_tech",
        type=str,
        default="none",
        choices=["rade", "dropmessage", "dropnode", "none"],
        help="Which augmentation family to use. "
             "'rade' uses your current edge drop/add pipeline. "
             "'dropmessage' uses message dropout (no RADE/unbiased/gradnorm). "
             "'dropnode' uses node-wise feature masking (no RADE/unbiased/gradnorm). "
             "'none' is clean training.",
    )

    # DropMessage hyperparameter
    parser.add_argument(
        "--dropmessage_rate",
        type=float,
        default=0.9,
        help="DropMessage rate (message dropout probability). Used only when --aug_tech=dropmessage.",
    )

    # DropNode hyperparameter
    parser.add_argument(
        "--dropnode_rate",
        type=float,
        default=0.9,
        help="DropNode rate (node dropout probability). Used only when --aug_tech=dropnode.",
    )

    # RADE augmentation args
    parser.add_argument("--aug_mode", type=str, default="both",
                        choices=["none", "drop", "add", "both"],
                        help="Graph augmentation mode per epoch.")
    parser.add_argument("--p", type=float, default=0.1,
                        help="Edge drop probability for edges (i,j) in E.")
    parser.add_argument("--q", type=float, default=0.000143,
                        help="Non-edge add probability for pairs (i,j) not in E (per-non-edge rate).")
    parser.add_argument("--unbiased", type=bool, default=False,
                        help="Use expectation-preserving (unbiased) aggregation for drop/add (RADE-style).")
    # augmentation (Bernoulli drop/add)
    parser.add_argument(
        "--mask_sharing",
        type=str,
        default="shared",
        choices=["shared", "layerwise"],
        help="RADE only: 'shared' uses one keep/add mask across all layers; "
             "'layerwise' samples independent keep/add per layer per epoch."
    )

    parser.add_argument(
        "--unbiased_mode",
        type=str,
        default="rade",
        choices=["rade", "rade-ic"],
        help="When --unbiased=True: "
             "'rade' uses train-time centering for additions (clean inference). "
             "'rade-ic' uses no centering at train and applies inference correction.",
    )

    # ---- GradNorm p/q tuning (epoch-wise) ----
    parser.add_argument("--pq_gradnorm", type=bool, default=False,
                        help="Enable epoch-wise GradNorm matching to adapt (p,q).")

    parser.add_argument("--p_max", type=float, default=0.6,
                        help="Upper bound for p during GradNorm tuning.")
    parser.add_argument("--q_max", type=float, default=1e-3,
                        help="Upper bound for q during GradNorm tuning.")

    parser.add_argument("--pq_grid_size", type=int, default=11,
                        help="Grid size for (p,q) search. Use 11 for GIN; use 3-5 for GCN (costlier).")

    parser.add_argument("--pq_subset_nodes", type=int, default=1024,
                        help="Number of train nodes used to estimate GradNorm norms.")
    parser.add_argument("--pq_ema", type=float, default=0.9,
                        help="EMA smoothing for p,q updates (stability).")

    parser.add_argument("--pq_update_every", type=int, default=1,
                        help="Update (p,q) every k epochs.")
    parser.add_argument("--pq_warmup_epochs", type=int, default=20,
                        help="Number of initial epochs to skip pq updates.")
    parser.add_argument("--pq_eps", type=float, default=1e-12,
                        help="Small epsilon for log/denominator stabilizers.")
    parser.add_argument("--pq_seed", type=int, default=0,
                        help="Seed for selecting the subset of nodes for pq selection.")


    # -----------------------
    # Weights & Biases (wandb)
    # -----------------------
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="rade-nc")
    parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity/team (optional)")
    parser.add_argument("--wandb_group", type=str, default=None, help="wandb group (optional)")
    parser.add_argument("--wandb_name", type=str, default=None, help="wandb run name (optional)")
    parser.add_argument("--wandb_tags", type=str, default="", help="comma-separated tags (optional)")
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="wandb mode",
    )
    parser.add_argument("--wandb_log_every", type=int, default=1, help="log every k epochs")

    # misc
    parser.add_argument("--display_step", type=int, default=1)
    parser.add_argument("--save_model", action="store_true", help="whether to save model (logger.py controls path)")
