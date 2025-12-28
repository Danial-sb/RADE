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
        rade_scope=args.rade_scope,
    ).to(device)
    return model


def parser_add_main_args(parser):
    # dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
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
    parser.add_argument("--epochs", type=int, default=200, help="number of training epochs")
    parser.add_argument("--runs", type=int, default=1, help="number of distinct runs")

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
    parser.add_argument("--gnn", type=str, default="gcn", choices=["gcn", "gin"])
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--local_layers", type=int, default=3)
    parser.add_argument("--pre_linear", action="store_true")
    parser.add_argument("--res", action="store_true")
    parser.add_argument("--ln", action="store_true")
    parser.add_argument("--bn", action="store_true")
    parser.add_argument("--jk", action="store_true")

    # optimization
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.0)

    # augmentation (Bernoulli drop/add)
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
        "--rade_scope",
        type=str,
        default="last",
        choices=["all", "last"],
        help="RADE application scope: 'all' applies RADE at every layer; "
             "'last' applies RADE only at final layer and removes last nonlinearity for a linear head.",
    )

    # misc
    parser.add_argument("--display_step", type=int, default=1)
    parser.add_argument("--save_model", action="store_true", help="whether to save model (logger.py controls path)")
