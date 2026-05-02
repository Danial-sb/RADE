import argparse

from models import MPNNs


def str2bool(value):
    if isinstance(value, bool):
        return value

    value_norm = str(value).strip().lower()
    if value_norm in {"1", "true", "t", "yes", "y"}:
        return True
    if value_norm in {"0", "false", "f", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


FULL_BATCH_AUTO_DEFAULTS = {
    ("cora", "gcn"): {
        "hidden_channels": 512,
        "epochs": 500,
        "p": 0.5,
        "q": 0.000721,
        "pq_grid_size": 5,
    },
    ("cora", "gin"): {
        "hidden_channels": 256,
        "epochs": 500,
        "p": 0.5,
        "q": 0.000721,
        "pq_grid_size": 11,
    },
    ("citeseer", "gcn"): {
        "hidden_channels": 512,
        "epochs": 500,
        "p": 0.2,
        "q": 0.000164,
        "pq_grid_size": 5,
    },
    ("citeseer", "gin"): {
        "hidden_channels": 512,
        "epochs": 500,
        "p": 0.2,
        "q": 0.000164,
        "pq_grid_size": 11,
    },
    ("pubmed", "gcn"): {
        "hidden_channels": 256,
        "epochs": 500,
        "p": 0.2,
        "q": 0.0000456,
        "pq_grid_size": 5,
    },
    ("pubmed", "gin"): {
        "hidden_channels": 512,
        "epochs": 500,
        "p": 0.2,
        "q": 0.0000456,
        "pq_grid_size": 11,
    },
    ("cs", "gcn"): {
        "hidden_channels": 64,
        "epochs": 1500,
        "p": 0.2,
        "q": 0.0000957,
        "pq_grid_size": 5,
    },
    ("cs", "gin"): {
        "hidden_channels": 256,
        "epochs": 1500,
        "p": 0.2,
        "q": 0.0000957,
        "pq_grid_size": 11,
    },
    ("computer", "gcn"): {
        "hidden_channels": 128,
        "epochs": 1000,
        "p": 0.5,
        "q": 0.001303,
        "pq_grid_size": 5,
        "patience": 200,
    },
    ("computer", "gin"): {
        "hidden_channels": 128,
        "epochs": 1000,
        "p": 0.5,
        "q": 0.001303,
        "pq_grid_size": 11,
        "patience": 200,
    },
    ("physics", "gcn"): {
        "hidden_channels": 64,
        "epochs": 1500,
        "p": 0.2,
        "q": 0.0000834,
        "pq_grid_size": 5,
    },
    ("physics", "gin"): {
        "hidden_channels": 64,
        "epochs": 1500,
        "p": 0.2,
        "q": 0.0000834,
        "pq_grid_size": 11,
    },
}


def get_explicit_arg_names(argv):
    explicit = set()

    for token in argv:
        if token == "--":
            break
        if not token.startswith("--"):
            continue

        option = token[2:].split("=", 1)[0].strip()
        if option:
            explicit.add(option.replace("-", "_"))

    return explicit


def apply_full_batch_auto_defaults(args, explicit_arg_names=None):
    explicit_arg_names = explicit_arg_names or set()

    dataset = str(getattr(args, "dataset", "")).lower().strip()
    gnn = str(getattr(args, "gnn", "")).lower().strip()
    preset_key = (dataset, gnn)
    preset = FULL_BATCH_AUTO_DEFAULTS.get(preset_key)

    if preset is None:
        return None, {}, {}

    applied = {}
    skipped = {}
    for field_name, field_value in preset.items():
        if field_name in explicit_arg_names:
            skipped[field_name] = getattr(args, field_name)
            continue
        setattr(args, field_name, field_value)
        applied[field_name] = field_value

    return preset_key, applied, skipped


def parse_method(args, num_nodes, num_classes, in_dim, device):
    ep_emp_average_mode = "ema" if bool(getattr(args, "pq_gradnorm", False)) else "running_mean"
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
        gnn=args.gnn,  # 'gcn' or 'gin'
        ep_correction=args.ep_correction,
        ep_expectation_mode=args.ep_expectation_mode,
        ep_emp_average_mode=ep_emp_average_mode,
        ep_emp_beta=args.ep_emp_beta,
        ep_emp_eps=args.ep_emp_eps,
        rade_variant=args.rade_variant,
        linear=bool(getattr(args, "linear", False)),
        aug_tech=str(getattr(args, "aug_tech", "rade")),
    ).to(device)
    return model


def parser_add_main_args(parser):
    # dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        choices=["cora", "citeseer", "pubmed", "cs", "computer", "physics", "flickr", "ogbn-arxiv"],
        help="Dataset name (must match nc_datasets_simple.py)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/",
        help="Root directory passed to load_nc_dataset(root=...)",
    )
    parser.add_argument(
        "--remove_citeseer_isolated_nodes",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Remove isolated nodes when loading CiteSeer. Ignored for other datasets.",
    )

    # system
    parser.add_argument("--device", type=int, default=3, help="GPU id (default: 0)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    # training schedule
    parser.add_argument("--epochs", type=int, default=500, help="number of training epochs")
    parser.add_argument("--runs", type=int, default=5, help="number of distinct runs")

    # split control
    parser.add_argument("--train_prop", type=float, default=0.6, help="training proportion (random-split datasets)")
    parser.add_argument("--valid_prop", type=float, default=0.2, help="validation proportion (random-split datasets)")

    parser.add_argument(
        "--rand_split",
        action="store_true",
        help="Override loader split with a fresh random split (requires main-script support).",
    )
    parser.add_argument(
        "--rand_split_class",
        action="store_true",
        help="Override loader split with class-balanced split (requires main-script support).",
    )
    parser.add_argument("--label_num_per_class", type=int, default=20)
    parser.add_argument("--valid_num", type=int, default=500)
    parser.add_argument("--test_num", type=int, default=1000)

    # metric
    parser.add_argument("--metric", type=str, default="acc", choices=["acc"], help="evaluation metric")

    # model
    parser.add_argument("--model", type=str, default="MPNN", choices=["MPNN"])
    parser.add_argument(
        "--gnn",
        type=str,
        default="gin",
        choices=["gcn", "gin", "sgc", "gin-linear"],
        help="Backbone: gcn/gin are standard. sgc routes to linear-GCN. gin-linear routes to linear-GIN."
    )
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--local_layers", type=int, default=2)
    parser.add_argument("--pre_linear", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--res", action="store_true")
    parser.add_argument("--ln", action="store_true")
    parser.add_argument("--bn", action="store_true")
    parser.add_argument("--jk", action="store_true")
    parser.add_argument(
        "--linear",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Use strictly linear model: disables ReLU/Dropout/BN/LN and makes GIN MLP linear.",
    )

    # optimization
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="Early stopping patience in epochs based on validation accuracy. Set <=0 to disable early stopping.",
    )

    # -----------------------
    # Augmentation selector
    # -----------------------
    parser.add_argument(
        "--aug_tech",
        type=str,
        default="rade",
        choices=["rade", "dropmessage", "dropnode", "none"],
        help="Which augmentation family to use. "
             "'rade' uses the edge drop/add RADE pipeline. "
             "'dropmessage' uses message dropout. "
             "'dropnode' uses node-wise feature masking. "
             "'none' is clean training.",
    )

    parser.add_argument(
        "--dropmessage_rate",
        type=float,
        default=0.9,
        help="DropMessage rate. Used only when --aug_tech=dropmessage.",
    )

    parser.add_argument(
        "--dropnode_rate",
        type=float,
        default=0.9,
        help="DropNode rate. Used only when --aug_tech=dropnode.",
    )

    # RADE perturbation args
    parser.add_argument(
        "--aug_mode",
        type=str,
        default="add",
        choices=["none", "drop", "add", "both"],
        help="Per-epoch graph perturbation mode.",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.2,
        help="Edge-drop probability for clean edges (i,j) in E.",
    )
    parser.add_argument(
        "--q",
        type=float,
        default=0.0000456,
        help="Non-edge-add probability for clean non-edges (i,j) not in E.",
    )

    parser.add_argument(
        "--ep_correction",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Use expectation-preserving aggregation correction for RADE.",
    )
    parser.add_argument(
        "--ep_expectation_mode",
        type=str,
        default="analytic",
        choices=["analytic", "empirical_ema"],
        help="How to obtain the expectation term used in EP correction. "
             "'analytic' uses the original closed-form / approximation. "
             "'empirical_ema' uses a lagged online EMA estimate as an ablation.",
    )
    parser.add_argument(
        "--ep_emp_beta",
        type=float,
        default=0.1,
        help="EMA update weight for the empirical expectation ablation.",
    )
    parser.add_argument(
        "--ep_emp_eps",
        type=float,
        default=1e-12,
        help="Small epsilon used to clamp empirical expectation denominators.",
    )

    parser.add_argument(
        "--mask_sharing",
        type=str,
        default="layerwise",
        choices=["shared", "layerwise"],
        help="RADE only: 'shared' uses one keep/add mask across all layers; "
             "'layerwise' samples independent keep/add masks per layer per epoch.",
    )

    parser.add_argument(
        "--rade_variant",
        type=str,
        default="rade-of",
        choices=["rade-of", "rade-ofs"],
        help="RADE variant. "
             "'rade-of' is the overfitting-oriented variant "
             "(center additions at train; clean inference). "
             "'rade-ofs' is the joint overfitting + over-squashing variant "
             "(uncentered additions at train; inference-time augmentation correction).",
    )

    # ---- GradNorm p/q tuning (epoch-wise) ----
    parser.add_argument(
        "--pq_gradnorm",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Enable epoch-wise GradNorm matching to adapt (p,q).",
    )

    parser.add_argument(
        "--pq_search_method",
        type=str,
        default="adam",
        choices=["grid", "powell", "newton", "adam", "gd"],
        help="How to update (p,q). "
             "'grid'/'powell'/'newton' use the existing epoch-wise search rule. "
             "'adam' uses a stateful online controller and one Adam step per PQ update. "
             "'gd' uses one projected gradient-descent step per PQ update.",
    )
    parser.add_argument(
        "--pq_compare_search_methods",
        action="store_true",
        help="Evaluate grid, Powell, and Newton on the same PQ-GradNorm tuning snapshot and log side-by-side results. "
             "The configured --pq_search_method still controls which result updates (p,q).",
    )
    parser.add_argument("--pq_powell_maxiter", type=int, default=25)
    parser.add_argument("--pq_powell_xtol", type=float, default=1e-3)
    parser.add_argument("--pq_powell_ftol", type=float, default=1e-3)
    parser.add_argument("--pq_newton_maxiter", type=int, default=10)
    parser.add_argument("--pq_newton_tol", type=float, default=1e-4)
    parser.add_argument("--pq_newton_damping", type=float, default=1e-4)

    parser.add_argument(
        "--pq_adam_lr",
        type=float,
        default=0.01,
        help="Learning rate for the Adam-based p/q controller when --pq_search_method=adam.",
    )
    parser.add_argument(
        "--pq_adam_beta1",
        type=float,
        default=0.9,
        help="Adam beta1 for the p/q controller.",
    )
    parser.add_argument(
        "--pq_adam_beta2",
        type=float,
        default=0.999,
        help="Adam beta2 for the p/q controller.",
    )
    parser.add_argument(
        "--pq_adam_eps",
        type=float,
        default=1e-8,
        help="Adam epsilon for the p/q controller.",
    )
    parser.add_argument(
        "--pq_optimize_rho",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="For online PQ-GradNorm controllers, optimize rho = q / d directly and convert back to q for G_reg.",
    )
    parser.add_argument(
        "--pq_densification_penalty_lambda",
        type=float,
        default=1,
        help="Coefficient lambda for the soft expected-densification penalty "
             "applied to rho = q / d in the full-batch PQ-GradNorm objective, "
             "where d = |E| / (|E| + |Ebar|) is the undirected graph density.",
    )
    parser.add_argument(
        "--pq_densification_penalty_type",
        type=str,
        default="quadratic",
        choices=["quadratic", "linear", "hinge"],
        help="Penalty form for full-batch PQ-GradNorm densification control: "
             "'quadratic' uses lambda * rho^2, 'linear' uses lambda * rho, and 'hinge' uses "
             "lambda * max(0, rho - rho0)^2 with rho = q / d.",
    )
    parser.add_argument(
        "--pq_densification_penalty_rho0",
        type=float,
        default=1.0,
        help="Hinge threshold rho0 used only when --pq_densification_penalty_type=hinge. "
             "Default is 1.0.",
    )
    parser.add_argument(
        "--pq_gd_lr",
        type=float,
        default=0.1,
        help="Learning rate for the projected GD p/q controller when --pq_search_method=gd.",
    )

    parser.add_argument("--pq_grid_size", type=int, default=5,
                        help="Grid size for raw (p,q) search. Use 11 for GIN; use 3-5 for GCN (costlier).")
    parser.add_argument(
        "--pq_surface_plot",
        action="store_true",
        help="Save objective heatmaps/contours over a frozen raw (p,q) snapshot with solver best points overlaid.",
    )
    parser.add_argument(
        "--pq_surface_animation",
        action="store_true",
        help="When --pq_surface_plot is enabled, also build a GIF over epochs if Pillow is available.",
    )
    parser.add_argument(
        "--pq_surface_grid_size",
        type=int,
        default=31,
        help="Grid resolution used for the optional objective surface export.",
    )
    parser.add_argument(
        "--pq_surface_every",
        type=int,
        default=1,
        help="Save an objective surface every k PQ update steps when --pq_surface_plot is enabled.",
    )
    parser.add_argument(
        "--pq_surface_run",
        type=int,
        default=1,
        help="1-based run index used for the optional objective surface export.",
    )

    parser.add_argument(
        "--pq_subset_nodes",
        type=int,
        default=0,
        help="Number of train nodes used to estimate GradNorm norms. "
             "Use 0 to use the full train_idx in the full-batch setting.",
    )
    parser.add_argument("--pq_ema", type=float, default=0.9,
                        help="EMA smoothing for p,q updates in search-based PQ tuning (grid/powell/newton only).")
    parser.add_argument("--pq_update_every", type=int, default=1, help="Update (p,q) every k epochs.")
    parser.add_argument("--pq_warmup_epochs", type=int, default=0, help="Number of initial epochs to skip pq updates.")
    parser.add_argument("--pq_eps", type=float, default=1e-3, help="Small epsilon for log/denominator stabilizers.")
    parser.add_argument("--pq_seed", type=int, default=0, help="Seed for selecting the subset of nodes for pq selection.")

    parser.add_argument(
        "--pq_data_anchor",
        type=str,
        default="clean",
        choices=["clean", "aug"],
        help="PQ-GradNorm: which data-loss gradient norm to anchor against. "
             "'clean' uses gradients from clean forward (p=q=0). "
             "'aug' uses gradients from one stochastic RADE-augmented forward using current (p,q) "
             "and the same mask_sharing policy (shared/layerwise).",
    )

    # -----------------------
    # Weights & Biases
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
