from __future__ import annotations

import argparse

from .mb_models import MPNNsMB


def str2bool(value):
    if isinstance(value, bool):
        return value

    value_norm = str(value).strip().lower()
    if value_norm in {"1", "true", "t", "yes", "y"}:
        return True
    if value_norm in {"0", "false", "f", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


MINI_BATCH_AUTO_DEFAULTS = {
    ("flickr", "gcn"): {
        "hidden_channels": 256,
        "epochs": 1000,
        "p": 0.0,
        "q": 0.0,
        "pq_grid_size": 5,
    },
    ("flickr", "gin"): {
        "hidden_channels": 256,
        "epochs": 1000,
        "p": 0.2,
        "q": 0.00001436,
        "pq_grid_size": 11,
    },
    ("ogbn-arxiv", "gcn"): {
        "hidden_channels": 256,
        "epochs": 2000,
        "p": 0.0,
        "q": 0.0,
        "pq_grid_size": 5,
    },
    ("ogbn-arxiv", "gin"): {
        "hidden_channels": 256,
        "epochs": 2000,
        "p": 0.2,
        "q": 0.000013446,
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


def apply_mini_batch_auto_defaults(args, explicit_arg_names=None):
    explicit_arg_names = explicit_arg_names or set()

    dataset = str(getattr(args, "dataset", "")).lower().strip()
    gnn = str(getattr(args, "gnn", "")).lower().strip()
    preset_key = (dataset, gnn)
    preset = MINI_BATCH_AUTO_DEFAULTS.get(preset_key)

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
    model = MPNNsMB(
        in_channels=in_dim,
        hidden_channels=args.hidden_channels,
        out_channels=num_classes,
        num_nodes=num_nodes,
        local_layers=args.local_layers,
        dropout=args.dropout,
        pre_linear=args.pre_linear,
        res=args.res,
        ln=args.ln,
        bn=args.bn,
        jk=args.jk,
        gnn=args.gnn,
        ep_correction=args.ep_correction,
        ep_expectation_mode=args.ep_expectation_mode,
        ep_emp_average_mode=ep_emp_average_mode,
        ep_emp_beta=args.ep_emp_beta,
        ep_emp_eps=args.ep_emp_eps,
        rade_variant=args.rade_variant,
        linear=bool(getattr(args, "linear", False)),
        aug_tech=str(getattr(args, "aug_tech", "none")),
        mb_rade_mode=str(getattr(args, "mb_rade_mode", "local")),
    ).to(device)
    return model


def parser_add_main_args(parser):
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-arxiv",
        choices=["flickr", "ogbn-arxiv"],
        help="Dataset name",
    )
    parser.add_argument("--data_dir", type=str, default="./data/", help="Root directory")
    parser.add_argument(
        "--remove_citeseer_isolated_nodes",
        action="store_true",
        help="Remove isolated nodes when loading CiteSeer. Ignored for other datasets.",
    )

    parser.add_argument("--device", type=int, default=1, help="GPU id (default: 0)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--epochs", type=int, default=500, help="number of training epochs")
    parser.add_argument("--runs", type=int, default=5, help="number of distinct runs")

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

    parser.add_argument("--metric", type=str, default="acc", choices=["acc"], help="evaluation metric")

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="Number of seed nodes per mini-batch.",
    )
    parser.add_argument(
        "--mb_shuffle",
        type=bool,
        default=True,
        help="Shuffle seed nodes each epoch before sampling mini-batches.",
    )
    parser.add_argument("--mb_num_workers", type=int, default=4, help="NeighborLoader workers.")
    parser.add_argument("--mb_pin_memory", action="store_true", help="Pin memory for NeighborLoader.")
    parser.add_argument(
        "--full_eval_cpu",
        action="store_true",
        help="Evaluate on CPU full graph to avoid GPU OOM during evaluation.",
    )
    parser.add_argument(
        "--mb_rade_mode",
        type=str,
        default="local",
        choices=["disabled", "local"],
        help="Mini-batch RADE semantics. 'local' applies RADE within the sampled batch graph.",
    )

    parser.add_argument("--model", type=str, default="MPNN", choices=["MPNN"])
    parser.add_argument(
        "--gnn",
        type=str,
        default="gcn",
        choices=["gcn", "gin", "sgc", "gin-linear"],
        help="Backbone: gcn/gin are standard. sgc routes to linear-GCN. gin-linear routes to linear-GIN.",
    )
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--local_layers", type=int, default=2)
    parser.add_argument("--pre_linear", type=bool, default=True)
    parser.add_argument("--res", action="store_true")
    parser.add_argument("--ln", action="store_true")
    parser.add_argument("--bn", action="store_true")
    parser.add_argument("--jk", action="store_true")
    parser.add_argument(
        "--linear",
        type=bool,
        default=False,
        help="Use strictly linear model: disables ReLU/Dropout/BN/LN and makes GIN MLP linear.",
    )

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="Early stopping patience in epochs based on validation accuracy. Set <=0 to disable.",
    )

    parser.add_argument(
        "--aug_tech",
        type=str,
        default="rade",
        choices=["rade", "dropmessage", "dropnode", "none"],
        help="Which augmentation family to use.",
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

    parser.add_argument(
        "--aug_mode",
        type=str,
        default="both",
        choices=["none", "drop", "add", "both"],
        help="Graph augmentation mode per epoch.",
    )
    parser.add_argument("--p", type=float, default=0.2, help="Edge drop probability for edges (i,j) in E.")
    parser.add_argument("--q", type=float, default=0.000013446, help="Non-edge add probability.")

    parser.add_argument(
        "--ep_correction",
        type=bool,
        default=True,
        help="Use expectation-preserving aggregation correction for RADE.",
    )
    parser.add_argument(
        "--ep_expectation_mode",
        type=str,
        default="analytic",
        choices=["analytic", "empirical_ema"],
        help="How to obtain the expectation term used in EP correction.",
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
        help="RADE only: shared or layerwise edge masks.",
    )
    parser.add_argument(
        "--rade_variant",
        type=str,
        default="rade-of",
        choices=["rade-of", "rade-ofs"],
        help="RADE variant: RADE-OF or RADE-OFS.",
    )

    parser.add_argument("--pq_gradnorm", type=bool, default=True, help="Enable epoch-wise GradNorm matching.")
    parser.add_argument(
        "--pq_search_method",
        type=str,
        default="adam",
        choices=["grid", "powell", "newton", "adam"],
        help="How to update (p,q). "
             "'grid'/'powell'/'newton' use the existing mini-batch search rules. "
             "'adam' uses a stateful online controller and one Adam step after each mini-batch PQ update.",
    )
    parser.add_argument(
        "--pq_compare_search_methods",
        action="store_true",
        help="Evaluate grid, Powell, and Newton on the same PQ-GradNorm snapshot. "
             "Not supported when --pq_search_method=adam.",
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
        default=0.1,
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
        help="For Adam PQ-GradNorm, optimize rho = q / d directly and convert back to q for G_reg.",
    )
    parser.add_argument(
        "--pq_densification_penalty_lambda",
        type=float,
        default=1,
        help="Coefficient lambda for the soft expected-densification penalty applied to rho = q / d.",
    )
    parser.add_argument(
        "--pq_densification_penalty_type",
        type=str,
        default="quadratic",
        choices=["quadratic", "linear", "hinge"],
        help="Penalty form for PQ-GradNorm densification control.",
    )
    parser.add_argument(
        "--pq_densification_penalty_rho0",
        type=float,
        default=1.0,
        help="Hinge threshold rho0 used only when --pq_densification_penalty_type=hinge.",
    )
    parser.add_argument(
        "--pq_epoch_objective_mode",
        type=str,
        default="batch_solutions",
        choices=["batch_solutions", "joint_objective"],
        help="Mini-batch PQ update rule. "
             "'batch_solutions' solves each batch separately then averages p,q over the epoch. "
             "'joint_objective' solves once on the weighted mean of per-batch squared log-gap objectives.",
    )
    parser.add_argument(
        "--pq_batch_weighting",
        type=str,
        default="uniform",
        choices=["node_count", "uniform"],
        help="How mini-batches are weighted inside epoch-level PQ updates.",
    )
    parser.add_argument(
        "--pq_grid_size",
        type=int,
        default=11,
        help="Grid size for (p,q) search. Use 11 for GIN; use 3-5 for GCN.",
    )
    parser.add_argument(
        "--pq_surface_plot",
        action="store_true",
        help="Save objective heatmaps/contours over a frozen batch-aggregated (p,q) snapshot.",
    )
    parser.add_argument(
        "--pq_surface_animation",
        action="store_true",
        help="When --pq_surface_plot is enabled, also build a GIF over epochs if Pillow is available.",
    )
    parser.add_argument("--pq_surface_grid_size", type=int, default=31)
    parser.add_argument("--pq_surface_every", type=int, default=1)
    parser.add_argument("--pq_surface_run", type=int, default=1)
    parser.add_argument(
        "--pq_store_batches",
        type=int,
        default=0,
        help="Optionally cache up to this many train mini-batches for epoch PQ tuning. Stored on CPU.",
    )
    parser.add_argument(
        "--pq_max_batches_for_tuning",
        type=int,
        default=0,
        help="Cap the number of mini-batches used by PQ tuning per epoch. 0 means use all available.",
    )
    parser.add_argument("--p_max", type=float, default=0.6, help="Upper bound for p in non-Adam mini-batch PQ search.")
    parser.add_argument("--q_max", type=float, default=1e-3, help="Upper bound for q in non-Adam mini-batch PQ search.")
    parser.add_argument(
        "--pq_subset_nodes",
        type=int,
        default=0,
        help="Number of train nodes used to estimate GradNorm norms within a batch.",
    )
    parser.add_argument(
        "--pq_ema",
        type=float,
        default=0.9,
        help="EMA smoothing for p,q updates in search-based PQ tuning (grid/powell/newton only).",
    )
    parser.add_argument("--pq_update_every", type=int, default=1, help="Update (p,q) every k epochs.")
    parser.add_argument("--pq_warmup_epochs", type=int, default=1, help="Number of initial epochs to skip pq updates.")
    parser.add_argument("--pq_eps", type=float, default=1e-12, help="Small epsilon for log/denominator stabilizers.")
    parser.add_argument("--pq_seed", type=int, default=0, help="Seed for selecting the subset of nodes for pq selection.")
    parser.add_argument(
        "--pq_data_anchor",
        type=str,
        default="clean",
        choices=["clean", "aug"],
        help="PQ-GradNorm anchor for the data-loss gradient norm.",
    )

    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="rade-nc-mb")
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

    parser.add_argument("--display_step", type=int, default=1)
    parser.add_argument("--save_model", action="store_true", help="whether to save model (logger.py controls path)")
