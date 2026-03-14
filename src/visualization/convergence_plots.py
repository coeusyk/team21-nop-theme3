"""
Convergence plots: objective value and sparsity vs iteration.

Task:   T16 (owner: Yash Karecha)
Branch: task/T16-T18-visualization

Depends on:
    T5  src/optim/ista.py           — objective_trace, sparsity_trace per inner iter
    T6  src/optim/fista.py          — objective_trace, sparsity_trace per inner iter
    T11 src/models/dynamic_reweighted_lasso.py
                                    — outer_objective_trace, outer_sparsity_trace
                                      per outer IRL1 iteration

Consumed by:
    T25 Results and Discussion — convergence figure included in paper

I/O contract:
    plot_convergence(traces_dict, output_dir, filename)
        traces_dict : dict[str, dict[str, list]]
            Maps a human-readable method label to a dict with two keys:
                "objective_trace" : list[float]  — objective value per iteration
                "sparsity_trace"  : list[int]    — ||β||_0 per iteration
            For ISTA/FISTA (T5/T6): one entry per inner proximal step.
            For dynamic variants (T11): one entry per outer IRL1 step.
            Any number of methods may be passed; all are plotted on the same axes.
        output_dir : str or Path — directory to write the PNG (created if absent)
        filename   : str         — filename including .png extension
    Returns:
        Path — resolved absolute path of the saved figure
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from sklearn.linear_model import RidgeCV

from src.data.split import build_dataset
from src.models.adaptive_lasso import run_adaptive_lasso
from src.models.dynamic_reweighted_lasso import run_dynamic_reweighted_lasso
from src.utils.seeds import set_seed


def plot_convergence(
    traces_dict: dict[str, dict[str, list]],
    output_dir: str | Path,
    filename: str,
) -> Path:
    """Plot objective value and sparsity vs iteration for multiple solvers.

    Produces a two-panel figure:
      - Top panel:    weighted objective value vs iteration index.
      - Bottom panel: ||β||_0 (number of non-zero coefficients) vs iteration index.

    Both panels share the same x-axis.  Each method in *traces_dict* is drawn
    as a distinct coloured line.  The objective y-axis uses a log scale when
    all values are strictly positive and the dynamic range exceeds one decade.

    Parameters
    ----------
    traces_dict : dict[str, dict[str, list]]
        Maps a method label string to a dict with keys
        ``"objective_trace"`` (list of float) and
        ``"sparsity_trace"`` (list of int).  The lists may have different
        lengths across methods (e.g. inner vs outer iterations).
    output_dir : str or Path
        Directory where the PNG will be saved.  Created automatically if it
        does not exist.
    filename : str
        Output filename including the ``.png`` extension.

    Returns
    -------
    Path
        Resolved absolute path of the saved figure file.
    """
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    palette = sns.color_palette("tab10", n_colors=max(len(traces_dict), 1))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=False,
        constrained_layout=True,
    )
    ax_obj, ax_spar = axes

    for idx, (label, traces) in enumerate(traces_dict.items()):
        obj_trace = traces.get("objective_trace", [])
        spar_trace = traces.get("sparsity_trace", [])
        color = palette[idx]

        if obj_trace:
            ax_obj.plot(
                range(1, len(obj_trace) + 1),
                obj_trace,
                label=label,
                color=color,
                linewidth=1.8,
                marker="o",
                markersize=3,
                markevery=max(1, len(obj_trace) // 20),
            )
        if spar_trace:
            ax_spar.plot(
                range(1, len(spar_trace) + 1),
                spar_trace,
                label=label,
                color=color,
                linewidth=1.8,
                marker="s",
                markersize=3,
                markevery=max(1, len(spar_trace) // 20),
            )

    # Apply log scale to objective axis if values are positive and span > 10x
    all_obj_vals = [
        v
        for traces in traces_dict.values()
        for v in traces.get("objective_trace", [])
    ]
    if all_obj_vals and min(all_obj_vals) > 0:
        dynamic_range = max(all_obj_vals) / min(all_obj_vals)
        if dynamic_range > 10.0:
            ax_obj.set_yscale("log")

    ax_obj.set_ylabel("Weighted objective value")
    ax_obj.set_title("Convergence: objective value vs iteration")
    ax_obj.legend(
        loc="best",
        framealpha=0.9,
        fontsize=10,
    )

    ax_spar.set_xlabel("Iteration")
    ax_spar.set_ylabel(r"$\|\hat{\beta}\|_0$ (non-zero coefficients)")
    ax_spar.set_title(r"Sparsity $\|\hat{\beta}^k\|_0$ vs iteration")
    ax_spar.legend(
        loc="best",
        framealpha=0.9,
        fontsize=10,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _load_yaml(path: str) -> dict:
    """Load YAML config into a dictionary."""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected mapping YAML at {path}")
    return cfg


def _resolve_alpha(X_train: np.ndarray, alpha_cfg: float | None) -> float:
    """Resolve alpha from config or compute as inverse Lipschitz constant."""
    if alpha_cfg is not None:
        return float(alpha_cfg)
    n = X_train.shape[0]
    lipschitz = float(np.linalg.eigvalsh(X_train.T @ X_train).max() / n)
    if lipschitz <= 0.0:
        raise ValueError("Computed non-positive Lipschitz constant; cannot set alpha")
    return 1.0 / lipschitz


def _solver_optim_params(
    solver: str,
    optim_cfg: dict,
    ista_cfg: dict,
    fista_cfg: dict,
    X_train: np.ndarray,
) -> tuple[float, int, float]:
    """Resolve alpha, max_iter, and tol for a selected solver."""
    solver_cfg = ista_cfg if solver == "ista" else fista_cfg
    alpha = _resolve_alpha(X_train, solver_cfg.get("alpha", optim_cfg.get("alpha")))
    max_iter = int(solver_cfg.get("max_iter", optim_cfg["max_iter"]))
    tol = float(solver_cfg.get("tol", optim_cfg["tol"]))
    return alpha, max_iter, tol


def _build_default_traces(
    config_path: str,
    dynamic_config_path: str,
    ista_config_path: str,
    fista_config_path: str,
    best_config_path: str,
) -> dict[str, dict[str, list]]:
    """Recompute convergence traces from current best configs."""
    cfg = _load_yaml(config_path)
    dynamic_cfg = _load_yaml(dynamic_config_path)
    ista_cfg = _load_yaml(ista_config_path)
    fista_cfg = _load_yaml(fista_config_path)
    best_cfg = _load_yaml(best_config_path)

    data_cfg = cfg["data"]
    optim_cfg = cfg["optim"]
    ridge_cfg = cfg["model"]["ridge"]

    set_seed(int(data_cfg["seed"]))
    X_train, _X_val, _X_test, y_train, _y_val, _y_test, _feature_names = build_dataset(
        raw_path=data_cfg["raw_path"],
        target_col=data_cfg["target_col"],
        val_size=float(data_cfg["val_size"]),
        test_size=float(data_cfg["test_size"]),
        seed=int(data_cfg["seed"]),
        numeric_impute_strategy=data_cfg.get("numeric_impute_strategy", "median"),
        categorical_impute_strategy=data_cfg.get(
            "categorical_impute_strategy", "most_frequent"
        ),
        drop_cols=data_cfg.get("drop_cols"),
    )

    # Match custom-solver training setup used by experiments.
    y_mean = float(y_train.mean())
    y_train_c = y_train - y_mean

    ridge_model = RidgeCV(alphas=ridge_cfg["alphas"])
    ridge_model.fit(X_train, y_train)
    ridge_coef = ridge_model.coef_.astype(np.float64)

    traces: dict[str, dict[str, list]] = {}

    for solver in ["ista", "fista"]:
        adaptive_key = f"adaptive_lasso_{solver}"
        dynamic_key = f"dynamic_reweighted_lasso_{solver}"
        if adaptive_key not in best_cfg or dynamic_key not in best_cfg:
            continue

        adaptive_best = best_cfg[adaptive_key]
        dynamic_best = best_cfg[dynamic_key]
        alpha, max_iter, tol = _solver_optim_params(
            solver=solver,
            optim_cfg=optim_cfg,
            ista_cfg=ista_cfg,
            fista_cfg=fista_cfg,
            X_train=X_train,
        )

        _beta_a, obj_a, sp_a, _rt_a = run_adaptive_lasso(
            X=X_train,
            y=y_train_c,
            lam=float(adaptive_best["lam"]),
            ridge_coef=ridge_coef,
            gamma=float(adaptive_best["gamma"]),
            eps=float(adaptive_best["eps"]),
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            solver=solver,
        )
        traces[f"Adaptive LASSO ({solver.upper()})"] = {
            "objective_trace": obj_a,
            "sparsity_trace": sp_a,
        }

        _beta_d, obj_d, sp_d, _w_d, _ic_d, _rt_d = run_dynamic_reweighted_lasso(
            X=X_train,
            y=y_train_c,
            lam=float(dynamic_best["lam"]),
            gamma=float(dynamic_best["gamma"]),
            eps=float(dynamic_best["eps"]),
            w_max=float(dynamic_cfg["w_max"]),
            alpha=alpha,
            max_outer_iter=int(dynamic_best["max_outer_iter"]),
            inner_max_iter=int(dynamic_cfg.get("inner_max_iter", max_iter)),
            inner_tol=float(dynamic_cfg.get("inner_tol", tol)),
            solver=solver,
        )
        traces[f"Dynamic IRL1 ({solver.upper()})"] = {
            "objective_trace": obj_d,
            "sparsity_trace": sp_d,
        }

    if not traces:
        raise FileNotFoundError(
            "No adaptive/dynamic best-config entries found. Run T12 first to generate outputs/logs/best_configs.yaml."
        )
    return traces


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for T16 convergence plot generation."""
    parser = argparse.ArgumentParser(description="Generate T16 convergence figure")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--dynamic-config", type=str, default="configs/dynamic_reweight.yaml"
    )
    parser.add_argument("--ista-config", type=str, default="configs/ista.yaml")
    parser.add_argument("--fista-config", type=str, default="configs/fista.yaml")
    parser.add_argument(
        "--best-config", type=str, default="outputs/logs/best_configs.yaml"
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--filename", type=str, default="t16_convergence.png")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    """CLI entry point for regenerating the T16 convergence figure."""
    args = _parse_args(argv)
    cfg = _load_yaml(args.config)
    output_dir = args.output_dir or str(cfg["outputs"]["figures"])
    traces = _build_default_traces(
        config_path=args.config,
        dynamic_config_path=args.dynamic_config,
        ista_config_path=args.ista_config,
        fista_config_path=args.fista_config,
        best_config_path=args.best_config,
    )
    out_path = plot_convergence(
        traces_dict=traces,
        output_dir=output_dir,
        filename=args.filename,
    )
    print(out_path)
    return out_path


if __name__ == "__main__":
    main()
