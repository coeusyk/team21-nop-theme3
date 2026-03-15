"""
Coefficient path plot: coefficient magnitude vs regularisation strength λ.

Task:   T18 (owner: Yash Karecha)
Branch: task/T16-T18-visualization

Depends on:
    T9  src/models/lasso.py               — coefficient vectors at λ sweep values
    T10 src/models/adaptive_lasso.py      — coefficient vectors at λ sweep values
    T11 src/models/dynamic_reweighted_lasso.py
                                          — coefficient vectors at λ sweep values

Consumed by:
    T25 Results and Discussion — coefficient path figure in paper

I/O contract:
    plot_coefficient_paths(coef_paths, lam_values, feature_names,
                           output_dir, filename, top_n=20)
        coef_paths   : dict[str, np.ndarray]
            Maps a human-readable method label to a coefficient matrix of
            shape (n_lam, p) where row i corresponds to lam_values[i].
            At minimum two entries are expected — one for a dynamic IRL1-PG
            variant and one for a static/LASSO baseline — to produce the
            side-by-side comparison described in T18.
        lam_values   : list[float]
            Sequence of λ values that index the rows of every coef matrix.
            Length must equal coef_paths[*].shape[0].
        feature_names: list[str]
            Feature name for every column index.
            Length must equal coef_paths[*].shape[1].
        output_dir   : str or Path — directory to write the PNG (created if absent)
        filename     : str         — filename including .png extension
        top_n        : int         — number of top features to display (default 20)
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


def _select_top_features(
    coef_paths: dict[str, np.ndarray],
    top_n: int,
) -> list[int]:
    """Return column indices of the top_n features by peak absolute coefficient.

    The peak is taken as ``max |β_j(λ)|`` across all λ values and across
    **all methods** in *coef_paths*.  This ensures the same features appear
    in every subplot for a fair visual comparison.

    Parameters
    ----------
    coef_paths : dict[str, np.ndarray]
        Maps method label to coefficient matrix of shape (n_lam, p).
    top_n : int
        Number of feature indices to return.

    Returns
    -------
    list[int]
        Column indices sorted by peak absolute coefficient value (descending),
        truncated to ``min(top_n, p)``.
    """
    if not coef_paths:
        return []

    first = next(iter(coef_paths.values()))
    p = first.shape[1]
    peak_abs = np.zeros(p, dtype=np.float64)
    for matrix in coef_paths.values():
        peak_abs = np.maximum(peak_abs, np.abs(matrix).max(axis=0))

    n_select = min(top_n, p)
    return list(np.argsort(peak_abs)[::-1][:n_select])


def plot_coefficient_paths(
    coef_paths: dict[str, np.ndarray],
    lam_values: list[float],
    feature_names: list[str],
    output_dir: str | Path,
    filename: str,
    top_n: int = 20,
) -> Path:
    """Plot coefficient magnitude vs λ for the top *top_n* features per method.

    Produces one subplot per method in *coef_paths*, arranged vertically.
    In each subplot, one line is drawn per feature from the top-N selection;
    the x-axis shows λ on a log scale (decreasing left-to-right so sparser
    solutions appear on the left).  This layout enables a direct visual
    comparison of how the dynamic IRL1-PG method zeroes out coefficients
    compared to a static baseline.

    Parameters
    ----------
    coef_paths : dict[str, np.ndarray]
        Maps a method label to a coefficient matrix of shape (n_lam, p).
        Row *i* is the coefficient vector obtained with ``lam_values[i]``.
    lam_values : list[float]
        Regularisation strength values corresponding to rows of each matrix.
        Will be displayed on a log-scaled x-axis.
    feature_names : list[str]
        One name per column (feature).  Length must equal ``p``.
    output_dir : str or Path
        Directory where the PNG will be saved.  Created automatically if it
        does not exist.
    filename : str
        Output filename including the ``.png`` extension.
    top_n : int
        Number of top features to display in each subplot.  Features are
        ranked by their peak absolute coefficient value across all methods
        and all λ values.  Default is 20.

    Returns
    -------
    Path
        Resolved absolute path of the saved figure file.

    Raises
    ------
    ValueError
        If *coef_paths* is empty, if the number of rows does not match
        ``len(lam_values)``, or if the number of columns does not match
        ``len(feature_names)``.
    """
    if not coef_paths:
        raise ValueError("coef_paths must contain at least one entry.")

    n_lam = len(lam_values)
    p = len(feature_names)

    for label, matrix in coef_paths.items():
        if matrix.shape[0] != n_lam:
            raise ValueError(
                f"coef_paths['{label}'] has {matrix.shape[0]} rows but "
                f"len(lam_values) = {n_lam}."
            )
        if matrix.shape[1] != p:
            raise ValueError(
                f"coef_paths['{label}'] has {matrix.shape[1]} columns but "
                f"len(feature_names) = {p}."
            )

    lam_arr = np.asarray(lam_values, dtype=np.float64)

    top_indices = _select_top_features(coef_paths, top_n)
    top_names = [feature_names[i] for i in top_indices]

    n_methods = len(coef_paths)
    # Use a high-contrast palette; one colour per feature (shared across subplots)
    feature_palette = sns.color_palette("tab20", n_colors=len(top_indices))

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

    fig, axes = plt.subplots(
        n_methods,
        1,
        figsize=(12, 5 * n_methods),
        constrained_layout=True,
        squeeze=False,
    )

    for row_idx, (label, matrix) in enumerate(coef_paths.items()):
        ax = axes[row_idx, 0]

        for feat_idx, col_idx in enumerate(top_indices):
            coef_path = matrix[:, col_idx]
            ax.plot(
                lam_arr,
                coef_path,
                color=feature_palette[feat_idx],
                linewidth=1.4,
                alpha=0.85,
                label=top_names[feat_idx],
            )

        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xscale("log")
        # Flip x-axis so that increasing λ (more sparse) goes right to left,
        # matching the convention used in sklearn's lasso_path visualisation.
        ax.invert_xaxis()
        ax.set_xlabel("Regularisation strength λ (log scale, decreasing →)")
        ax.set_ylabel("Coefficient value")
        ax.set_title(f"Coefficient path — {label} (top {len(top_indices)} features)")
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=8,
            framealpha=0.9,
            ncol=1,
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


def _build_default_coef_paths(
    config_path: str,
    dynamic_config_path: str,
    ista_config_path: str,
    fista_config_path: str,
    best_config_path: str,
) -> tuple[dict[str, np.ndarray], list[float], list[str]]:
    """Recompute coefficient paths over lambda sweep from current configs."""
    cfg = _load_yaml(config_path)
    dynamic_cfg = _load_yaml(dynamic_config_path)
    ista_cfg = _load_yaml(ista_config_path)
    fista_cfg = _load_yaml(fista_config_path)
    best_cfg = _load_yaml(best_config_path)

    data_cfg = cfg["data"]
    optim_cfg = cfg["optim"]
    ridge_cfg = cfg["model"]["ridge"]
    search_cfg = dynamic_cfg["search"]
    lam_values = [float(v) for v in search_cfg["lam_values"]]

    set_seed(int(data_cfg["seed"]))
    X_train, _X_val, _X_test, y_train, _y_val, _y_test, feature_names = build_dataset(
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

    coef_paths: dict[str, np.ndarray] = {}
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

        adaptive_rows: list[np.ndarray] = []
        dynamic_rows: list[np.ndarray] = []

        for lam in lam_values:
            beta_a, _obj_a, _sp_a, _rt_a = run_adaptive_lasso(
                X=X_train,
                y=y_train_c,
                lam=float(lam),
                ridge_coef=ridge_coef,
                gamma=float(adaptive_best["gamma"]),
                eps=float(adaptive_best["eps"]),
                alpha=alpha,
                max_iter=max_iter,
                tol=tol,
                solver=solver,
            )
            adaptive_rows.append(beta_a)

            beta_d, _obj_d, _sp_d, _w_d, _ic_d, _rt_d = run_dynamic_reweighted_lasso(
                X=X_train,
                y=y_train_c,
                lam=float(lam),
                gamma=float(dynamic_best["gamma"]),
                eps=float(dynamic_best["eps"]),
                w_max=float(dynamic_cfg["w_max"]),
                alpha=alpha,
                max_outer_iter=int(dynamic_best["max_outer_iter"]),
                inner_max_iter=int(dynamic_cfg.get("inner_max_iter", max_iter)),
                inner_tol=float(dynamic_cfg.get("inner_tol", tol)),
                solver=solver,
            )
            dynamic_rows.append(beta_d)

        coef_paths[f"Adaptive LASSO ({solver.upper()})"] = np.vstack(adaptive_rows)
        coef_paths[f"Dynamic IRL1 ({solver.upper()})"] = np.vstack(dynamic_rows)

    if not coef_paths:
        raise FileNotFoundError(
            "No adaptive/dynamic best-config entries found. Run T12 first to generate outputs/logs/best_configs.yaml."
        )
    return coef_paths, lam_values, [str(name) for name in feature_names]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for T18 coefficient-path figure generation."""
    parser = argparse.ArgumentParser(description="Generate T18 coefficient-path figure")
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
    parser.add_argument("--filename", type=str, default="t18_coefficient_paths.png")
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    """CLI entry point for regenerating the T18 coefficient-path figure."""
    args = _parse_args(argv)
    cfg = _load_yaml(args.config)
    output_dir = args.output_dir or str(cfg["outputs"]["figures"])
    coef_paths, lam_values, feature_names = _build_default_coef_paths(
        config_path=args.config,
        dynamic_config_path=args.dynamic_config,
        ista_config_path=args.ista_config,
        fista_config_path=args.fista_config,
        best_config_path=args.best_config,
    )
    out_path = plot_coefficient_paths(
        coef_paths=coef_paths,
        lam_values=lam_values,
        feature_names=feature_names,
        output_dir=output_dir,
        filename=args.filename,
        top_n=int(args.top_n),
    )
    print(out_path)
    return out_path


if __name__ == "__main__":
    main()
