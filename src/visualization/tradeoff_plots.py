"""
Sparsity–error tradeoff plot: validation MSE vs non-zero coefficients as λ varies.

Task:   T17 (owner: Yash Karecha)
Branch: task/T16-T18-visualization

Depends on:
    T9  src/models/lasso.py               — standard LASSO results
    T10 src/models/adaptive_lasso.py      — static adaptive LASSO results
    T11 src/models/dynamic_reweighted_lasso.py — dynamic IRL1-PG results
    T12 src/experiments/cross_validate.py — outputs/tables/validation_mse_grid.csv
                                             with columns: method, solver, lam,
                                             val_mse, nnz

Consumed by:
    T25 Results and Discussion — sparsity–error tradeoff figure in paper

I/O contract:
    plot_sparsity_error_tradeoff(results_df, output_dir, filename)
        results_df : pd.DataFrame
            Must contain columns: method (str), solver (str), lam (float),
            val_mse (float), nnz (int).
            Rows represent individual (method, solver, λ) evaluation points.
            Can be read directly from outputs/tables/validation_mse_grid.csv
            produced by T12.
        output_dir : str or Path — directory to write the PNG (created if absent)
        filename   : str         — filename including .png extension
    Returns:
        Path — resolved absolute path of the saved figure
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Canonical display labels for (method, solver) combinations.
_METHOD_LABELS: dict[tuple[str, str], str] = {
    ("lasso", "sklearn"): "Standard LASSO",
    ("lasso", ""): "Standard LASSO",
    ("adaptive_lasso", "ista"): "Static Adaptive LASSO (ISTA)",
    ("adaptive_lasso", "fista"): "Static Adaptive LASSO (FISTA)",
    ("dynamic_reweighted_lasso", "ista"): "Dynamic IRL1-PG (ISTA)",
    ("dynamic_reweighted_lasso", "fista"): "Dynamic IRL1-PG (FISTA)",
}

_MARKERS: list[str] = ["o", "s", "^", "D", "v", "P"]


def _make_label(method: str, solver: str) -> str:
    """Return a human-readable label for a (method, solver) pair."""
    key = (str(method).lower(), str(solver).lower())
    if key in _METHOD_LABELS:
        return _METHOD_LABELS[key]
    if solver:
        return f"{method} ({solver.upper()})"
    return method


def plot_sparsity_error_tradeoff(
    results_df: pd.DataFrame,
    output_dir: str | Path,
    filename: str,
) -> Path:
    """Plot validation MSE vs number of non-zero coefficients as λ varies.

    Each (method, solver) combination is shown as a separate coloured line.
    Points on each line correspond to different values of the regularisation
    strength λ; moving right along the line means decreasing λ (denser
    solutions); moving left means increasing λ (sparser solutions).  Arrow
    annotations indicating the λ direction are not added automatically because
    they depend on how the caller sorted the DataFrame.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain at minimum columns: ``method`` (str), ``solver`` (str),
        ``lam`` (float), ``val_mse`` (float), ``nnz`` (int).  Rows may mix
        multiple methods and solvers.  Typically loaded from
        ``outputs/tables/validation_mse_grid.csv`` produced by T12.
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
    required_cols = {"method", "solver", "lam", "val_mse", "nnz"}
    missing = required_cols - set(results_df.columns)
    if missing:
        raise ValueError(
            f"results_df is missing required columns: {sorted(missing)}"
        )

    df = results_df.copy()
    df["solver"] = df["solver"].fillna("").astype(str)
    df["method"] = df["method"].astype(str)

    # Group by (method, solver), sort each group by lam ascending so lines
    # trace from sparsest (high λ) to densest (low λ) left-to-right on nnz.
    groups = df.groupby(["method", "solver"], sort=True)
    n_groups = len(groups)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    palette = sns.color_palette("tab10", n_colors=max(n_groups, 1))

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

    for idx, ((method, solver), group) in enumerate(groups):
        group_sorted = group.sort_values("lam", ascending=True)
        label = _make_label(str(method), str(solver))
        marker = _MARKERS[idx % len(_MARKERS)]
        color = palette[idx]

        ax.plot(
            group_sorted["nnz"],
            group_sorted["val_mse"],
            label=label,
            color=color,
            marker=marker,
            markersize=5,
            linewidth=1.8,
            alpha=0.85,
        )

    ax.set_xlabel("Number of non-zero coefficients ($\\|\\hat{\\beta}\\|_0$)")
    ax.set_ylabel("Validation MSE")
    ax.set_title(
        "Sparsity–error tradeoff: validation MSE vs non-zero features ($\\lambda$ sweep)"
    )
    ax.legend(loc="best", framealpha=0.9, fontsize=10)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (out_dir / filename).resolve()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
