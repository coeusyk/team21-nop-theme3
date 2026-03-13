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

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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
    out_path = (out_dir / filename).resolve()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
