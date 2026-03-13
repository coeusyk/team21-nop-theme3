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

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


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
    out_path = (out_dir / filename).resolve()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
