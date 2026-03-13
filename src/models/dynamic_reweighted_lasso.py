"""
Dynamic reweighted LASSO via outer IRL1 weight iteration.

Task:    T11 (owner: Yash Karecha)
Branch:  task/T11-dynamic-reweighted-lasso

Math reference:
    §2  (weighted LASSO objective)         — docs/math_spec.md
    §7  (IRL1 weight update)               — docs/math_spec.md
    §10 (convergence scope — outer loop)   — docs/math_spec.md
    §11 (stopping criterion)               — docs/math_spec.md

Algorithm — IRL1 proximal gradient (Candès, Wakin & Boyd, 2008):
    This module implements iteratively reweighted ℓ₁ minimisation (IRL1).
    No theoretical novelty is claimed for the reweighting scheme itself.
    The contribution is its integration into an ISTA/FISTA proximal-gradient
    framework and systematic empirical comparison (see docs/math_spec.md §12).

    Outer loop (k = 0, 1, …, max_outer_iter - 1):
        1. Call ISTA or FISTA with the current weight vector w^(k) (fixed inside the
           inner call), obtaining β^(k+1).
        2. Outer convergence check:
               ||β^(k+1) - β^(k)||_2 / max(1, ||β^(k)||_2) < inner_tol  → break
        3. Update weights:
               w^(k+1) = clip(compute_weights(β^(k+1), γ, ε), w_max)

    NOTE: The outer objective changes at every outer step because w changes.
    Standard ISTA/FISTA convergence guarantees (O(1/k) and O(1/k²) respectively)
    apply only within each fixed-weight inner subproblem. No global convergence
    guarantee is claimed for the outer loop. Outer convergence is an empirical
    property monitored via the returned traces.

Consumed by:
    T12  src/experiments/cross_validate.py        — Tenzin Kunga
         Sweeps lam, gamma, eps, max_outer_iter.
         Uses outer_objective_trace[-1] as the validation-objective proxy.
    T13  src/experiments/run_dynamic.py            — Tenzin Kunga
         Calls with best config; uses beta, outer traces, inner_iter_counts,
         total_runtime for the comparison table.
    T14  src/metrics/stability_metrics.py          — Both
         Uses beta (p,) across seeds → Jaccard feature-support stability.
    T16  src/visualization/convergence_plots.py    — Tenzin Kunga
         Uses outer_objective_trace and outer_sparsity_trace.
    T17  src/visualization/tradeoff_plots.py       — Tenzin Kunga
         Uses outer_sparsity_trace[-1] across λ values.
    T18  src/visualization/coefficient_plots.py    — Tenzin Kunga
         Uses weight_trace and beta across λ values.
    T20  Feature interpretation report             — Both
         Uses beta (p,) and weight_trace for domain interpretation.

I/O contract:
    run_dynamic_reweighted_lasso(
        X, y, lam, gamma, eps, w_max, alpha,
        max_outer_iter, inner_max_iter, inner_tol, solver
    )
        X             : np.ndarray, shape (n, p), float64 — standardised design matrix
        y             : np.ndarray, shape (n,),   float64 — target vector
        lam           : float >= 0    — regularisation strength
        gamma         : float > 0     — IRL1 weight sharpness exponent
        eps           : float > 0     — numerical stability constant
        w_max         : float > 0     — maximum allowed weight (clip_weights bound)
        alpha         : float > 0     — gradient step size; caller sets to 1/L
        max_outer_iter: int > 0       — maximum outer IRL1 iterations
        inner_max_iter: int > 0       — maximum inner (ISTA/FISTA) iterations per outer step
        inner_tol     : float > 0     — inner relative-change tolerance + outer stop tol
        solver        : str           — "ista" or "fista"

    Returns (6-tuple):
        beta                : np.ndarray, shape (p,)         — final coefficient vector
        outer_objective_trace: list[float], len <= max_outer_iter
                               Weighted objective (last inner value) per outer iter.
        outer_sparsity_trace : list[int],   len <= max_outer_iter
                               ||beta||_0 per outer iter.
        weight_trace         : list[np.ndarray], len <= max_outer_iter
                               Weight vector w^(k) used in each outer iter (before
                               updating for the next step); shape (p,) per entry.
        inner_iter_counts    : list[int], len <= max_outer_iter
                               Number of inner iterations used per outer step.
        total_runtime        : float
                               Cumulative wall-clock seconds across all outer steps.
"""

import numpy as np

from src.optim.fista import fista_solve
from src.optim.ista import ista_solve
from src.optim.reweight import clip_weights, compute_weights
from src.optim.stopping import has_converged


def run_dynamic_reweighted_lasso(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    gamma: float,
    eps: float,
    w_max: float,
    alpha: float,
    max_outer_iter: int,
    inner_max_iter: int,
    inner_tol: float,
    solver: str = "ista",
) -> tuple[
    np.ndarray,
    list[float],
    list[int],
    list[np.ndarray],
    list[int],
    float,
]:
    """Run the outer IRL1 loop with ISTA or FISTA as the inner solver.

    Implements iteratively reweighted ℓ₁ minimisation (Candès et al., 2008)
    inside a proximal-gradient framework.  At each outer iteration k the inner
    solver minimises the weighted LASSO subproblem with weights w^(k) held
    fixed, then weights are updated from the new coefficient vector.

    No global convergence theorem is claimed for the outer loop.  Convergence
    is monitored empirically via the returned traces (see math_spec.md §10).

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Standardised design matrix.
    y : np.ndarray, shape (n,)
        Target vector.
    lam : float
        Regularisation strength (>= 0).
    gamma : float
        IRL1 weight sharpness exponent (> 0).
    eps : float
        Numerical stability constant (> 0) preventing division by zero.
    w_max : float
        Upper bound for weight clipping; prevents extreme per-coordinate
        thresholds from destabilising the inner solver.
    alpha : float
        Gradient step size. Caller should set to ``1.0 / L`` where
        ``L = np.linalg.eigvalsh(X.T @ X).max() / n``.
    max_outer_iter : int
        Maximum number of outer IRL1 iterations.
    inner_max_iter : int
        Maximum number of inner proximal gradient iterations per outer step.
    inner_tol : float
        Convergence tolerance for the inner solver AND the outer stopping
        criterion (relative change in beta across consecutive outer steps).
    solver : str
        Inner solver: ``"ista"`` (O(1/k)) or ``"fista"`` (O(1/k²)).

    Returns
    -------
    beta : np.ndarray, shape (p,)
        Final coefficient vector after the outer loop terminates.
    outer_objective_trace : list[float]
        Last inner-solver objective value at each outer iteration.
        Length <= max_outer_iter.
    outer_sparsity_trace : list[int]
        ||beta||_0 at the end of each outer iteration.
        Length <= max_outer_iter.
    weight_trace : list[np.ndarray]
        Weight vector w^(k) used in each outer iteration (snapshot taken
        before calling the inner solver).  Each array has shape (p,).
        Length <= max_outer_iter.
    inner_iter_counts : list[int]
        Number of inner iterations consumed per outer step.
        Length <= max_outer_iter.
    total_runtime : float
        Cumulative wall-clock time in seconds across all inner solves.

    Raises
    ------
    ValueError
        If ``solver`` is not ``"ista"`` or ``"fista"``.
    """
    if solver not in {"ista", "fista"}:
        raise ValueError(
            f"solver must be 'ista' or 'fista', got {solver!r}"
        )

    _inner_solve = ista_solve if solver == "ista" else fista_solve

    p = X.shape[1]

    # Initialise β = 0; at β=0, w_j = 1/ε^γ for all j (math_spec §7)
    beta = np.zeros(p, dtype=np.float64)
    w = clip_weights(compute_weights(beta, gamma, eps), w_max)

    outer_objective_trace: list[float] = []
    outer_sparsity_trace: list[int] = []
    weight_trace: list[np.ndarray] = []
    inner_iter_counts: list[int] = []
    total_runtime: float = 0.0

    for _ in range(max_outer_iter):
        beta_prev = beta.copy()

        # Snapshot weight used in this outer step (before inner solve)
        weight_trace.append(w.copy())

        # Inner solve: minimise weighted LASSO with fixed w^(k)
        beta_new, obj_t, sp_t, rt = _inner_solve(
            X, y, lam, w, alpha, inner_max_iter, inner_tol
        )

        # Log outer-level traces
        outer_objective_trace.append(obj_t[-1] if obj_t else float("nan"))
        outer_sparsity_trace.append(sp_t[-1] if sp_t else int(np.count_nonzero(beta_new)))
        inner_iter_counts.append(len(obj_t))
        total_runtime += rt

        # Outer convergence check (math_spec §10 and §11)
        beta = beta_new
        if has_converged(beta_prev, beta, inner_tol):
            break

        # IRL1 weight update: w_j^(k+1) = 1 / (|β_j| + ε)^γ  (math_spec §7)
        w = clip_weights(compute_weights(beta, gamma, eps), w_max)

    return (
        beta,
        outer_objective_trace,
        outer_sparsity_trace,
        weight_trace,
        inner_iter_counts,
        total_runtime,
    )
