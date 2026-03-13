"""
Static adaptive LASSO model using IRL1 weights initialised from Ridge coefficients.

Task:    T10 (owner: Yash Karecha, implementing on behalf of Tenzin Kunga)
Branch:  task/T10-static-adaptive-lasso

Math reference:
    §6  (weighted soft-thresholding)       — docs/math_spec.md
    §7  (IRL1 weight formula)              — docs/math_spec.md

Algorithm (Zou, 2006 / Candès et al., 2008 applied statically):
    1. Accept Ridge-fitted coefficients β̂_ridge (pre-computed by caller, T8).
    2. Compute feature weights once:  w_j = 1/(|β̂_ridge_j| + ε)^γ   ∀ j
    3. Clip weights to prevent extreme thresholds.
    4. Run ISTA or FISTA with those FIXED weights for the entire inner solve.
    5. Return the same 4-tuple as ista_solve / fista_solve.

No theoretical novelty is claimed.  The IRL1 weight formula is from
Candès, Wakin, & Boyd (2008).

Consumed by:
    T12  src/experiments/cross_validate.py        — Tenzin Kunga
         Input : X (n,p), y (n,), lam from sweep, ridge_coef from Ridge CV
         Output: (beta (p,), obj_trace, sparsity_trace, runtime) used for val-MSE
    T13  src/experiments/run_baselines.py          — Tenzin Kunga
         Input : same + best config from T12
         Output: all 4 values → comparison table row
    T14  src/metrics/stability_metrics.py          — Both
         Input : beta (p,) across seeds → Jaccard stability
    T16  src/visualization/convergence_plots.py    — Tenzin Kunga
         Input : objective_trace (list[float]), sparsity_trace (list[int])
    T17  src/visualization/tradeoff_plots.py       — Tenzin Kunga
         Input : sparsity_trace[-1] (int) as non-zero count, lam sweep results
    T18  src/visualization/coefficient_plots.py    — Tenzin Kunga
         Input : beta (p,) across λ values

I/O contract:
    run_adaptive_lasso(X, y, lam, ridge_coef, gamma, eps, alpha, max_iter, tol, solver)
        X          : np.ndarray, shape (n, p), float64 — standardised design matrix
        y          : np.ndarray, shape (n,),   float64 — target vector
        lam        : float >= 0                — regularisation strength
        ridge_coef : np.ndarray, shape (p,), float64 — Ridge coefficient vector
                     used ONLY to compute static weights; not the initial β
        gamma      : float > 0                — weight sharpness exponent
        eps        : float > 0                — numerical stability constant
        alpha      : float > 0                — gradient step size (caller sets to 1/L)
        max_iter   : int > 0                  — maximum inner iterations
        tol        : float > 0                — relative-change convergence tolerance
        solver     : str, "ista" or "fista"   — inner solver selection
    Returns:
        beta            : np.ndarray, shape (p,)      — converged coefficient vector
        objective_trace : list[float], len <= max_iter — weighted objective per iter
        sparsity_trace  : list[int],   len <= max_iter — ||beta||_0 per iter
        runtime         : float                        — wall-clock seconds
"""

import numpy as np

from src.optim.fista import fista_solve
from src.optim.ista import ista_solve
from src.optim.reweight import clip_weights, compute_weights

# Large but finite ceiling; keeps α·λ·w_j from reaching numerical overflow.
_DEFAULT_W_MAX: float = 1.0e6


def run_adaptive_lasso(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    ridge_coef: np.ndarray,
    gamma: float,
    eps: float,
    alpha: float,
    max_iter: int,
    tol: float,
    solver: str = "ista",
) -> tuple[np.ndarray, list[float], list[int], float]:
    """Fit static adaptive LASSO using IRL1 weights from Ridge coefficients.

    Computes per-feature weights w_j = 1/(|beta_ridge_j| + eps)^gamma once,
    then runs a single call to ``ista_solve`` or ``fista_solve`` with those
    fixed weights.  The inner solve produces the same 4-tuple contract as the
    raw solvers so downstream tasks (T12, T13, T16, T17, T18) can use both
    interchangeably.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Standardised design matrix.
    y : np.ndarray, shape (n,)
        Target vector.
    lam : float
        Regularisation strength (>= 0).
    ridge_coef : np.ndarray, shape (p,)
        Ridge coefficient vector used to initialise weights.  Must have the
        same feature dimension p as X.  Not used as the starting point for β.
    gamma : float
        Weight sharpness exponent (> 0).  From default.yaml ``model.adaptive_lasso.gamma``.
    eps : float
        Numerical stability constant (> 0).  From default.yaml ``model.adaptive_lasso.epsilon``.
    alpha : float
        Gradient step size.  Caller should set to ``1.0 / L`` where
        ``L = np.linalg.eigvalsh(X.T @ X).max() / n``.
    max_iter : int
        Maximum number of inner proximal gradient iterations.
    tol : float
        Relative-change convergence tolerance.
    solver : str
        Inner solver: ``"ista"`` (default) or ``"fista"``.

    Returns
    -------
    beta : np.ndarray, shape (p,)
        Solution coefficient vector at termination.
    objective_trace : list[float]
        Weighted LASSO objective value after each inner iteration.
    sparsity_trace : list[int]
        Number of non-zero coefficients (||beta||_0) after each inner iteration.
    runtime : float
        Wall-clock time in seconds for the inner solve only.

    Raises
    ------
    ValueError
        If ``solver`` is not ``"ista"`` or ``"fista"``.
    """
    if solver not in {"ista", "fista"}:
        raise ValueError(
            f"solver must be 'ista' or 'fista', got {solver!r}"
        )

    # Step 1 — Compute static IRL1 weights from Ridge coefficients (math_spec §7)
    w = compute_weights(ridge_coef, gamma, eps)

    # Step 2 — Clip to prevent extreme per-coordinate thresholds
    w = clip_weights(w, _DEFAULT_W_MAX)

    # Step 3 — Run inner solver with fixed weights (math_spec §6)
    if solver == "ista":
        return ista_solve(X, y, lam, w, alpha, max_iter, tol)
    else:
        return fista_solve(X, y, lam, w, alpha, max_iter, tol)
