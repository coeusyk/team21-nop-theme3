"""
ISTA (Iterative Shrinkage-Thresholding Algorithm) solver for weighted LASSO.

Math reference: docs/math_spec.md §3 (gradient), §6 (weighted soft-threshold),
                §11 (stopping criterion).
Algorithm: proximal gradient descent with fixed weights per call.
           For the plain LASSO special case set w = np.ones(p).

Consumed by:
    T10 (src/models/adaptive_lasso.py)        — Tenzin Kunga
        Input:  X (n,p), y (n,), w fixed from Ridge, alpha=1/L, lam from CV
        Output: beta (p,), objective_trace, sparsity_trace, runtime
    T11 (src/models/dynamic_reweighted_lasso.py) — Yash Karecha
        Input:  X (n,p), y (n,), w updated each outer IRL1 step, alpha=1/L
        Output: beta (p,), objective_trace, sparsity_trace, runtime (per outer step)
    T13 (src/experiments/run_baselines.py,
         src/experiments/run_dynamic.py)        — Tenzin Kunga
        Input:  configs from configs/ista.yaml / configs/dynamic_reweight.yaml
        Output: all four return values used for comparison table + logs

I/O contract:
    ista_solve(X, y, lam, w, alpha, max_iter, tol)
        X        : np.ndarray, shape (n, p), float64 — standardised design matrix
        y        : np.ndarray, shape (n,),   float64 — target vector
        lam      : float >= 0    — regularisation strength
        w        : np.ndarray, shape (p,), float64 >= 0 — per-coordinate weights
        alpha    : float > 0     — gradient step size; caller sets to 1/L or smaller
        max_iter : int > 0       — maximum inner iterations
        tol      : float > 0     — relative-change convergence tolerance
    Returns:
        beta            : np.ndarray, shape (p,)     — converged coefficient vector
        objective_trace : list[float], len <= max_iter — weighted objective per iter
        sparsity_trace  : list[int],   len <= max_iter — ||beta||_0 per iter
        runtime         : float — wall-clock seconds for the entire solve
"""

import time

import numpy as np

from src.optim.objective import compute_loss
from src.optim.prox_ops import weighted_soft_threshold
from src.optim.stopping import has_converged


def ista_solve(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    w: np.ndarray,
    alpha: float,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, list[float], list[int], float]:
    """Run ISTA for the weighted LASSO subproblem with fixed weights w.

    Performs proximal gradient descent:
        z       = beta - alpha * grad_f(beta)
        beta    = weighted_soft_threshold(z, alpha, lam, w)
    until has_converged or max_iter is reached.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Standardised design matrix.
    y : np.ndarray, shape (n,)
        Target vector.
    lam : float
        Regularisation strength (>= 0).
    w : np.ndarray, shape (p,)
        Per-coordinate non-negative weights (fixed for this call).
    alpha : float
        Gradient step size. Must satisfy 0 < alpha <= 1/L where
        L = lambda_max(X.T @ X) / n.
    max_iter : int
        Maximum number of proximal gradient iterations.
    tol : float
        Relative-change convergence tolerance.

    Returns
    -------
    beta : np.ndarray, shape (p,)
        Solution coefficient vector at termination.
    objective_trace : list[float]
        Weighted LASSO objective value after each iteration.
    sparsity_trace : list[int]
        Number of non-zero coefficients (||beta||_0) after each iteration.
    runtime : float
        Wall-clock time in seconds for the entire solve.
    """
    n, p = X.shape
    beta = np.zeros(p, dtype=np.float64)

    objective_trace: list[float] = []
    sparsity_trace: list[int] = []

    t_start = time.perf_counter()

    for _ in range(max_iter):
        beta_prev = beta.copy()

        # Gradient of smooth part: ∇f(β) = X.T @ (X @ β - y) / n  (math_spec §3)
        grad = X.T @ (X @ beta - y) / n

        # Gradient step
        z = beta - alpha * grad

        # Proximal step: weighted soft-thresholding  (math_spec §6)
        beta = weighted_soft_threshold(z, alpha, lam, w)

        # Log objective and sparsity after update
        objective_trace.append(compute_loss(X, y, beta, lam, w))
        sparsity_trace.append(int(np.count_nonzero(beta)))

        # Stopping criterion  (math_spec §11)
        if has_converged(beta_prev, beta, tol):
            break

    runtime = time.perf_counter() - t_start

    return beta, objective_trace, sparsity_trace, runtime
