"""
FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) solver for weighted LASSO.

Math reference: docs/math_spec.md §3 (gradient), §6 (weighted soft-threshold),
                §8 (FISTA momentum update), §11 (stopping criterion).
Algorithm: accelerated proximal gradient descent (Beck & Teboulle, 2009) with
           fixed weights per call.  Achieves O(1/k²) convergence rate vs
           O(1/k) for ISTA.  Note: the objective trace is NOT guaranteed
           monotone; FISTA trades monotone descent for faster convergence rate.
           For the plain LASSO special case set w = np.ones(p).

Consumed by:
    T10 (src/models/adaptive_lasso.py)           — Tenzin Kunga
        Input:  X (n,p), y (n,), w fixed from Ridge, alpha=1/L, lam from CV
        Output: beta (p,), objective_trace, sparsity_trace, runtime
    T11 (src/models/dynamic_reweighted_lasso.py) — Yash Karecha
        Input:  X (n,p), y (n,), w updated each outer IRL1 step, alpha=1/L
        Output: beta (p,), objective_trace, sparsity_trace, runtime (per outer step)
    T12 (src/experiments/cross_validate.py)      — Tenzin Kunga
        Input:  configs from configs/fista.yaml / configs/dynamic_reweight.yaml
        Output: objective_trace length used as iteration-count metric
    T13 (src/experiments/run_baselines.py,
         src/experiments/run_dynamic.py)          — Tenzin Kunga
        Input:  configs from configs/fista.yaml / configs/dynamic_reweight.yaml
        Output: all four return values used for comparison table + logs
    T16 (src/visualization/convergence_plots.py) — Tenzin Kunga
        Input:  objective_trace (list[float]), sparsity_trace (list[int])
        Output: convergence and sparsity subplot for FISTA and dynamic-FISTA

I/O contract:
    fista_solve(X, y, lam, w, alpha, max_iter, tol)
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


def fista_solve(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    w: np.ndarray,
    alpha: float,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, list[float], list[int], float]:
    """Run FISTA for the weighted LASSO subproblem with fixed weights w.

    Performs accelerated proximal gradient descent (Beck & Teboulle, 2009):
        t_{k+1} = (1 + sqrt(1 + 4*t_k^2)) / 2           (momentum update)
        v       = x_k + ((t_k - 1) / t_{k+1}) * (x_k - x_{k-1})  (extrapolation)
        z       = v - alpha * grad_f(v)                   (gradient step at v)
        x_{k+1} = weighted_soft_threshold(z, alpha, lam, w)        (proximal step)
    until has_converged or max_iter is reached.

    The gradient and proximal steps are evaluated at the extrapolated point v,
    not at x_k.  This gives O(1/k^2) convergence vs O(1/k) for ISTA.
    The objective trace is non-monotone by design.

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
        Weighted LASSO objective value after each iteration (non-monotone).
    sparsity_trace : list[int]
        Number of non-zero coefficients (||beta||_0) after each iteration.
    runtime : float
        Wall-clock time in seconds for the entire solve.
    """
    n, p = X.shape

    # State variables: x_k (current iterate), x_km1 (previous iterate)
    x_k = np.zeros(p, dtype=np.float64)
    x_km1 = np.zeros(p, dtype=np.float64)
    t_k = 1.0

    objective_trace: list[float] = []
    sparsity_trace: list[int] = []

    t_start = time.perf_counter()

    for _ in range(max_iter):
        # Momentum scalar update  (math_spec §8)
        t_kp1 = (1.0 + np.sqrt(1.0 + 4.0 * t_k ** 2)) / 2.0

        # Extrapolation step — variable named `v` to avoid shadowing `y` param
        # v = x_k + ((t_k - 1) / t_{k+1}) * (x_k - x_{k-1})  (math_spec §8)
        v = x_k + ((t_k - 1.0) / t_kp1) * (x_k - x_km1)

        # Gradient of smooth part at extrapolated point  (math_spec §3)
        grad = X.T @ (X @ v - y) / n

        # Gradient step at v
        z = v - alpha * grad

        # Proximal step: weighted soft-thresholding  (math_spec §6)
        x_new = weighted_soft_threshold(z, alpha, lam, w)

        # Log objective and sparsity after proximal update
        objective_trace.append(compute_loss(X, y, x_new, lam, w))
        sparsity_trace.append(int(np.count_nonzero(x_new)))

        # Stopping criterion  (math_spec §11)
        if has_converged(x_k, x_new, tol):
            x_k = x_new
            break

        x_km1 = x_k
        x_k = x_new
        t_k = t_kp1

    runtime = time.perf_counter() - t_start

    return x_k, objective_trace, sparsity_trace, runtime
