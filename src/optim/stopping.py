"""
Convergence stopping criterion for proximal gradient solvers.

Consumed by: T5 (ista.py), T6 (fista.py), T11 (dynamic_reweighted_lasso.py)

Contract:
    has_converged(beta_prev, beta_curr, tol) -> bool
        beta_prev: ndarray, shape (p,) — coefficient vector at previous step
        beta_curr: ndarray, shape (p,) — coefficient vector at current step
        tol      : float > 0           — convergence tolerance
        returns  : True if relative change < tol

    Relative change criterion (docs/math_spec.md §11):
        ||beta_curr - beta_prev||_2  /  max(1.0, ||beta_prev||_2)  <  tol

    The max(1.0, ...) denominator prevents false convergence when beta_prev
    is near zero (e.g., at initialisation).
"""

import numpy as np


def has_converged(
    beta_prev: np.ndarray,
    beta_curr: np.ndarray,
    tol: float,
) -> bool:
    """Return True if the relative change ||Δβ||/max(1,||β_prev||) is below tol."""
    diff_norm = np.linalg.norm(beta_curr - beta_prev)
    denom = max(1.0, float(np.linalg.norm(beta_prev)))
    return bool(diff_norm / denom < tol)
