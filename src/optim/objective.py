"""
Weighted LASSO objective function.

Consumed by: T5 (ista.py), T6 (fista.py), T11 (dynamic_reweighted_lasso.py)

Contract:
    compute_loss(X, y, beta, lam, w) -> float
        X   : ndarray, shape (n, p)  — standardised design matrix
        y   : ndarray, shape (n,)    — target vector
        beta: ndarray, shape (p,)    — coefficient vector
        lam : float >= 0             — regularisation strength
        w   : ndarray, shape (p,), entries >= 0 — per-coordinate weights
        returns: scalar float
            (1/2n) * ||y - X @ beta||_2^2  +  lam * sum_j( w_j * |beta_j| )
"""

import numpy as np


def compute_loss(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    lam: float,
    w: np.ndarray,
) -> float:
    """Return weighted LASSO objective (1/2n)||y-Xβ||² + λ·Σ w_j|β_j|."""
    n = X.shape[0]
    residual = y - X @ beta
    smooth_term = 0.5 * np.dot(residual, residual) / n
    reg_term = lam * np.dot(w, np.abs(beta))
    return float(smooth_term + reg_term)
