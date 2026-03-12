"""
Proximal operators for plain and weighted ℓ₁ penalties.

Consumed by: T5 (ista.py), T6 (fista.py), T10 (adaptive_lasso.py via T5/T6),
             T11 (dynamic_reweighted_lasso.py)

Contract:
    soft_threshold(z, tau) -> ndarray, shape same as z
        z  : ndarray          — gradient step vector
        tau: float or ndarray — threshold (broadcast-compatible with z)
        returns: sign(z) * max(|z| - tau, 0)  element-wise

    weighted_soft_threshold(z, alpha, lam, w) -> ndarray, shape (p,)
        z    : ndarray, shape (p,) — gradient step vector
        alpha: float               — step size
        lam  : float               — regularisation strength
        w    : ndarray, shape (p,) — per-coordinate weights (>= 0)
        returns: sign(z_j) * max(|z_j| - alpha*lam*w_j, 0) for each j

Math reference: docs/math_spec.md §5 (plain) and §6 (weighted).
IRL1 weight update is NOT performed here; see src/optim/reweight.py (T7).
"""

import numpy as np


def soft_threshold(z: np.ndarray, tau) -> np.ndarray:
    """Apply coordinate-wise soft-thresholding S_tau(z) = sign(z)*max(|z|-tau, 0)."""
    return np.sign(z) * np.maximum(np.abs(z) - tau, 0.0)


def weighted_soft_threshold(
    z: np.ndarray,
    alpha: float,
    lam: float,
    w: np.ndarray,
) -> np.ndarray:
    """Apply weighted soft-thresholding with per-coordinate threshold alpha*lam*w_j."""
    tau = alpha * lam * w
    return soft_threshold(z, tau)
