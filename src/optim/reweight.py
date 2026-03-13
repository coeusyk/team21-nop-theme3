"""
IRL1 reweighting engine for dynamic weighted LASSO.

Math reference: docs/math_spec.md §7 (IRL1 weight update).
Algorithm origin: iteratively reweighted ℓ₁ minimisation (Candès, Wakin, Boyd, 2008).
This module is an application of IRL1 — no theoretical novelty is claimed.

Consumed by:
    T11 (src/models/dynamic_reweighted_lasso.py) — Yash Karecha
        Calls compute_weights after each inner solve to update w for next outer step.
        Calls clip_weights immediately after compute_weights to bound extreme values.
        Input to compute_weights:  beta (p,) float64 — current coefficient vector
        Input to clip_weights:     w (p,) float64    — raw weights from compute_weights
        Output of compute_weights: w (p,) float64    — IRL1 weights
        Output of clip_weights:    w (p,) float64    — clipped weights

    T12 (src/experiments/cross_validate.py)      — Tenzin Kunga
        Sweeps gamma ∈ {0.5, 1.0, 2.0} and eps ∈ {1e-3, 1e-4} as arguments to
        compute_weights (passed through dynamic_reweighted_lasso.py).

    T13 (src/experiments/run_dynamic.py)         — Tenzin Kunga
        Drives the outer IRL1 loop; relies on this module via T11.

I/O contract:
    compute_weights(beta, gamma, eps)
        beta  : np.ndarray, shape (p,), float64 — current coefficient vector
        gamma : float > 0  — weight sharpness exponent
        eps   : float > 0  — numerical stability constant (prevents division by zero)
        returns: np.ndarray, shape (p,), float64
            w_j = 1 / (|β_j| + ε)^γ   for each coordinate j

    clip_weights(w, w_max)
        w     : np.ndarray, shape (p,), float64 — weight vector (entries >= 0)
        w_max : float > 0  — maximum allowed weight value
        returns: np.ndarray, shape (p,), float64
            min(w_j, w_max)   for each coordinate j
"""

import numpy as np


def compute_weights(
    beta: np.ndarray,
    gamma: float,
    eps: float,
) -> np.ndarray:
    """Return IRL1 weights w_j = 1 / (|β_j| + ε)^γ (Candès et al., 2008).

    At initialisation with β = 0, all weights equal 1/ε^γ (maximally large).
    As |β_j| grows, w_j decreases, relaxing the ℓ₁ penalty on that coordinate.
    Features that stay near zero retain large weights, enforcing sparsity.

    Parameters
    ----------
    beta : np.ndarray, shape (p,)
        Current coefficient vector. May be all zeros at initialisation.
    gamma : float
        Weight sharpness exponent (> 0). Larger γ sharpens the sparsity contrast.
    eps : float
        Numerical stability constant (> 0). Prevents division by zero when β_j = 0.

    Returns
    -------
    w : np.ndarray, shape (p,)
        Non-negative weight vector. Entry j equals 1 / (|β_j| + ε)^γ.
    """
    return 1.0 / (np.abs(beta) + eps) ** gamma


def clip_weights(
    w: np.ndarray,
    w_max: float,
) -> np.ndarray:
    """Clip weights to [0, w_max] to prevent extreme values from destabilising the solver.

    Should be applied immediately after compute_weights when eps is very small
    or beta values are near zero, to bound the per-coordinate threshold
    α·λ·w_j passed to the inner proximal solver.

    Parameters
    ----------
    w : np.ndarray, shape (p,)
        Weight vector to clip (entries >= 0).
    w_max : float
        Maximum allowed weight value (> 0).

    Returns
    -------
    w_clipped : np.ndarray, shape (p,)
        Weight vector with all entries capped at w_max.
    """
    return np.minimum(w, w_max)
