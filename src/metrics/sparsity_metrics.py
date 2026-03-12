"""Sparsity metrics shared across all sparse model baselines."""

from __future__ import annotations

import numpy as np


def count_nonzero(beta: np.ndarray, tol: float = 1e-8) -> int:
    """Count the number of non-zero coefficients in a coefficient vector.

    Parameters
    ----------
    beta : np.ndarray, shape (p,)
        Coefficient vector.
    tol : float
        Coefficients with ``|beta_j| <= tol`` are treated as zero.
        Default is ``1e-8``.

    Returns
    -------
    int
        Number of entries in ``beta`` whose absolute value exceeds ``tol``.
    """
    return int(np.sum(np.abs(beta) > tol))


def sparsity_ratio(beta: np.ndarray, tol: float = 1e-8) -> float:
    """Return the fraction of coefficients that are exactly zero (within tol).

    Parameters
    ----------
    beta : np.ndarray, shape (p,)
        Coefficient vector.
    tol : float
        Entries with ``|beta_j| <= tol`` are counted as zero.

    Returns
    -------
    float
        Value in ``[0, 1]``: fraction of coordinates that are zero.
    """
    p = len(beta)
    if p == 0:
        return 0.0
    return float(np.sum(np.abs(beta) <= tol) / p)
