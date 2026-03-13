"""Regression evaluation metrics shared across all model baselines."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute MSE and MAE for a set of regression predictions.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Ground-truth target values.
    y_pred : np.ndarray, shape (n,)
        Model predictions.

    Returns
    -------
    dict with keys ``"mse"`` and ``"mae"`` (both float).
    """
    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"mse": mse, "mae": mae}
