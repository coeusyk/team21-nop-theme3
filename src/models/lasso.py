"""Standard LASSO baseline using sklearn LassoCV.

Run as a script
---------------
    uv run python -m src.models.lasso --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.split import build_dataset
from src.metrics.regression_metrics import compute_regression_metrics
from src.metrics.sparsity_metrics import count_nonzero
from src.utils.io import save_log, save_table
from src.utils.logging_utils import get_logger
from src.utils.seeds import set_seed

logger = get_logger(__name__)


def run_lasso(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    alphas: list[float],
    seed: int,
) -> dict:
    """Fit LassoCV on training data and evaluate on validation and test splits.

    ``LassoCV`` performs k-fold cross-validation (``cv=5``) over the provided
    ``alphas`` grid to select the best regularisation strength.  Sparsity is
    reported as the count of non-zero coefficients using
    :func:`~src.metrics.sparsity_metrics.count_nonzero`.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, p)
        Standardised training feature matrix from
        :func:`~src.data.split.build_dataset`.
    y_train : np.ndarray, shape (n_train,)
        Training target vector.
    X_val : np.ndarray, shape (n_val, p)
        Validation feature matrix.
    y_val : np.ndarray, shape (n_val,)
        Validation target vector.
    X_test : np.ndarray, shape (n_test, p)
        Test feature matrix.
    y_test : np.ndarray, shape (n_test,)
        Test target vector.
    feature_names : list[str]
        Column names for each of the ``p`` features in X.
    alphas : list[float]
        Regularisation strengths to search over via ``LassoCV``.
    seed : int
        Random seed forwarded to ``LassoCV`` for reproducible fold selection.

    Returns
    -------
    dict with keys:
        ``beta``        — np.ndarray, shape (p,), fitted coefficients
        ``alpha_best``  — float, best alpha chosen by ``LassoCV``
        ``val_mse``     — float
        ``val_mae``     — float
        ``test_mse``    — float
        ``test_mae``    — float
        ``nnz``         — int, number of non-zero coefficients
        ``runtime``     — float, wall-clock seconds for fitting
        ``feature_names`` — list[str]
    """
    from sklearn.linear_model import LassoCV

    set_seed(seed)

    t0 = time.perf_counter()
    model = LassoCV(alphas=alphas, cv=5, random_state=seed, max_iter=10000)
    model.fit(X_train, y_train)
    runtime = time.perf_counter() - t0

    beta: np.ndarray = model.coef_.astype(np.float64)
    alpha_best: float = float(model.alpha_)

    val_metrics = compute_regression_metrics(y_val, model.predict(X_val))
    test_metrics = compute_regression_metrics(y_test, model.predict(X_test))
    nnz: int = count_nonzero(beta)

    logger.info(
        "LassoCV: alpha_best=%.6f  nnz=%d  val_mse=%.4f  test_mse=%.4f",
        alpha_best,
        nnz,
        val_metrics["mse"],
        test_metrics["mse"],
    )

    return {
        "beta": beta,
        "alpha_best": alpha_best,
        "val_mse": val_metrics["mse"],
        "val_mae": val_metrics["mae"],
        "test_mse": test_metrics["mse"],
        "test_mae": test_metrics["mae"],
        "nnz": nnz,
        "runtime": runtime,
        "feature_names": feature_names,
    }


def _save_outputs(results: dict, cfg: dict, timestamp: str) -> None:
    """Save metrics CSV, coefficient CSV, and a JSON run log."""
    tables_dir = Path(cfg["outputs"]["tables"])
    logs_dir = Path(cfg["outputs"]["logs"])

    # --- Metrics table ---
    metrics_df = pd.DataFrame(
        [
            {
                "method": "lasso",
                "alpha_best": results["alpha_best"],
                "val_mse": results["val_mse"],
                "val_mae": results["val_mae"],
                "test_mse": results["test_mse"],
                "test_mae": results["test_mae"],
                "nnz": results["nnz"],
                "runtime_s": results["runtime"],
            }
        ]
    )
    save_table(metrics_df, tables_dir, f"lasso_metrics_{timestamp}.csv")
    logger.info("Metrics saved to %s/lasso_metrics_%s.csv", tables_dir, timestamp)

    # --- Coefficient table (non-zero only) ---
    coef_df = pd.DataFrame(
        {"feature": results["feature_names"], "coefficient": results["beta"]}
    )
    coef_df = coef_df[np.abs(coef_df["coefficient"]) > 0].sort_values(
        "coefficient", key=np.abs, ascending=False
    )
    save_table(coef_df, tables_dir, f"lasso_coefficients_{timestamp}.csv")
    logger.info(
        "Coefficients saved to %s/lasso_coefficients_%s.csv", tables_dir, timestamp
    )

    # --- Run log ---
    log_payload = {
        "task": "T9-lasso-baseline",
        "timestamp": timestamp,
        "config": cfg,
        "metrics": {
            "alpha_best": results["alpha_best"],
            "val_mse": results["val_mse"],
            "val_mae": results["val_mae"],
            "test_mse": results["test_mse"],
            "test_mae": results["test_mae"],
            "nnz": results["nnz"],
            "runtime_s": results["runtime"],
        },
    }
    save_log(log_payload, logs_dir, f"lasso_run_{timestamp}.json")
    logger.info("Run log saved to %s/lasso_run_%s.json", logs_dir, timestamp)


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the LASSO script."""
    parser = argparse.ArgumentParser(
        description="T9 — Standard LASSO baseline (LassoCV)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to a YAML config file (default: configs/default.yaml)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """CLI entry point: load config, run LassoCV, save outputs, return results dict."""
    args = _parse_args(argv)
    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    data_cfg = cfg["data"]
    lasso_cfg = cfg["model"]["lasso"]

    logger.info("Loading dataset from %s", data_cfg["raw_path"])
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = build_dataset(
        raw_path=data_cfg["raw_path"],
        target_col=data_cfg["target_col"],
        val_size=data_cfg["val_size"],
        test_size=data_cfg["test_size"],
        seed=data_cfg["seed"],
        numeric_impute_strategy=data_cfg.get("numeric_impute_strategy", "median"),
        categorical_impute_strategy=data_cfg.get(
            "categorical_impute_strategy", "most_frequent"
        ),
        drop_cols=data_cfg.get("drop_cols"),
    )

    results = run_lasso(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        alphas=lasso_cfg["alphas"],
        seed=data_cfg["seed"],
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _save_outputs(results, cfg, timestamp)

    return results


if __name__ == "__main__":
    main()
