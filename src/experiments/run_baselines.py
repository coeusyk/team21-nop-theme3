"""
Run baseline experiment comparison rows for T13.

Task: T13

Consumed by:
	src/experiments/run_dynamic.py to produce the final all-method table.

I/O contract:
	run_baseline_experiments(config_path, dynamic_config_path, best_config_path)
		-> pd.DataFrame with columns:
			method, solver, val_mse, val_mae, test_mse, test_mae,
			nnz, runtime_s, iterations
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.split import build_dataset
from src.metrics.regression_metrics import compute_regression_metrics
from src.metrics.sparsity_metrics import count_nonzero
from src.models.adaptive_lasso import run_adaptive_lasso
from src.models.lasso import run_lasso
from src.models.ridge import run_ridge
from src.utils.io import save_log, save_table
from src.utils.logging_utils import get_logger
from src.utils.seeds import set_seed

logger = get_logger(__name__)


def _resolve_alpha(X_train: np.ndarray, alpha_cfg: float | None) -> float:
	"""Resolve alpha from config, or compute as 1/L from training design matrix."""
	if alpha_cfg is not None:
		return float(alpha_cfg)
	n = X_train.shape[0]
	lipschitz = float(np.linalg.eigvalsh(X_train.T @ X_train).max() / n)
	if lipschitz <= 0.0:
		raise ValueError("Computed non-positive Lipschitz constant; cannot set alpha")
	return 1.0 / lipschitz


def _load_best_configs(best_config_path: Path) -> dict:
	"""Load T12 best config YAML with strict presence checks."""
	if not best_config_path.exists():
		raise NotImplementedError(
			"Missing outputs/logs/best_configs.yaml. Run T12 first to generate best "
			"hyperparameters before running T13."
		)
	with open(best_config_path, "r", encoding="utf-8") as fh:
		return yaml.safe_load(fh)


def _select_best_adaptive_config(best_configs: dict) -> dict:
	"""Select the best adaptive_lasso config across available solvers by val_mse."""
	candidates = [
		value
		for key, value in best_configs.items()
		if key.startswith("adaptive_lasso_")
	]
	if not candidates:
		raise NotImplementedError(
			"T12 best configs do not contain adaptive_lasso entries. "
			"Expected keys like 'adaptive_lasso_ista' or 'adaptive_lasso_fista'."
		)
	return min(candidates, key=lambda row: float(row["val_mse"]))


def _load_solver_cfg(path: str) -> dict:
	"""Load a solver YAML config and return a dictionary."""
	with open(path, "r", encoding="utf-8") as fh:
		cfg = yaml.safe_load(fh)
	if not isinstance(cfg, dict):
		raise NotImplementedError(
			f"Expected a mapping in solver config file: {path}"
		)
	return cfg


def _solver_optim_params(
	solver: str,
	optim_cfg: dict,
	ista_cfg: dict,
	fista_cfg: dict,
	X_train: np.ndarray,
) -> tuple[float, int, float]:
	"""Resolve alpha, max_iter, and tol for a given inner solver."""
	solver_cfg = ista_cfg if solver == "ista" else fista_cfg
	alpha = _resolve_alpha(X_train, solver_cfg.get("alpha", optim_cfg.get("alpha")))
	max_iter = int(solver_cfg.get("max_iter", optim_cfg["max_iter"]))
	tol = float(solver_cfg.get("tol", optim_cfg["tol"]))
	return alpha, max_iter, tol


def run_baseline_experiments(
	config_path: str = "configs/default.yaml",
	dynamic_config_path: str = "configs/dynamic_reweight.yaml",
	ista_config_path: str = "configs/ista.yaml",
	fista_config_path: str = "configs/fista.yaml",
	best_config_path: str | None = None,
) -> pd.DataFrame:
	"""Run ridge, lasso, and static adaptive lasso and save baseline comparison table.

	Parameters
	----------
	config_path : str
		Path to default YAML config.
	dynamic_config_path : str
		Path to dynamic reweight YAML config (for inner tolerances and limits).
	best_config_path : str or None
		Optional explicit path to T12 best config YAML. If None, defaults to
		outputs/logs/best_configs.yaml from config.

	Returns
	-------
	pd.DataFrame
		Baseline comparison rows with metrics and runtime columns.
	"""
	with open(config_path, "r", encoding="utf-8") as fh:
		cfg = yaml.safe_load(fh)
	ista_cfg = _load_solver_cfg(ista_config_path)
	fista_cfg = _load_solver_cfg(fista_config_path)
	data_cfg = cfg["data"]
	outputs_cfg = cfg["outputs"]
	optim_cfg = cfg["optim"]
	ridge_cfg = cfg["model"]["ridge"]
	lasso_cfg = cfg["model"]["lasso"]

	best_path = (
		Path(best_config_path)
		if best_config_path is not None
		else Path(outputs_cfg["logs"]) / "best_configs.yaml"
	)
	best_configs = _load_best_configs(best_path)
	best_adaptive = _select_best_adaptive_config(best_configs)

	set_seed(int(data_cfg["seed"]))

	X_train, X_val, X_test, y_train, y_val, y_test, feature_names = build_dataset(
		raw_path=data_cfg["raw_path"],
		target_col=data_cfg["target_col"],
		val_size=float(data_cfg["val_size"]),
		test_size=float(data_cfg["test_size"]),
		seed=int(data_cfg["seed"]),
		numeric_impute_strategy=data_cfg.get("numeric_impute_strategy", "median"),
		categorical_impute_strategy=data_cfg.get(
			"categorical_impute_strategy", "most_frequent"
		),
		drop_cols=data_cfg.get("drop_cols"),
	)

	# Center y so zero-intercept custom solvers predict correctly
	y_mean = float(y_train.mean())
	y_train_c = y_train - y_mean
	y_val_c = y_val - y_mean
	y_test_c = y_test - y_mean

	adaptive_solver = str(best_adaptive["solver"])
	alpha, max_iter, tol = _solver_optim_params(
		solver=adaptive_solver,
		optim_cfg=optim_cfg,
		ista_cfg=ista_cfg,
		fista_cfg=fista_cfg,
		X_train=X_train,
	)

	ridge_results = run_ridge(
		X_train=X_train,
		y_train=y_train,
		X_val=X_val,
		y_val=y_val,
		X_test=X_test,
		y_test=y_test,
		feature_names=feature_names,
		alphas=ridge_cfg["alphas"],
		seed=int(data_cfg["seed"]),
	)

	lasso_results = run_lasso(
		X_train=X_train,
		y_train=y_train,
		X_val=X_val,
		y_val=y_val,
		X_test=X_test,
		y_test=y_test,
		feature_names=feature_names,
		alphas=lasso_cfg["alphas"],
		seed=int(data_cfg["seed"]),
	)

	beta_a, obj_a, _sp_a, rt_a = run_adaptive_lasso(
		X=X_train,
		y=y_train_c,
		lam=float(best_adaptive["lam"]),
		ridge_coef=ridge_results["beta"],
		gamma=float(best_adaptive["gamma"]),
		eps=float(best_adaptive["eps"]),
		alpha=alpha,
		max_iter=max_iter,
		tol=tol,
		solver=adaptive_solver,
	)
	val_metrics_a = compute_regression_metrics(y_val, X_val @ beta_a + y_mean)
	test_metrics_a = compute_regression_metrics(y_test, X_test @ beta_a + y_mean)

	rows = [
		{
			"method": "ridge",
			"solver": "sklearn_ridgecv",
			"val_mse": float(ridge_results["val_mse"]),
			"val_mae": float(ridge_results["val_mae"]),
			"test_mse": float(ridge_results["test_mse"]),
			"test_mae": float(ridge_results["test_mae"]),
			"nnz": int(ridge_results["nnz"]),
			"runtime_s": float(ridge_results["runtime"]),
			"iterations": np.nan,
		},
		{
			"method": "lasso",
			"solver": "sklearn_lassocv",
			"val_mse": float(lasso_results["val_mse"]),
			"val_mae": float(lasso_results["val_mae"]),
			"test_mse": float(lasso_results["test_mse"]),
			"test_mae": float(lasso_results["test_mae"]),
			"nnz": int(lasso_results["nnz"]),
			"runtime_s": float(lasso_results["runtime"]),
			"iterations": np.nan,
		},
		{
			"method": "adaptive_lasso",
			"solver": str(best_adaptive["solver"]),
			"val_mse": float(val_metrics_a["mse"]),
			"val_mae": float(val_metrics_a["mae"]),
			"test_mse": float(test_metrics_a["mse"]),
			"test_mae": float(test_metrics_a["mae"]),
			"nnz": int(count_nonzero(beta_a)),
			"runtime_s": float(rt_a),
			"iterations": int(len(obj_a)),
		},
	]

	result_df = pd.DataFrame(rows)

	tables_dir = Path(outputs_cfg["tables"])
	logs_dir = Path(outputs_cfg["logs"])
	table_path = save_table(result_df, tables_dir, "baselines_comparison.csv")
	save_log(
		{
			"task": "T13-run-baselines",
			"config_path": config_path,
			"dynamic_config_path": dynamic_config_path,
			"ista_config_path": ista_config_path,
			"fista_config_path": fista_config_path,
			"best_config_path": str(best_path),
			"adaptive_selected": best_adaptive,
			"rows": rows,
			"table_path": str(table_path),
		},
		logs_dir,
		"t13_run_baselines_log.json",
	)

	logger.info("Saved baseline comparison table: %s", table_path)
	return result_df


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	"""Parse command-line arguments for baseline experiment runner."""
	parser = argparse.ArgumentParser(description="T13 - Run baseline comparisons")
	parser.add_argument(
		"--config",
		type=str,
		default="configs/default.yaml",
		help="Path to default YAML config.",
	)
	parser.add_argument(
		"--dynamic-config",
		type=str,
		default="configs/dynamic_reweight.yaml",
		help="Path to dynamic reweight YAML config.",
	)
	parser.add_argument(
		"--ista-config",
		type=str,
		default="configs/ista.yaml",
		help="Path to ISTA YAML config.",
	)
	parser.add_argument(
		"--fista-config",
		type=str,
		default="configs/fista.yaml",
		help="Path to FISTA YAML config.",
	)
	parser.add_argument(
		"--best-config",
		type=str,
		default=None,
		help="Optional path to outputs/logs/best_configs.yaml",
	)
	return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> pd.DataFrame:
	"""CLI entry point for T13 baseline experiment runner."""
	args = _parse_args(argv)
	return run_baseline_experiments(
		config_path=args.config,
		dynamic_config_path=args.dynamic_config,
		ista_config_path=args.ista_config,
		fista_config_path=args.fista_config,
		best_config_path=args.best_config,
	)


if __name__ == "__main__":
	main()
