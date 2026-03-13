"""
Hyperparameter search for adaptive and dynamic reweighted LASSO models.

Task: T12

Consumed by:
	T13 src/experiments/run_baselines.py and src/experiments/run_dynamic.py

I/O contract:
	run_hyperparameter_search(config_path, dynamic_config_path) -> dict
		Reads dataset/preprocessing settings from config YAML files.
		Runs validation-grid search and writes:
			- outputs/logs/best_configs.yaml
			- outputs/tables/validation_mse_grid.csv
		Returns a dictionary containing best settings and result paths.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import RidgeCV

from src.data.split import build_dataset
from src.metrics.regression_metrics import compute_regression_metrics
from src.metrics.sparsity_metrics import count_nonzero
from src.models.adaptive_lasso import run_adaptive_lasso
from src.models.dynamic_reweighted_lasso import run_dynamic_reweighted_lasso
from src.utils.io import save_log, save_table
from src.utils.logging_utils import get_logger
from src.utils.seeds import set_seed

logger = get_logger(__name__)


def _resolve_alpha(X_train: np.ndarray, alpha_cfg: float | None) -> float:
	"""Resolve step size alpha from config or compute 1/L from training design."""
	if alpha_cfg is not None:
		return float(alpha_cfg)
	n = X_train.shape[0]
	lipschitz = float(np.linalg.eigvalsh(X_train.T @ X_train).max() / n)
	if lipschitz <= 0.0:
		raise ValueError("Computed non-positive Lipschitz constant; cannot set alpha")
	return 1.0 / lipschitz


def _get_search_grid(dynamic_cfg: dict) -> dict:
	"""Return validated T12 search grid from dynamic config."""
	if "search" not in dynamic_cfg:
		raise NotImplementedError(
			"T12 requires 'search' in configs/dynamic_reweight.yaml with lam_values, "
			"gamma_values, eps_values, max_outer_iter_values, and solvers."
		)

	search_cfg = dynamic_cfg["search"]
	required_keys = [
		"lam_values",
		"gamma_values",
		"eps_values",
		"max_outer_iter_values",
		"solvers",
	]
	missing_keys = [key for key in required_keys if key not in search_cfg]
	if missing_keys:
		raise NotImplementedError(
			"T12 search grid is incomplete in configs/dynamic_reweight.yaml. "
			f"Missing keys: {missing_keys}."
		)
	return search_cfg


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


def run_hyperparameter_search(
	config_path: str = "configs/default.yaml",
	dynamic_config_path: str = "configs/dynamic_reweight.yaml",
	ista_config_path: str = "configs/ista.yaml",
	fista_config_path: str = "configs/fista.yaml",
) -> dict:
	"""Run T12 validation search and save best configs and full CSV results.

	Parameters
	----------
	config_path : str
		Path to default YAML config (data, outputs, model, optim).
	dynamic_config_path : str
		Path to dynamic YAML config including search grids.

	Returns
	-------
	dict
		Dictionary with best configuration payload and output file paths.
	"""
	with open(config_path, "r", encoding="utf-8") as fh:
		cfg = yaml.safe_load(fh)
	with open(dynamic_config_path, "r", encoding="utf-8") as fh:
		dynamic_cfg = yaml.safe_load(fh)
	ista_cfg = _load_solver_cfg(ista_config_path)
	fista_cfg = _load_solver_cfg(fista_config_path)

	data_cfg = cfg["data"]
	outputs_cfg = cfg["outputs"]
	optim_cfg = cfg["optim"]
	ridge_cfg = cfg["model"]["ridge"]

	search_cfg = _get_search_grid(dynamic_cfg)

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

	_ = X_test, y_test, feature_names  # Explicitly unused in T12.

	ridge_model = RidgeCV(alphas=ridge_cfg["alphas"])
	ridge_model.fit(X_train, y_train)
	ridge_coef = ridge_model.coef_.astype(np.float64)

	records: list[dict] = []

	for solver in search_cfg["solvers"]:
		alpha, max_iter, tol = _solver_optim_params(
			solver=str(solver),
			optim_cfg=optim_cfg,
			ista_cfg=ista_cfg,
			fista_cfg=fista_cfg,
			X_train=X_train,
		)
		for lam in search_cfg["lam_values"]:
			for gamma in search_cfg["gamma_values"]:
				for eps in search_cfg["eps_values"]:
					beta_a, obj_a, sp_a, rt_a = run_adaptive_lasso(
						X=X_train,
						y=y_train,
						lam=float(lam),
						ridge_coef=ridge_coef,
						gamma=float(gamma),
						eps=float(eps),
						alpha=alpha,
						max_iter=max_iter,
						tol=tol,
						solver=str(solver),
					)
					pred_val_a = X_val @ beta_a
					val_metrics_a = compute_regression_metrics(y_val, pred_val_a)
					records.append(
						{
							"method": "adaptive_lasso",
							"solver": str(solver),
							"lam": float(lam),
							"gamma": float(gamma),
							"eps": float(eps),
							"max_outer_iter": np.nan,
							"val_mse": float(val_metrics_a["mse"]),
							"val_mae": float(val_metrics_a["mae"]),
							"nnz": int(count_nonzero(beta_a)),
							"runtime_s": float(rt_a),
							"iterations": int(len(obj_a)),
							"outer_iterations": np.nan,
							"inner_iterations_total": int(len(obj_a)),
							"objective_last": float(obj_a[-1]) if obj_a else np.nan,
							"sparsity_last": int(sp_a[-1]) if sp_a else int(count_nonzero(beta_a)),
						}
					)

					for max_outer_iter in search_cfg["max_outer_iter_values"]:
						(
							beta_d,
							obj_d,
							sp_d,
							_w_trace,
							inner_counts,
							rt_d,
						) = run_dynamic_reweighted_lasso(
							X=X_train,
							y=y_train,
							lam=float(lam),
							gamma=float(gamma),
							eps=float(eps),
							w_max=float(dynamic_cfg["w_max"]),
							alpha=alpha,
							max_outer_iter=int(max_outer_iter),
							inner_max_iter=int(dynamic_cfg.get("inner_max_iter", max_iter)),
							inner_tol=float(dynamic_cfg.get("inner_tol", tol)),
							solver=str(solver),
						)
						pred_val_d = X_val @ beta_d
						val_metrics_d = compute_regression_metrics(y_val, pred_val_d)
						records.append(
							{
								"method": "dynamic_reweighted_lasso",
								"solver": str(solver),
								"lam": float(lam),
								"gamma": float(gamma),
								"eps": float(eps),
								"max_outer_iter": int(max_outer_iter),
								"val_mse": float(val_metrics_d["mse"]),
								"val_mae": float(val_metrics_d["mae"]),
								"nnz": int(count_nonzero(beta_d)),
								"runtime_s": float(rt_d),
								"iterations": int(sum(inner_counts)),
								"outer_iterations": int(len(obj_d)),
								"inner_iterations_total": int(sum(inner_counts)),
								"objective_last": float(obj_d[-1]) if obj_d else np.nan,
								"sparsity_last": int(sp_d[-1]) if sp_d else int(count_nonzero(beta_d)),
							}
						)

	results_df = pd.DataFrame(records)

	best_configs: dict[str, dict] = {}
	for method in ["adaptive_lasso", "dynamic_reweighted_lasso"]:
		method_df = results_df[results_df["method"] == method]
		if method_df.empty:
			raise RuntimeError(f"No search results found for method: {method}")

		for solver in search_cfg["solvers"]:
			solver_df = method_df[method_df["solver"] == solver]
			if solver_df.empty:
				continue
			best_row = solver_df.loc[solver_df["val_mse"].idxmin()]
			key = f"{method}_{solver}"
			best_configs[key] = {
				"method": method,
				"solver": str(solver),
				"lam": float(best_row["lam"]),
				"gamma": float(best_row["gamma"]),
				"eps": float(best_row["eps"]),
				"max_outer_iter": (
					int(best_row["max_outer_iter"])
					if not pd.isna(best_row["max_outer_iter"])
					else None
				),
				"val_mse": float(best_row["val_mse"]),
				"val_mae": float(best_row["val_mae"]),
				"nnz": int(best_row["nnz"]),
				"runtime_s": float(best_row["runtime_s"]),
				"iterations": int(best_row["iterations"]),
			}

	tables_dir = Path(outputs_cfg["tables"])
	logs_dir = Path(outputs_cfg["logs"])

	csv_path = save_table(results_df, tables_dir, "validation_mse_grid.csv")
	yaml_path = logs_dir / "best_configs.yaml"
	yaml_path.parent.mkdir(parents=True, exist_ok=True)
	with open(yaml_path, "w", encoding="utf-8") as fh:
		yaml.safe_dump(best_configs, fh, sort_keys=True)

	save_log(
		{
			"task": "T12-hyperparameter-search",
			"config_path": config_path,
			"dynamic_config_path": dynamic_config_path,
			"ista_config_path": ista_config_path,
			"fista_config_path": fista_config_path,
			"result_rows": int(len(results_df)),
			"best_configs": best_configs,
			"csv_path": str(csv_path),
			"best_configs_yaml": str(yaml_path.resolve()),
		},
		logs_dir,
		"t12_search_log.json",
	)

	logger.info("Saved validation grid CSV: %s", csv_path)
	logger.info("Saved best configs YAML: %s", yaml_path.resolve())

	return {
		"best_configs": best_configs,
		"validation_csv": str(csv_path),
		"best_configs_yaml": str(yaml_path.resolve()),
	}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	"""Parse command-line arguments for T12 hyperparameter search."""
	parser = argparse.ArgumentParser(description="T12 - Hyperparameter search")
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
	return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
	"""CLI entry point for T12 hyperparameter search."""
	args = _parse_args(argv)
	return run_hyperparameter_search(
		config_path=args.config,
		dynamic_config_path=args.dynamic_config,
		ista_config_path=args.ista_config,
		fista_config_path=args.fista_config,
	)


if __name__ == "__main__":
	main()
