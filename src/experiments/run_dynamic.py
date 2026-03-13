"""
Run dynamic experiment comparison rows and assemble the full T13 table.

Task: T13

Depends on:
	- T12 best config artifact at outputs/logs/best_configs.yaml
	- Baseline table from src/experiments/run_baselines.py

I/O contract:
	run_dynamic_experiments(config_path, dynamic_config_path, best_config_path)
		-> pd.DataFrame with five methods:
			ridge, lasso, adaptive_lasso,
			dynamic_reweighted_lasso_ista, dynamic_reweighted_lasso_fista
		and columns:
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
from src.models.dynamic_reweighted_lasso import run_dynamic_reweighted_lasso
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


def _get_dynamic_cfg(best_configs: dict, solver: str) -> dict:
	"""Get dynamic best config for a given solver from T12 output."""
	key = f"dynamic_reweighted_lasso_{solver}"
	if key not in best_configs:
		raise NotImplementedError(
			"Missing dynamic best config key in T12 output: "
			f"{key}. Re-run T12 with both solvers in search.solvers."
		)
	return best_configs[key]


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


def run_dynamic_experiments(
	config_path: str = "configs/default.yaml",
	dynamic_config_path: str = "configs/dynamic_reweight.yaml",
	ista_config_path: str = "configs/ista.yaml",
	fista_config_path: str = "configs/fista.yaml",
	best_config_path: str | None = None,
) -> pd.DataFrame:
	"""Run dynamic ISTA/FISTA and save dynamic-only plus full T13 comparison tables.

	Parameters
	----------
	config_path : str
		Path to default YAML config.
	dynamic_config_path : str
		Path to dynamic reweight YAML config.
	best_config_path : str or None
		Optional explicit path to T12 best config YAML. If None, defaults to
		outputs/logs/best_configs.yaml from config.

	Returns
	-------
	pd.DataFrame
		Full comparison DataFrame with all five methods.
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

	best_path = (
		Path(best_config_path)
		if best_config_path is not None
		else Path(outputs_cfg["logs"]) / "best_configs.yaml"
	)
	best_configs = _load_best_configs(best_path)
	best_dyn_ista = _get_dynamic_cfg(best_configs, "ista")
	best_dyn_fista = _get_dynamic_cfg(best_configs, "fista")

	set_seed(int(data_cfg["seed"]))

	X_train, X_val, X_test, y_train, y_val, y_test, _feature_names = build_dataset(
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

	dynamic_rows: list[dict] = []
	for best_row in [best_dyn_ista, best_dyn_fista]:
		solver = str(best_row["solver"])
		alpha, solver_max_iter, solver_tol = _solver_optim_params(
			solver=solver,
			optim_cfg=optim_cfg,
			ista_cfg=ista_cfg,
			fista_cfg=fista_cfg,
			X_train=X_train,
		)
		(
			beta_d,
			_obj_d,
			_sp_d,
			_weight_trace,
			inner_counts,
			runtime_d,
		) = run_dynamic_reweighted_lasso(
			X=X_train,
			y=y_train,
			lam=float(best_row["lam"]),
			gamma=float(best_row["gamma"]),
			eps=float(best_row["eps"]),
			w_max=float(dynamic_cfg["w_max"]),
			alpha=alpha,
			max_outer_iter=int(best_row["max_outer_iter"]),
			inner_max_iter=int(dynamic_cfg.get("inner_max_iter", solver_max_iter)),
			inner_tol=float(dynamic_cfg.get("inner_tol", solver_tol)),
			solver=solver,
		)
		val_metrics = compute_regression_metrics(y_val, X_val @ beta_d)
		test_metrics = compute_regression_metrics(y_test, X_test @ beta_d)
		dynamic_rows.append(
			{
				"method": f"dynamic_reweighted_lasso_{solver}",
				"solver": solver,
				"val_mse": float(val_metrics["mse"]),
				"val_mae": float(val_metrics["mae"]),
				"test_mse": float(test_metrics["mse"]),
				"test_mae": float(test_metrics["mae"]),
				"nnz": int(count_nonzero(beta_d)),
				"runtime_s": float(runtime_d),
				"iterations": int(sum(inner_counts)),
			}
		)

	dynamic_df = pd.DataFrame(dynamic_rows)

	tables_dir = Path(outputs_cfg["tables"])
	logs_dir = Path(outputs_cfg["logs"])
	dynamic_table_path = save_table(dynamic_df, tables_dir, "dynamic_comparison.csv")

	baselines_path = tables_dir / "baselines_comparison.csv"
	if not baselines_path.exists():
		raise NotImplementedError(
			"Missing baseline comparison table. Run src.experiments.run_baselines "
			"before src.experiments.run_dynamic to assemble full T13 output."
		)
	baseline_df = pd.read_csv(baselines_path)

	full_df = pd.concat([baseline_df, dynamic_df], ignore_index=True)
	full_table_path = save_table(full_df, tables_dir, "full_comparison_table.csv")

	save_log(
		{
			"task": "T13-run-dynamic",
			"config_path": config_path,
			"dynamic_config_path": dynamic_config_path,
			"ista_config_path": ista_config_path,
			"fista_config_path": fista_config_path,
			"best_config_path": str(best_path.resolve()),
			"dynamic_rows": dynamic_rows,
			"dynamic_table_path": str(dynamic_table_path),
			"baseline_table_path": str(baselines_path.resolve()),
			"full_table_path": str(full_table_path),
		},
		logs_dir,
		"t13_run_dynamic_log.json",
	)

	logger.info("Saved dynamic comparison table: %s", dynamic_table_path)
	logger.info("Saved full comparison table: %s", full_table_path)
	return full_df


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	"""Parse command-line arguments for dynamic experiment runner."""
	parser = argparse.ArgumentParser(description="T13 - Run dynamic comparisons")
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
	"""CLI entry point for T13 dynamic experiment runner."""
	args = _parse_args(argv)
	return run_dynamic_experiments(
		config_path=args.config,
		dynamic_config_path=args.dynamic_config,
		ista_config_path=args.ista_config,
		fista_config_path=args.fista_config,
		best_config_path=args.best_config,
	)


if __name__ == "__main__":
	main()
