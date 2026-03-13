"""Support stability analysis via Jaccard index across repeated runs.

Task: T14 (owner: Both)

Consumed by:
	T25 Results and Discussion section.

I/O contract:
	run_stability_analysis(config_path, dynamic_config_path, best_config_path)
		-> pd.DataFrame with one row per method and columns:
			method, mean_jaccard, std_jaccard, min_jaccard,
			max_jaccard, num_runs, num_pairs
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.split import build_dataset
from src.models.adaptive_lasso import run_adaptive_lasso
from src.models.dynamic_reweighted_lasso import run_dynamic_reweighted_lasso
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
			"hyperparameters before running T14."
		)
	with open(best_config_path, "r", encoding="utf-8") as fh:
		return yaml.safe_load(fh)


def _select_best_adaptive_config(best_configs: dict) -> dict:
	"""Select best adaptive config across solvers by validation MSE."""
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


def _support_set(beta: np.ndarray, tol: float) -> set[int]:
	"""Return active-feature index set with |beta_j| > tol."""
	return set(np.flatnonzero(np.abs(beta) > tol).tolist())


def jaccard_index(set_a: set[int], set_b: set[int]) -> float:
	"""Compute Jaccard index J(A, B) = |A∩B| / |A∪B|.

	For two empty sets, this function returns 1.0.
	"""
	union = set_a | set_b
	if not union:
		return 1.0
	return float(len(set_a & set_b) / len(union))


def _pairwise_jaccard(supports: list[set[int]]) -> list[float]:
	"""Compute all pairwise Jaccard scores across support sets."""
	return [
		jaccard_index(support_a, support_b)
		for support_a, support_b in itertools.combinations(supports, 2)
	]


def run_stability_analysis(
	config_path: str = "configs/default.yaml",
	dynamic_config_path: str = "configs/dynamic_reweight.yaml",
	ista_config_path: str = "configs/ista.yaml",
	fista_config_path: str = "configs/fista.yaml",
	best_config_path: str | None = None,
) -> pd.DataFrame:
	"""Run T14 stability analysis across seeds and save stability table.

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
		Stability table with one row per method.
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
	lasso_cfg = cfg["model"]["lasso"]

	eval_cfg = cfg.get("evaluation", {})
	stability_cfg = eval_cfg.get("stability", {})
	if "seeds" not in stability_cfg:
		raise NotImplementedError(
			"T14 requires evaluation.stability.seeds in config YAML. "
			"Provide 5 seeds for pairwise Jaccard stability analysis."
		)

	seeds = [int(seed) for seed in stability_cfg["seeds"]]
	if len(seeds) < 2:
		raise ValueError("T14 requires at least 2 seeds to compute Jaccard pairs")
	nonzero_tol = float(stability_cfg.get("nonzero_tol", 1.0e-8))

	best_path = (
		Path(best_config_path)
		if best_config_path is not None
		else Path(outputs_cfg["logs"]) / "best_configs.yaml"
	)
	best_configs = _load_best_configs(best_path)
	best_adaptive = _select_best_adaptive_config(best_configs)
	best_dyn_ista = _get_dynamic_cfg(best_configs, "ista")
	best_dyn_fista = _get_dynamic_cfg(best_configs, "fista")

	supports_by_method: dict[str, list[set[int]]] = {
		"ridge": [],
		"lasso": [],
		"adaptive_lasso": [],
		"dynamic_reweighted_lasso_ista": [],
		"dynamic_reweighted_lasso_fista": [],
	}

	for seed in seeds:
		set_seed(seed)
		X_train, X_val, X_test, y_train, y_val, y_test, feature_names = build_dataset(
			raw_path=data_cfg["raw_path"],
			target_col=data_cfg["target_col"],
			val_size=float(data_cfg["val_size"]),
			test_size=float(data_cfg["test_size"]),
			seed=seed,
			numeric_impute_strategy=data_cfg.get("numeric_impute_strategy", "median"),
			categorical_impute_strategy=data_cfg.get(
				"categorical_impute_strategy", "most_frequent"
			),
			drop_cols=data_cfg.get("drop_cols"),
		)

		# Center y so zero-intercept custom solvers predict correctly
		y_mean = float(y_train.mean())
		y_train_c = y_train - y_mean

		_ = X_val, X_test, y_val, y_test, feature_names

		adaptive_solver = str(best_adaptive["solver"])
		adaptive_alpha, adaptive_max_iter, adaptive_tol = _solver_optim_params(
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
			seed=seed,
		)
		supports_by_method["ridge"].append(
			_support_set(np.asarray(ridge_results["beta"]), nonzero_tol)
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
			seed=seed,
		)
		supports_by_method["lasso"].append(
			_support_set(np.asarray(lasso_results["beta"]), nonzero_tol)
		)

		beta_a, _obj_a, _sp_a, _rt_a = run_adaptive_lasso(
			X=X_train,
			y=y_train_c,
			lam=float(best_adaptive["lam"]),
			ridge_coef=np.asarray(ridge_results["beta"]),
			gamma=float(best_adaptive["gamma"]),
			eps=float(best_adaptive["eps"]),
			alpha=adaptive_alpha,
			max_iter=adaptive_max_iter,
			tol=adaptive_tol,
			solver=adaptive_solver,
		)
		supports_by_method["adaptive_lasso"].append(_support_set(beta_a, nonzero_tol))

		ista_alpha, ista_max_iter, ista_tol = _solver_optim_params(
			solver="ista",
			optim_cfg=optim_cfg,
			ista_cfg=ista_cfg,
			fista_cfg=fista_cfg,
			X_train=X_train,
		)
		beta_di, _obj_di, _sp_di, _wt_di, _ic_di, _rt_di = run_dynamic_reweighted_lasso(
			X=X_train,
			y=y_train_c,
			lam=float(best_dyn_ista["lam"]),
			gamma=float(best_dyn_ista["gamma"]),
			eps=float(best_dyn_ista["eps"]),
			w_max=float(dynamic_cfg["w_max"]),
			alpha=ista_alpha,
			max_outer_iter=int(best_dyn_ista["max_outer_iter"]),
			inner_max_iter=int(dynamic_cfg.get("inner_max_iter", ista_max_iter)),
			inner_tol=float(dynamic_cfg.get("inner_tol", ista_tol)),
			solver="ista",
		)
		supports_by_method["dynamic_reweighted_lasso_ista"].append(
			_support_set(beta_di, nonzero_tol)
		)

		fista_alpha, fista_max_iter, fista_tol = _solver_optim_params(
			solver="fista",
			optim_cfg=optim_cfg,
			ista_cfg=ista_cfg,
			fista_cfg=fista_cfg,
			X_train=X_train,
		)
		beta_df, _obj_df, _sp_df, _wt_df, _ic_df, _rt_df = run_dynamic_reweighted_lasso(
			X=X_train,
			y=y_train_c,
			lam=float(best_dyn_fista["lam"]),
			gamma=float(best_dyn_fista["gamma"]),
			eps=float(best_dyn_fista["eps"]),
			w_max=float(dynamic_cfg["w_max"]),
			alpha=fista_alpha,
			max_outer_iter=int(best_dyn_fista["max_outer_iter"]),
			inner_max_iter=int(dynamic_cfg.get("inner_max_iter", fista_max_iter)),
			inner_tol=float(dynamic_cfg.get("inner_tol", fista_tol)),
			solver="fista",
		)
		supports_by_method["dynamic_reweighted_lasso_fista"].append(
			_support_set(beta_df, nonzero_tol)
		)

	rows: list[dict] = []
	for method, supports in supports_by_method.items():
		jaccard_scores = _pairwise_jaccard(supports)
		if not jaccard_scores:
			raise ValueError(
				f"Method {method} has insufficient runs for pairwise Jaccard analysis"
			)
		rows.append(
			{
				"method": method,
				"mean_jaccard": float(np.mean(jaccard_scores)),
				"std_jaccard": float(np.std(jaccard_scores)),
				"min_jaccard": float(np.min(jaccard_scores)),
				"max_jaccard": float(np.max(jaccard_scores)),
				"num_runs": int(len(supports)),
				"num_pairs": int(len(jaccard_scores)),
			}
		)

	stability_df = pd.DataFrame(rows).sort_values("method").reset_index(drop=True)

	tables_dir = Path(outputs_cfg["tables"])
	logs_dir = Path(outputs_cfg["logs"])
	table_path = save_table(stability_df, tables_dir, "stability_table.csv")
	save_log(
		{
			"task": "T14-stability-analysis",
			"config_path": config_path,
			"dynamic_config_path": dynamic_config_path,
			"ista_config_path": ista_config_path,
			"fista_config_path": fista_config_path,
			"best_config_path": str(best_path),
			"seeds": seeds,
			"nonzero_tol": nonzero_tol,
			"table_path": str(table_path),
			"rows": rows,
		},
		logs_dir,
		"t14_stability_log.json",
	)

	logger.info("Saved stability table: %s", table_path)
	return stability_df


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	"""Parse command-line arguments for T14 stability analysis."""
	parser = argparse.ArgumentParser(description="T14 - Stability analysis")
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
	"""CLI entry point for T14 stability analysis."""
	args = _parse_args(argv)
	return run_stability_analysis(
		config_path=args.config,
		dynamic_config_path=args.dynamic_config,
		ista_config_path=args.ista_config,
		fista_config_path=args.fista_config,
		best_config_path=args.best_config,
	)


if __name__ == "__main__":
	main()
