"""
Evaluation utilities for fairness and stretch analyses.

Tasks:
	T15 Fairness audit (owner: Both)
	T19 Synthetic correlated-design support recovery (owner: Yash Karecha)
	T20 Feature interpretation report (owner: Both)

Depends on:
	T5  src/optim/ista.py
	T6  src/optim/fista.py
	T7  src/optim/reweight.py
	T11 src/models/dynamic_reweighted_lasso.py
	T13 outputs/tables/full_comparison_table.csv (T15), best_configs.yaml (T20 optional)

Consumed by:
	T25 Results and Discussion
	T26 Limitations section

I/O contracts:
	run_fairness_audit(config_path, full_table_path) -> dict
	run_synthetic_correlated_design_experiment(...) -> pd.DataFrame
	run_feature_interpretation_report(...) -> pd.DataFrame
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.linalg import toeplitz
from sklearn.linear_model import Lasso, Ridge

from src.data.split import build_dataset
from src.metrics.regression_metrics import compute_regression_metrics
from src.metrics.sparsity_metrics import count_nonzero
from src.models.adaptive_lasso import run_adaptive_lasso
from src.models.dynamic_reweighted_lasso import run_dynamic_reweighted_lasso
from src.utils.io import save_log, save_table
from src.utils.logging_utils import get_logger
from src.utils.seeds import set_seed

logger = get_logger(__name__)


def _load_yaml(path: str) -> dict:
	"""Load YAML config as a dictionary."""
	with open(path, "r", encoding="utf-8") as fh:
		content = fh.read()
	try:
		cfg = yaml.safe_load(content)
	except yaml.YAMLError:
		# Some project configs may contain tab indentation; normalize to spaces
		# for tolerant loading without mutating files on disk.
		cfg = yaml.safe_load(content.replace("\t", "  "))
	if not isinstance(cfg, dict):
		raise ValueError(f"Expected mapping YAML at {path}")
	return cfg


def _resolve_alpha(X: np.ndarray, alpha_cfg: float | None) -> float:
	"""Resolve alpha from config, or compute alpha = 1/L from X."""
	if alpha_cfg is not None:
		return float(alpha_cfg)
	n = X.shape[0]
	lipschitz = float(np.linalg.eigvalsh(X.T @ X).max() / n)
	if lipschitz <= 0.0:
		raise ValueError("Computed non-positive Lipschitz constant; cannot set alpha")
	return 1.0 / lipschitz


def _load_full_table(path: Path) -> pd.DataFrame:
	"""Load T13 full comparison table with strict presence checks."""
	if not path.exists():
		raise NotImplementedError(
			"Missing outputs/tables/full_comparison_table.csv. "
			"Run T13 first to produce the full comparison table before T15."
		)
	return pd.read_csv(path)


def _validate_required_methods(df: pd.DataFrame) -> None:
	"""Validate that all five T13 methods are present."""
	required_methods = {
		"ridge",
		"lasso",
		"adaptive_lasso",
		"dynamic_reweighted_lasso_ista",
		"dynamic_reweighted_lasso_fista",
	}
	present_methods = set(df["method"].astype(str).tolist())
	missing = sorted(required_methods - present_methods)
	if missing:
		raise NotImplementedError(
			"T15 fairness audit requires all five methods in the full table. "
			f"Missing methods: {missing}."
		)


def run_fairness_audit(
	config_path: str = "configs/default.yaml",
	full_table_path: str | None = None,
) -> dict:
	"""Run T15 fairness audit and save separated comparison tables."""
	cfg = _load_yaml(config_path)

	outputs_cfg = cfg["outputs"]
	tables_dir = Path(outputs_cfg["tables"])
	logs_dir = Path(outputs_cfg["logs"])

	full_path = (
		Path(full_table_path)
		if full_table_path is not None
		else tables_dir / "full_comparison_table.csv"
	)
	full_df = _load_full_table(full_path)

	required_cols = {
		"method",
		"solver",
		"val_mse",
		"val_mae",
		"test_mse",
		"test_mae",
		"nnz",
		"runtime_s",
		"iterations",
	}
	missing_cols = sorted(required_cols - set(full_df.columns))
	if missing_cols:
		raise NotImplementedError(
			"T15 fairness audit requires full comparison columns from T13. "
			f"Missing columns: {missing_cols}."
		)

	_validate_required_methods(full_df)

	accuracy_df = full_df[
		[
			"method",
			"solver",
			"val_mse",
			"val_mae",
			"test_mse",
			"test_mae",
			"nnz",
			"runtime_s",
			"iterations",
		]
	].copy()
	accuracy_df = accuracy_df.sort_values("test_mse").reset_index(drop=True)

	efficiency_df = full_df[~full_df["solver"].astype(str).str.startswith("sklearn_")]
	if efficiency_df.empty:
		raise NotImplementedError(
			"No custom solver rows found for optimizer efficiency comparison."
		)
	efficiency_df = efficiency_df[
		["method", "solver", "runtime_s", "iterations", "val_mse", "test_mse"]
	].copy()
	efficiency_df = efficiency_df.sort_values(["runtime_s", "iterations"]).reset_index(
		drop=True
	)

	accuracy_path = save_table(accuracy_df, tables_dir, "accuracy_comparison.csv")
	efficiency_path = save_table(
		efficiency_df,
		tables_dir,
		"optimizer_efficiency_comparison.csv",
	)

	best_sklearn_test_mse = float(
		full_df[full_df["solver"].astype(str).str.startswith("sklearn_")][
			"test_mse"
		].min()
	)
	best_custom_test_mse = float(
		full_df[~full_df["solver"].astype(str).str.startswith("sklearn_")][
			"test_mse"
		].min()
	)
	data_supports_custom_beats_claim = bool(best_custom_test_mse < best_sklearn_test_mse)

	summary = {
		"task": "T15-fairness-audit",
		"full_table_path": str(full_path),
		"accuracy_comparison_path": str(accuracy_path),
		"optimizer_efficiency_path": str(efficiency_path),
		"accuracy_includes_all_methods": bool(len(accuracy_df) >= 5),
		"efficiency_only_custom_solvers": bool(
			(~efficiency_df["solver"].astype(str).str.startswith("sklearn_")).all()
		),
		"best_sklearn_test_mse": best_sklearn_test_mse,
		"best_custom_test_mse": best_custom_test_mse,
		"data_supports_custom_beats_sklearn_claim": data_supports_custom_beats_claim,
		"note": (
			"Do not claim custom solver superiority over sklearn unless "
			"data_supports_custom_beats_sklearn_claim is true."
		),
	}

	save_log(summary, logs_dir, "fairness_audit.json")
	logger.info("Saved fairness audit outputs to %s and %s", accuracy_path, efficiency_path)
	return summary


def _support_set(beta: np.ndarray, tol: float) -> set[int]:
	"""Return active-feature index set with |beta_j| > tol."""
	return set(np.flatnonzero(np.abs(beta) > tol).tolist())


def _jaccard_index(set_a: set[int], set_b: set[int]) -> float:
	"""Compute Jaccard index J(A, B) = |A∩B| / |A∪B| with empty-empty -> 1."""
	union = set_a | set_b
	if not union:
		return 1.0
	return float(len(set_a & set_b) / len(union))


def _f1_score(precision: float, recall: float) -> float:
	"""Return F1 score from precision and recall, guarding zero denominator."""
	denom = precision + recall
	if denom == 0.0:
		return 0.0
	return float(2.0 * precision * recall / denom)


def _support_recovery_metrics(
	selected: set[int],
	true_support: set[int],
) -> dict[str, float]:
	"""Compute support-recovery precision/recall/F1/Jaccard metrics."""
	tp = len(selected & true_support)
	fp = len(selected - true_support)
	fn = len(true_support - selected)
	precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
	recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
	return {
		"precision": precision,
		"recall": recall,
		"f1": _f1_score(precision, recall),
		"jaccard": _jaccard_index(selected, true_support),
		"support_size": float(len(selected)),
	}


def _generate_toeplitz_synthetic_regression(
	n_samples: int,
	n_features: int,
	support_size: int,
	rho: float,
	noise_std: float,
	beta_signal: float,
	seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Generate Toeplitz-correlated linear regression data with known support."""
	if not (0.0 <= rho < 1.0):
		raise ValueError("rho must satisfy 0 <= rho < 1")
	if support_size <= 0 or support_size > n_features:
		raise ValueError("support_size must be in [1, n_features]")

	rng = np.random.default_rng(seed)
	col_idx = np.arange(n_features)
	cov = toeplitz(rho ** col_idx)
	X = rng.multivariate_normal(
		mean=np.zeros(n_features, dtype=np.float64),
		cov=cov,
		size=n_samples,
	)

	beta_true = np.zeros(n_features, dtype=np.float64)
	beta_true[:support_size] = beta_signal
	noise = rng.normal(loc=0.0, scale=noise_std, size=n_samples)
	y = X @ beta_true + noise

	col_mean = X.mean(axis=0)
	col_std = X.std(axis=0)
	col_std[col_std == 0.0] = 1.0
	X_std = (X - col_mean) / col_std

	return X_std.astype(np.float64), y.astype(np.float64), beta_true


def run_synthetic_correlated_design_experiment(
	config_path: str = "configs/default.yaml",
	dynamic_config_path: str = "configs/dynamic_reweight.yaml",
	ista_config_path: str = "configs/ista.yaml",
	fista_config_path: str = "configs/fista.yaml",
	best_config_path: str | None = None,
	n_samples: int = 240,
	n_features: int = 300,
	support_size: int = 20,
	rho: float = 0.8,
	noise_std: float = 1.0,
	beta_signal: float = 2.5,
	n_trials: int = 5,
	nonzero_tol: float = 1.0e-6,
	ridge_alpha: float = 1.0,
) -> pd.DataFrame:
	"""Run T19 synthetic Toeplitz experiment and save support-recovery table."""
	cfg = _load_yaml(config_path)
	dynamic_cfg = _load_yaml(dynamic_config_path)
	ista_cfg = _load_yaml(ista_config_path)
	fista_cfg = _load_yaml(fista_config_path)

	outputs_cfg = cfg["outputs"]
	optim_cfg = cfg["optim"]

	best_cfg: dict[str, dict] = {}
	if best_config_path is not None:
		best_path = Path(best_config_path)
	else:
		best_path = Path(outputs_cfg["logs"]) / "best_configs.yaml"
	if best_path.exists():
		with open(best_path, "r", encoding="utf-8") as fh:
			payload = yaml.safe_load(fh)
			if isinstance(payload, dict):
				best_cfg = payload

	dyn_ista = best_cfg.get("dynamic_reweighted_lasso_ista", {})
	dyn_fista = best_cfg.get("dynamic_reweighted_lasso_fista", {})
	adaptive_candidates = [
		val for key, val in best_cfg.items() if str(key).startswith("adaptive_lasso_")
	]
	best_adaptive = (
		min(adaptive_candidates, key=lambda row: float(row["val_mse"]))
		if adaptive_candidates
		else {}
	)

	lam_default = float(dynamic_cfg["lam"])
	gamma_default = float(dynamic_cfg["gamma"])
	eps_default = float(dynamic_cfg["eps"])
	max_outer_default = int(dynamic_cfg["max_outer_iter"])
	w_max = float(dynamic_cfg["w_max"])

	results_rows: list[dict] = []
	trial_seeds = list(range(1, n_trials + 1))
	true_support = set(range(support_size))

	for trial_seed in trial_seeds:
		set_seed(trial_seed)
		X, y, _beta_true = _generate_toeplitz_synthetic_regression(
			n_samples=n_samples,
			n_features=n_features,
			support_size=support_size,
			rho=rho,
			noise_std=noise_std,
			beta_signal=beta_signal,
			seed=trial_seed,
		)

		alpha_ista = _resolve_alpha(X, ista_cfg.get("alpha", optim_cfg.get("alpha")))
		alpha_fista = _resolve_alpha(X, fista_cfg.get("alpha", optim_cfg.get("alpha")))
		max_iter_ista = int(ista_cfg.get("max_iter", optim_cfg["max_iter"]))
		max_iter_fista = int(fista_cfg.get("max_iter", optim_cfg["max_iter"]))
		tol_ista = float(ista_cfg.get("tol", optim_cfg["tol"]))
		tol_fista = float(fista_cfg.get("tol", optim_cfg["tol"]))

		lasso_lam = float(best_adaptive.get("lam", lam_default))
		lasso = Lasso(alpha=lasso_lam, max_iter=10000)
		lasso.fit(X, y)
		beta_lasso = lasso.coef_.astype(np.float64)

		ridge = Ridge(alpha=ridge_alpha)
		ridge.fit(X, y)
		ridge_coef = ridge.coef_.astype(np.float64)

		adaptive_lam = float(best_adaptive.get("lam", lam_default))
		adaptive_gamma = float(best_adaptive.get("gamma", gamma_default))
		adaptive_eps = float(best_adaptive.get("eps", eps_default))

		beta_adapt_ista, _, _, _ = run_adaptive_lasso(
			X=X,
			y=y,
			lam=adaptive_lam,
			ridge_coef=ridge_coef,
			gamma=adaptive_gamma,
			eps=adaptive_eps,
			alpha=alpha_ista,
			max_iter=max_iter_ista,
			tol=tol_ista,
			solver="ista",
		)
		beta_adapt_fista, _, _, _ = run_adaptive_lasso(
			X=X,
			y=y,
			lam=adaptive_lam,
			ridge_coef=ridge_coef,
			gamma=adaptive_gamma,
			eps=adaptive_eps,
			alpha=alpha_fista,
			max_iter=max_iter_fista,
			tol=tol_fista,
			solver="fista",
		)

		beta_dyn_ista, _, _, _, _, _ = run_dynamic_reweighted_lasso(
			X=X,
			y=y,
			lam=float(dyn_ista.get("lam", lam_default)),
			gamma=float(dyn_ista.get("gamma", gamma_default)),
			eps=float(dyn_ista.get("eps", eps_default)),
			w_max=w_max,
			alpha=alpha_ista,
			max_outer_iter=int(dyn_ista.get("max_outer_iter", max_outer_default)),
			inner_max_iter=int(dynamic_cfg.get("inner_max_iter", max_iter_ista)),
			inner_tol=float(dynamic_cfg.get("inner_tol", tol_ista)),
			solver="ista",
		)
		beta_dyn_fista, _, _, _, _, _ = run_dynamic_reweighted_lasso(
			X=X,
			y=y,
			lam=float(dyn_fista.get("lam", lam_default)),
			gamma=float(dyn_fista.get("gamma", gamma_default)),
			eps=float(dyn_fista.get("eps", eps_default)),
			w_max=w_max,
			alpha=alpha_fista,
			max_outer_iter=int(dyn_fista.get("max_outer_iter", max_outer_default)),
			inner_max_iter=int(dynamic_cfg.get("inner_max_iter", max_iter_fista)),
			inner_tol=float(dynamic_cfg.get("inner_tol", tol_fista)),
			solver="fista",
		)

		method_betas = {
			"lasso": beta_lasso,
			"adaptive_lasso_ista": beta_adapt_ista,
			"adaptive_lasso_fista": beta_adapt_fista,
			"dynamic_reweighted_lasso_ista": beta_dyn_ista,
			"dynamic_reweighted_lasso_fista": beta_dyn_fista,
		}

		for method_name, beta_hat in method_betas.items():
			est_support = _support_set(beta_hat, nonzero_tol)
			metric_row = _support_recovery_metrics(est_support, true_support)
			row = {**metric_row, "method": method_name, "trial_seed": trial_seed}
			results_rows.append(row)

	results_df = pd.DataFrame(results_rows)
	summary_df = (
		results_df.groupby("method", as_index=False)
		.agg(
			precision_mean=("precision", "mean"),
			precision_std=("precision", "std"),
			recall_mean=("recall", "mean"),
			recall_std=("recall", "std"),
			f1_mean=("f1", "mean"),
			f1_std=("f1", "std"),
			jaccard_mean=("jaccard", "mean"),
			jaccard_std=("jaccard", "std"),
			support_size_mean=("support_size", "mean"),
			support_size_std=("support_size", "std"),
		)
		.sort_values("jaccard_mean", ascending=False)
		.reset_index(drop=True)
	)

	tables_dir = Path(outputs_cfg["tables"])
	logs_dir = Path(outputs_cfg["logs"])
	table_path = save_table(summary_df, tables_dir, "t19_synthetic_support_recovery.csv")

	save_log(
		{
			"task": "T19-synthetic-correlated-design",
			"n_samples": n_samples,
			"n_features": n_features,
			"support_size": support_size,
			"rho": rho,
			"noise_std": noise_std,
			"beta_signal": beta_signal,
			"n_trials": n_trials,
			"nonzero_tol": nonzero_tol,
			"results_table_path": str(table_path),
			"best_method_by_jaccard": (
				summary_df.iloc[0]["method"] if not summary_df.empty else None
			),
		},
		logs_dir,
		"t19_synthetic_support_recovery.json",
	)

	logger.info("Saved T19 synthetic support-recovery summary: %s", table_path)
	return summary_df


def _interpret_feature_name(feature: str, sign: float) -> str:
	"""Return a short house-pricing interpretation string for one feature."""
	sign_text = "increases" if sign >= 0.0 else "decreases"
	name = feature.lower()

	if "neighborhood_" in name:
		return f"Neighborhood effect: this location-level indicator {sign_text} price."
	if "qual" in name or "cond" in name:
		return f"Quality/condition proxy: better values typically {sign_text} price."
	if "sf" in name or "area" in name or "flr" in name or "grliv" in name:
		return f"Size-related feature: larger usable space generally {sign_text} price."
	if "year" in name or "yr" in name or "built" in name or "remod" in name:
		return f"Age/renovation signal: newer or renovated homes usually {sign_text} price."
	if "garage" in name or "car" in name:
		return f"Parking/garage capacity signal: more capacity often {sign_text} price."
	if "bath" in name or "bed" in name or "kitchen" in name or "room" in name:
		return f"Layout/amenity feature: configuration differences can {sign_text} price."
	if "lot" in name or "porch" in name or "deck" in name:
		return f"Lot/outdoor space proxy: outdoor utility tends to {sign_text} price."
	return f"General structural/categorical feature that appears to {sign_text} price."


def run_feature_interpretation_report(
	config_path: str = "configs/default.yaml",
	dynamic_config_path: str = "configs/dynamic_reweight.yaml",
	ista_config_path: str = "configs/ista.yaml",
	fista_config_path: str = "configs/fista.yaml",
	best_config_path: str | None = None,
	solver: str = "fista",
	top_k: int = 20,
	nonzero_tol: float = 1.0e-8,
) -> pd.DataFrame:
	"""Run T20 feature interpretation report from dynamic model coefficients."""
	if solver not in {"ista", "fista"}:
		raise ValueError("solver must be 'ista' or 'fista'")

	cfg = _load_yaml(config_path)
	dynamic_cfg = _load_yaml(dynamic_config_path)
	ista_cfg = _load_yaml(ista_config_path)
	fista_cfg = _load_yaml(fista_config_path)

	data_cfg = cfg["data"]
	outputs_cfg = cfg["outputs"]
	optim_cfg = cfg["optim"]

	best_cfg: dict[str, dict] = {}
	if best_config_path is not None:
		best_path = Path(best_config_path)
	else:
		best_path = Path(outputs_cfg["logs"]) / "best_configs.yaml"
	if best_path.exists():
		with open(best_path, "r", encoding="utf-8") as fh:
			payload = yaml.safe_load(fh)
			if isinstance(payload, dict):
				best_cfg = payload

	dyn_key = f"dynamic_reweighted_lasso_{solver}"
	dyn_best = best_cfg.get(dyn_key, {})

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

	solver_cfg = ista_cfg if solver == "ista" else fista_cfg
	alpha = _resolve_alpha(X_train, solver_cfg.get("alpha", optim_cfg.get("alpha")))
	max_iter = int(solver_cfg.get("max_iter", optim_cfg["max_iter"]))
	tol = float(solver_cfg.get("tol", optim_cfg["tol"]))

	beta, _obj_trace, _sp_trace, _w_trace, inner_counts, runtime_s = (
		run_dynamic_reweighted_lasso(
			X=X_train,
			y=y_train,
			lam=float(dyn_best.get("lam", dynamic_cfg["lam"])),
			gamma=float(dyn_best.get("gamma", dynamic_cfg["gamma"])),
			eps=float(dyn_best.get("eps", dynamic_cfg["eps"])),
			w_max=float(dynamic_cfg["w_max"]),
			alpha=alpha,
			max_outer_iter=int(
				dyn_best.get("max_outer_iter", dynamic_cfg["max_outer_iter"])
			),
			inner_max_iter=int(dynamic_cfg.get("inner_max_iter", max_iter)),
			inner_tol=float(dynamic_cfg.get("inner_tol", tol)),
			solver=solver,
		)
	)

	val_metrics = compute_regression_metrics(y_val, X_val @ beta)
	test_metrics = compute_regression_metrics(y_test, X_test @ beta)

	coef_series = pd.Series(beta, index=feature_names)
	coef_series = coef_series[np.abs(coef_series) > nonzero_tol]
	coef_series = coef_series.reindex(coef_series.abs().sort_values(ascending=False).index)
	top_series = coef_series.head(top_k)

	report_df = pd.DataFrame(
		{
			"feature": top_series.index.tolist(),
			"coefficient": top_series.values.astype(float),
			"abs_coefficient": np.abs(top_series.values.astype(float)),
		}
	)
	report_df["direction"] = np.where(
		report_df["coefficient"] >= 0.0,
		"positive",
		"negative",
	)
	report_df["interpretation"] = [
		_interpret_feature_name(str(feat), float(coef))
		for feat, coef in zip(report_df["feature"], report_df["coefficient"])
	]

	tables_dir = Path(outputs_cfg["tables"])
	logs_dir = Path(outputs_cfg["logs"])
	table_path = save_table(report_df, tables_dir, "t20_feature_interpretation.csv")

	save_log(
		{
			"task": "T20-feature-interpretation",
			"solver": solver,
			"top_k": top_k,
			"nonzero_tol": nonzero_tol,
			"nnz_total": int(count_nonzero(beta, tol=nonzero_tol)),
			"iterations_total": int(sum(inner_counts)),
			"runtime_s": float(runtime_s),
			"val_mse": float(val_metrics["mse"]),
			"val_mae": float(val_metrics["mae"]),
			"test_mse": float(test_metrics["mse"]),
			"test_mae": float(test_metrics["mae"]),
			"table_path": str(table_path),
			"note": (
				"Interpretations are domain-guided qualitative summaries for Ames "
				"housing features to support T25 discussion."
			),
		},
		logs_dir,
		"t20_feature_interpretation.json",
	)

	logger.info("Saved T20 feature interpretation report: %s", table_path)
	return report_df


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	"""Parse command-line arguments for T15, T19, and T20."""
	parser = argparse.ArgumentParser(description="T15/T19/T20 evaluation utilities")
	parser.add_argument(
		"--task",
		type=str,
		default="t15",
		choices=["t15", "t19", "t20"],
		help="Which evaluation task to run: t15 (fairness), t19 (synthetic), t20 (feature report).",
	)
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
		help="Optional path to T12 best_configs.yaml.",
	)
	parser.add_argument(
		"--full-table",
		type=str,
		default=None,
		help="Optional path to full_comparison_table.csv",
	)
	parser.add_argument("--n-samples", type=int, default=240)
	parser.add_argument("--n-features", type=int, default=300)
	parser.add_argument("--support-size", type=int, default=20)
	parser.add_argument("--rho", type=float, default=0.8)
	parser.add_argument("--noise-std", type=float, default=1.0)
	parser.add_argument("--beta-signal", type=float, default=2.5)
	parser.add_argument("--n-trials", type=int, default=5)
	parser.add_argument("--nonzero-tol", type=float, default=1.0e-6)
	parser.add_argument("--ridge-alpha", type=float, default=1.0)
	parser.add_argument("--solver", type=str, default="fista", choices=["ista", "fista"])
	parser.add_argument("--top-k", type=int, default=20)
	return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
	"""CLI entry point for T15 fairness audit, T19, and T20."""
	args = _parse_args(argv)

	if args.task == "t15":
		return run_fairness_audit(config_path=args.config, full_table_path=args.full_table)

	if args.task == "t19":
		df = run_synthetic_correlated_design_experiment(
			config_path=args.config,
			dynamic_config_path=args.dynamic_config,
			ista_config_path=args.ista_config,
			fista_config_path=args.fista_config,
			best_config_path=args.best_config,
			n_samples=args.n_samples,
			n_features=args.n_features,
			support_size=args.support_size,
			rho=args.rho,
			noise_std=args.noise_std,
			beta_signal=args.beta_signal,
			n_trials=args.n_trials,
			nonzero_tol=args.nonzero_tol,
			ridge_alpha=args.ridge_alpha,
		)
		return {
			"task": "t19",
			"rows": int(len(df)),
			"best_method": (
				str(df.iloc[0]["method"]) if not df.empty and "method" in df.columns else None
			),
		}

	df = run_feature_interpretation_report(
		config_path=args.config,
		dynamic_config_path=args.dynamic_config,
		ista_config_path=args.ista_config,
		fista_config_path=args.fista_config,
		best_config_path=args.best_config,
		solver=args.solver,
		top_k=args.top_k,
		nonzero_tol=args.nonzero_tol,
	)
	return {
		"task": "t20",
		"rows": int(len(df)),
		"top_feature": str(df.iloc[0]["feature"]) if not df.empty else None,
	}


if __name__ == "__main__":
	main()
