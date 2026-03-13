"""
Fairness audit for separating accuracy and optimizer-efficiency comparisons.

Task: T15 (owner: Both)

Consumed by:
	T25 Results and Discussion
	T26 Limitations section

I/O contract:
	run_fairness_audit(config_path, full_table_path) -> dict
		Requires full comparison table from T13.
		Saves:
			- outputs/tables/accuracy_comparison.csv
			- outputs/tables/optimizer_efficiency_comparison.csv
			- outputs/logs/fairness_audit.json
		Returns audit summary dictionary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.utils.io import save_log, save_table
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


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
	"""Run T15 fairness audit and save separated comparison tables.

	Parameters
	----------
	config_path : str
		Path to default YAML config.
	full_table_path : str or None
		Optional explicit path to T13 full comparison table. If None, defaults
		to outputs/tables/full_comparison_table.csv from config.

	Returns
	-------
	dict
		Audit summary with compliance booleans and output paths.
	"""
	with open(config_path, "r", encoding="utf-8") as fh:
		cfg = yaml.safe_load(fh)

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

	# Accuracy comparison: all methods including sklearn baselines.
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

	# Optimizer efficiency: custom ISTA/FISTA variants only (exclude sklearn rows).
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
		"full_table_path": str(full_path.resolve()),
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	"""Parse command-line arguments for T15 fairness audit."""
	parser = argparse.ArgumentParser(description="T15 - Fairness audit")
	parser.add_argument(
		"--config",
		type=str,
		default="configs/default.yaml",
		help="Path to default YAML config.",
	)
	parser.add_argument(
		"--full-table",
		type=str,
		default=None,
		help="Optional path to full_comparison_table.csv",
	)
	return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
	"""CLI entry point for T15 fairness audit."""
	args = _parse_args(argv)
	return run_fairness_audit(config_path=args.config, full_table_path=args.full_table)


if __name__ == "__main__":
	main()
