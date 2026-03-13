"""Preprocessing utilities: imputation, one-hot encoding, and standardisation.

Output contract
---------------
* Column naming: ``pd.get_dummies`` with ``drop_first=False``, then all
  non-alphanumeric/underscore characters replaced with ``_``, columns sorted
  lexicographically.  Example: ``"MS Zoning"`` + value ``"RL"`` →
  ``"MS_Zoning_RL"``.
* Numeric NaNs: filled with the training-set **median**.
* Categorical NaNs: filled with the training-set **mode** (most frequent).
* X dtype: ``numpy.float64``.
* y dtype: ``numpy.float64``.
* Shape: all splits share the identical ``p`` columns produced by
  :func:`fit_preprocessor`; val/test columns are aligned via ``reindex``.
* Standardisation: fit on training only; zero-variance features use
  ``scale = 1.0`` so they are left unchanged (not dropped).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sanitise_columns(columns: pd.Index) -> pd.Index:
    """Replace whitespace and non-identifier characters with underscores."""
    return pd.Index(
        [re.sub(r"[^A-Za-z0-9_]", "_", col.replace(" ", "_")) for col in columns]
    )


def _identify_columns(
    df: pd.DataFrame, target_col: str
) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols) for feature columns only."""
    feature_df = df.drop(columns=[target_col], errors="ignore")
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric_cols, categorical_cols


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_preprocessor(
    df_train: pd.DataFrame,
    target_col: str,
    numeric_impute_strategy: str = "median",
    categorical_impute_strategy: str = "most_frequent",
) -> dict:
    """Fit imputation statistics, OHE column list, and scaler on training data.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training split that must contain ``target_col``.
    target_col : str
        Name of the regression target column.
    numeric_impute_strategy : str
        ``"median"`` (default) or ``"mean"`` for numeric imputation.
    categorical_impute_strategy : str
        ``"most_frequent"`` (default) — only mode imputation is currently
        supported for categorical columns.

    Returns
    -------
    dict
        Preprocessor state with keys ``target_col``, ``numeric_cols``,
        ``categorical_cols``, ``numeric_fill``, ``categorical_fill``,
        ``ohe_columns``, ``scaler_mean``, ``scaler_scale``.  Pass this dict
        to :func:`apply_preprocessor` for all splits.
    """
    numeric_cols, categorical_cols = _identify_columns(df_train, target_col)

    # --- Fit numeric imputation values ---
    numeric_fill: dict[str, float] = {}
    for col in numeric_cols:
        if numeric_impute_strategy == "mean":
            numeric_fill[col] = float(df_train[col].mean())
        else:
            numeric_fill[col] = float(df_train[col].median())

    # --- Fit categorical imputation values ---
    categorical_fill: dict[str, str] = {}
    for col in categorical_cols:
        mode_series = df_train[col].mode()
        categorical_fill[col] = str(mode_series.iloc[0]) if len(mode_series) > 0 else "MISSING"

    # --- Apply imputation to get the reference OHE column set ---
    df_imp = df_train.drop(columns=[target_col]).copy()
    for col, val in numeric_fill.items():
        df_imp[col] = df_imp[col].fillna(val)
    for col, val in categorical_fill.items():
        df_imp[col] = df_imp[col].fillna(val)

    df_enc = pd.get_dummies(df_imp, columns=categorical_cols, drop_first=False)
    df_enc.columns = _sanitise_columns(df_enc.columns)
    ohe_columns: list[str] = sorted(df_enc.columns.tolist())
    df_enc = df_enc.reindex(columns=ohe_columns, fill_value=0.0)

    # --- Fit standardisation on training matrix ---
    X_train = df_enc.values.astype(np.float64)
    scaler_mean: np.ndarray = X_train.mean(axis=0)
    scaler_std: np.ndarray = X_train.std(axis=0)
    # Zero-variance columns: keep unchanged (scale = 1.0 so division is safe)
    scaler_scale: np.ndarray = np.where(scaler_std == 0.0, 1.0, scaler_std)

    return {
        "target_col": target_col,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "numeric_fill": numeric_fill,
        "categorical_fill": categorical_fill,
        "ohe_columns": ohe_columns,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "numeric_impute_strategy": numeric_impute_strategy,
        "categorical_impute_strategy": categorical_impute_strategy,
    }


def apply_preprocessor(
    df: pd.DataFrame,
    preprocessor: dict,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Apply a fitted preprocessor to a DataFrame split.

    Parameters
    ----------
    df : pd.DataFrame
        Any split (train, val, or test) that must contain ``target_col``.
    preprocessor : dict
        Fitted preprocessor state returned by :func:`fit_preprocessor`.

    Returns
    -------
    tuple of (X, y, feature_names)
        X : np.ndarray, shape ``(n, p)``, dtype ``float64``
        y : np.ndarray, shape ``(n,)``, dtype ``float64``
        feature_names : list[str] of length ``p`` matching X columns in order
    """
    target_col: str = preprocessor["target_col"]
    numeric_cols: list[str] = preprocessor["numeric_cols"]
    categorical_cols: list[str] = preprocessor["categorical_cols"]
    numeric_fill: dict[str, float] = preprocessor["numeric_fill"]
    categorical_fill: dict[str, str] = preprocessor["categorical_fill"]
    ohe_columns: list[str] = preprocessor["ohe_columns"]
    scaler_mean: np.ndarray = preprocessor["scaler_mean"]
    scaler_scale: np.ndarray = preprocessor["scaler_scale"]

    # --- Target vector ---
    y = df[target_col].values.astype(np.float64)

    # --- Impute features ---
    df_imp = df.drop(columns=[target_col]).copy()
    for col in numeric_cols:
        if col in df_imp.columns:
            df_imp[col] = df_imp[col].fillna(numeric_fill.get(col, 0.0))
    for col in categorical_cols:
        if col in df_imp.columns:
            df_imp[col] = df_imp[col].fillna(categorical_fill.get(col, "MISSING"))

    # --- One-hot encode ---
    df_enc = pd.get_dummies(df_imp, columns=categorical_cols, drop_first=False)
    df_enc.columns = _sanitise_columns(df_enc.columns)

    # --- Align to training column set (fills unseen columns with 0.0) ---
    df_enc = df_enc.reindex(columns=ohe_columns, fill_value=0.0)

    # --- Convert to float64 and standardise ---
    X = df_enc.values.astype(np.float64)
    X = (X - scaler_mean) / scaler_scale

    return X, y, list(ohe_columns)
