"""Reproducible train / validation / test splitting and one-call dataset builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.load_data import load_csv
from src.data.preprocess import apply_preprocessor, fit_preprocessor


def split_dataframe(
    df: pd.DataFrame,
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train, validation, and test sets.

    The split is performed in two deterministic steps: first the test set is
    carved off, then the remaining data is split into train and validation.
    All indices are reset so downstream code can assume contiguous integer
    indexing.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset including the target column.
    val_size : float
        Fraction of the **total** dataset to reserve for validation.
        Must satisfy ``val_size + test_size < 1``.
    test_size : float
        Fraction of the **total** dataset to reserve for testing.
    seed : int
        Random seed forwarded to :func:`sklearn.model_selection.train_test_split`
        for reproducibility.

    Returns
    -------
    tuple of (df_train, df_val, df_test)
        Each returned object is a ``pd.DataFrame`` with a reset integer index.
    """
    df_trainval, df_test = train_test_split(
        df, test_size=test_size, random_state=seed, shuffle=True
    )
    # Fraction of the trainval portion that becomes validation
    val_fraction_of_trainval = val_size / (1.0 - test_size)
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_fraction_of_trainval,
        random_state=seed,
        shuffle=True,
    )
    return (
        df_train.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_test.reset_index(drop=True),
    )


def build_dataset(
    raw_path: str | Path,
    target_col: str,
    val_size: float,
    test_size: float,
    seed: int,
    numeric_impute_strategy: str = "median",
    categorical_impute_strategy: str = "most_frequent",
    drop_cols: list[str] | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
]:
    """Full one-call pipeline: load CSV → split → fit preprocessor → apply to all splits.

    The preprocessor is **fit exclusively on the training split** and then
    applied to val and test to prevent data leakage.

    Parameters
    ----------
    raw_path : str or Path
        Path to the raw CSV file.
    target_col : str
        Name of the regression target column (e.g. ``"SalePrice"``).
    val_size : float
        Fraction of the total dataset reserved for validation.
    test_size : float
        Fraction of the total dataset reserved for testing.
    seed : int
        Random seed for reproducible splitting.
    numeric_impute_strategy : str
        ``"median"`` (default) or ``"mean"`` for numeric missing-value
        imputation.
    categorical_impute_strategy : str
        ``"most_frequent"`` (default) for categorical missing-value
        imputation.
    drop_cols : list[str] or None
        Columns to drop at load time (e.g. non-feature identifiers like
        ``"Id"``).  Forwarded to :func:`~src.data.load_data.load_csv`.

    Returns
    -------
    tuple of (X_train, X_val, X_test, y_train, y_val, y_test, feature_names)
        X_* : np.ndarray, shape ``(n_split, p)``, dtype ``float64``
        y_* : np.ndarray, shape ``(n_split,)``, dtype ``float64``
        feature_names : list[str] of length ``p`` identifying each column of X
    """
    df = load_csv(raw_path, drop_cols=drop_cols)
    df_train, df_val, df_test = split_dataframe(
        df, val_size=val_size, test_size=test_size, seed=seed
    )
    preprocessor = fit_preprocessor(
        df_train,
        target_col=target_col,
        numeric_impute_strategy=numeric_impute_strategy,
        categorical_impute_strategy=categorical_impute_strategy,
    )
    X_train, y_train, feature_names = apply_preprocessor(df_train, preprocessor)
    X_val, y_val, _ = apply_preprocessor(df_val, preprocessor)
    X_test, y_test, _ = apply_preprocessor(df_test, preprocessor)
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names
