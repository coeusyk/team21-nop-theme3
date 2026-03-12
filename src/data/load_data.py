"""Raw CSV loading utilities for the preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_csv(path: str | Path, drop_cols: list[str] | None = None) -> pd.DataFrame:
    """Load a raw CSV file and return a DataFrame with sanitised string column names.

    Parameters
    ----------
    path : str or Path
        Absolute or relative path to the CSV file.
    drop_cols : list[str] or None
        Optional list of column names to drop immediately after loading
        (e.g. identifier columns such as ``"Id"``).

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with column names cast to ``str``.  The ``drop_cols``
        columns, if given, are removed before returning.
    """
    df = pd.read_csv(Path(path))
    df.columns = df.columns.astype(str)
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df
