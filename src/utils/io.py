"""I/O helpers for saving tables and run logs to the outputs/ directory."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def save_table(df: pd.DataFrame, directory: str | Path, filename: str) -> Path:
    """Save a DataFrame as a CSV file, creating the directory if needed.

    Parameters
    ----------
    df : pd.DataFrame
        Table to persist.
    directory : str or Path
        Target directory (e.g. ``outputs/tables``).
    filename : str
        File name including ``.csv`` extension.

    Returns
    -------
    Path
        Resolved path of the written file.
    """
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    df.to_csv(out_path, index=False)
    return out_path.resolve()


def save_log(data: dict, directory: str | Path, filename: str) -> Path:
    """Save a dictionary as a JSON log file, creating the directory if needed.

    Parameters
    ----------
    data : dict
        Log payload — must be JSON-serialisable (use plain Python types).
    directory : str or Path
        Target directory (e.g. ``outputs/logs``).
    filename : str
        File name including ``.json`` extension.

    Returns
    -------
    Path
        Resolved path of the written file.
    """
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    return out_path.resolve()
