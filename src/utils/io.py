"""I/O helpers for saving tables and run logs to the outputs/ directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path_to_project_relative(path_like: str | Path) -> str:
    """Return a project-root-relative path string when possible."""
    path_obj = Path(path_like)
    try:
        resolved = path_obj.resolve()
    except OSError:
        resolved = path_obj

    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path_obj.as_posix()


def _sanitize_log_value(value: Any) -> Any:
    """Recursively convert persisted absolute paths to relative paths."""
    if isinstance(value, Path):
        return _path_to_project_relative(value)
    if isinstance(value, dict):
        return {key: _sanitize_log_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_sanitize_log_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_log_value(item) for item in value)
    if isinstance(value, str) and value.startswith("/"):
        return _path_to_project_relative(value)
    return value


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
        Project-relative path of the written file.
    """
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    df.to_csv(out_path, index=False)
    return Path(_path_to_project_relative(out_path))


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
        Project-relative path of the written file.
    """
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(_sanitize_log_value(data), fh, indent=2, default=str)
    return Path(_path_to_project_relative(out_path))
