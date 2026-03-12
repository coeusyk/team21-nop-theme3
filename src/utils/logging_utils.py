"""Logging configuration helpers for experiment scripts."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger that writes to stdout with a consistent format.

    Parameters
    ----------
    name : str
        Logger name, typically ``__name__`` of the calling module.
    level : int
        Logging level (default: ``logging.INFO``).

    Returns
    -------
    logging.Logger
        Configured logger instance.  Repeated calls with the same *name*
        return the same logger without adding duplicate handlers.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
