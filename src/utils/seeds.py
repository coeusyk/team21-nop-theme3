"""Utilities for setting random seeds to ensure reproducible experiments."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Set seeds for Python built-in random and NumPy to guarantee reproducibility.

    Parameters
    ----------
    seed : int
        Integer seed value.  Must be passed explicitly — never hardcoded at
        call sites.  Obtain from the ``data.seed`` field of your config YAML.

    Returns
    -------
    None
    """
    random.seed(seed)
    np.random.seed(seed)
