"""Seed utilities for reproducible randomness."""

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Set the random seed for Python's random module and numpy.

    Args:
        seed: A non-negative integer seed value.

    Raises:
        ValueError: If seed is negative.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    random.seed(seed)
    np.random.seed(seed)


def get_rng(seed: int) -> np.random.Generator:
    """Return a numpy.random.Generator seeded with the given value.

    Args:
        seed: A non-negative integer seed value.

    Returns:
        A seeded numpy.random.Generator instance.

    Raises:
        ValueError: If seed is negative.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    return np.random.default_rng(seed)
