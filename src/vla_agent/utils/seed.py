import random
import numpy as np


def set_seed(seed: int) -> None:
    if seed < 0:
        raise ValueError("Seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed)


def get_rng(seed: int) -> np.random.Generator:
    if seed < 0:
        raise ValueError("Seed must be a non-negative integer")
    return np.random.default_rng(seed)
