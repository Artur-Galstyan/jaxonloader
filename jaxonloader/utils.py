from typing import Any

from numpy.random import default_rng


def get_rng(seed: int | None) -> Any:
    return default_rng(seed) if seed is not None else default_rng()
