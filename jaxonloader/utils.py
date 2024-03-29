import os
import pathlib
from functools import wraps
from typing import Any

from loguru import logger
from numpy.random import default_rng


JAXONLOADER_PATH = pathlib.Path.home() / ".jaxonloader"


def _make_jaxonloader_dir_if_not_exists():
    if not os.path.exists(JAXONLOADER_PATH):
        os.makedirs(JAXONLOADER_PATH)


def _make_data_dir_if_not_exists(dataset_name: str):
    data_path = JAXONLOADER_PATH / dataset_name
    if not os.path.exists(data_path):
        os.makedirs(data_path)


def jaxonloader_cache(dataset_name: str) -> Any:
    def decorator(func: Any) -> Any:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            _make_jaxonloader_dir_if_not_exists()
            _make_data_dir_if_not_exists(dataset_name)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def deprecation_warning(message: str) -> Any:
    def decorator(func: Any) -> Any:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            logger.warning(message)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def get_rng(seed: int | None) -> Any:
    return default_rng(seed) if seed is not None else default_rng()
