import os
import pathlib
from functools import wraps
from typing import Any


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
