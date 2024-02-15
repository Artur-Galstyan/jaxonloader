import os
import pathlib
from functools import wraps

JAXONLOADER_PATH = pathlib.Path.home() / ".jaxonloader"


def _make_jaxonloader_dir_if_not_exists():
    if not os.path.exists(JAXONLOADER_PATH):
        os.makedirs(JAXONLOADER_PATH)


def jaxonloader_cache(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        _make_jaxonloader_dir_if_not_exists()
        return func(*args, **kwargs)

    return wrapper
