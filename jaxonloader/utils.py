import os
import pathlib
import shutil
import urllib.request
import zipfile
from functools import wraps
from typing import Any

import progressbar
from loguru import logger
from numpy.random import default_rng

from jaxonloader.config import JAXONLOADER_PATH


pbar = None


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


def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None


def download(url: str, data_path: pathlib.Path) -> None:
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    file_name = url.split("/")[-1]
    logger.info(f"Downloading from {url}")
    urllib.request.urlretrieve(url, data_path / file_name, show_progress)
    if os.path.exists(data_path / "__MACOSX"):
        shutil.rmtree(data_path / "__MACOSX")


def download_and_extract_zip(url: str, data_path: pathlib.Path) -> None:
    if os.path.exists(data_path / ".DS_Store"):
        os.remove(data_path / ".DS_Store")
    if not os.path.exists(data_path) or len(os.listdir(data_path)) == 0:
        logger.info(f"Downloading the dataset from {url}")
        urllib.request.urlretrieve(url, data_path / "temp.zip", show_progress)
        with zipfile.ZipFile(data_path / "temp.zip", "r") as zip_ref:
            logger.info(f"Extracting the dataset to {data_path}")
            zip_ref.extractall(data_path)
        os.remove(data_path / "temp.zip")
        if os.path.exists(data_path / "__MACOSX"):
            shutil.rmtree(data_path / "__MACOSX")
    else:
        logger.info(f"Dataset already exists in {data_path}")
