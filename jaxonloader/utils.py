import functools
import os
import pathlib
import shutil
import zipfile
from functools import wraps
from typing import Any, Optional

import progressbar
from numpy.random import default_rng

from jaxonloader.boto_client import BotoClient
from jaxonloader.config import get_expected_files, JAXONLOADER_PATH


pbar = None
downloaded = 0


def _make_data_dir_if_not_exists(
    dataset_name: str, target_path: Optional[pathlib.Path]
):
    data_path = target_path if target_path else JAXONLOADER_PATH
    data_path = data_path / dataset_name
    if not os.path.exists(data_path):
        os.makedirs(data_path)


def jaxonloader_cache(dataset_name: str) -> Any:
    def decorator(func: Any) -> Any:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            target_path = (
                pathlib.Path(str(kwargs.get("target_path"))).resolve()
                if "target_path" in kwargs and kwargs.get("target_path") is not None
                else JAXONLOADER_PATH
            )
            expected_files = get_expected_files(dataset_name)
            if not all(
                os.path.exists(target_path / dataset_name / file)
                for file in expected_files
            ):
                _make_data_dir_if_not_exists(dataset_name, target_path)
                _download_files(dataset_name, target_path)
                _concenate_files(dataset_name, target_path)
                _unzip_files(dataset_name, target_path)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _unzip_files(dataset_name: str, target_path: pathlib.Path) -> None:
    for file in os.listdir(target_path / dataset_name):
        if file.endswith(".zip"):
            with zipfile.ZipFile(target_path / dataset_name / file, "r") as zip_ref:
                zip_ref.extractall(target_path / dataset_name)
            os.remove(target_path / dataset_name / file)

    if os.path.exists(target_path / dataset_name / "__MACOSX"):
        shutil.rmtree(target_path / dataset_name / "__MACOSX")


def _concenate_files(dataset_name: str, target_path: pathlib.Path) -> None:
    parts = [
        os.path.join(target_path / dataset_name, part)
        if part.startswith("part_")
        else None
        for part in os.listdir(target_path / dataset_name)
    ]

    if len(parts) > 0 and all(parts):
        os.system(
            f"cat {target_path}/{dataset_name}/part_* > "
            + f"{target_path}/{dataset_name}/{dataset_name}.zip"
        )
        for part in parts:
            os.remove(target_path / dataset_name / part) if part else None

    if os.path.exists(target_path / dataset_name / "__MACOSX"):
        shutil.rmtree(target_path / dataset_name / "__MACOSX")


def _download_files(dataset_name: str, target_path: pathlib.Path) -> None:
    s3 = BotoClient.get()
    objs: dict = s3.list_objects(Bucket="omnisium")
    for obj in objs["Contents"]:
        obj_key = obj["Key"]
        if obj_key.startswith(f"{dataset_name}/") and obj_key != f"{dataset_name}/":
            file_name = obj_key.split("/")[-1]
            download_path = target_path / dataset_name / file_name
            callback = functools.partial(show_progress, total_size=obj["Size"])
            s3.download_file("omnisium", obj_key, download_path, Callback=callback)

    if os.path.exists(target_path / dataset_name / ".DS_Store"):
        os.remove(target_path / dataset_name / ".DS_Store")


def get_rng(seed: int | None) -> Any:
    return default_rng(seed) if seed is not None else default_rng()


def show_progress(chunk, total_size):
    global pbar, downloaded
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        downloaded = 0
        pbar.start()
    downloaded += chunk
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None


def get_data_path(dataset_name: str, target_path: Optional[str] = None) -> pathlib.Path:
    data_path = (
        pathlib.Path(JAXONLOADER_PATH)
        if target_path is None
        else pathlib.Path(target_path)
    )
    data_path = data_path / dataset_name
    return data_path
