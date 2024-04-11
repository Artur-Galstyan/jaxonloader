from pathlib import Path
from typing import Optional

from jaxonloader.utils import (
    get_data_path,
    jaxonloader_cache,
)


@jaxonloader_cache(dataset_name="mnist")
def download_mnist(*, target_path: Optional[str] = None) -> Path:
    return get_data_path("mnist", target_path)


@jaxonloader_cache(dataset_name="titanic")
def download_titanic(*, target_path: Optional[str] = None) -> Path:
    return get_data_path("titanic", target_path)


@jaxonloader_cache(dataset_name="hms")
def download_hms(*, target_path: Optional[str] = None) -> Path:
    return get_data_path("hms", target_path)


@jaxonloader_cache(dataset_name="cifar10")
def download_cifar10(*, target_path: Optional[str] = None) -> Path:
    return get_data_path("cifar10", target_path)


@jaxonloader_cache(dataset_name="cifar100")
def download_cifar100(*, target_path: Optional[str] = None) -> Path:
    return get_data_path("cifar100", target_path)


@jaxonloader_cache(dataset_name="tinyshakespeare")
def download_tinyshakespeare(*, target_path: Optional[str] = None) -> Path:
    return get_data_path("tinyshakespeare", target_path)
