from typing import Optional

from jaxonloader.utils import (
    jaxonloader_cache,
)


@jaxonloader_cache(dataset_name="mnist")
def download_mnist(*, target_path: Optional[str]) -> None:
    return


@jaxonloader_cache(dataset_name="titanic")
def download_titanic(*, target_path: Optional[str]) -> None:
    pass


@jaxonloader_cache(dataset_name="hms")
def download_hms(*, target_path: Optional[str]) -> None:
    pass


@jaxonloader_cache(dataset_name="cifar10")
def download_cifar10(*, target_path: Optional[str]) -> None:
    pass


@jaxonloader_cache(dataset_name="cifar100")
def download_cifar100(*, target_path: Optional[str]) -> None:
    pass


@jaxonloader_cache(dataset_name="tinyshakespeare")
def download_tinyshakespeare(*, target_path: Optional[str]) -> None:
    pass
