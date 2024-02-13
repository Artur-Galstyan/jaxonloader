from abc import ABC, abstractmethod
from typing import Literal

from jaxonloader._datasets import _get_dataset
from jaxonloader.dataloader import DataLoader  # noqa


class Dataset(ABC):
    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError


def get_dataset(
    dataset: Literal[
        "tinyshakespeare", "imagenet", "mnist", "cifar10", "fashion_mnist"
    ],
) -> Dataset:
    return _get_dataset(dataset)
