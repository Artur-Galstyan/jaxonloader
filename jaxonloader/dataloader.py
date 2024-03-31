import numpy as np
from numpy import typing as npt

from jaxonloader import JaxonDataset
from jaxonloader.utils import get_rng


class JaxonDataLoader:
    def __init__(
        self,
        dataset: JaxonDataset,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        *,
        key: int | None = None,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.dataset = dataset

        if drop_last:
            self.indices = np.array(
                list(range(len(dataset) - len(dataset) % batch_size))
            )
        else:
            self.indices = np.array(list(range(len(dataset))))
        rng = get_rng(key)

        if self.shuffle:
            self.indices = rng.permutation(self.indices)
        self.index = 0

    def __iter__(self) -> "JaxonDataLoader":
        return self

    def __next__(self) -> npt.NDArray | tuple[npt.NDArray, ...]:
        if self.index < len(self.indices):
            batch_indices = self.indices[self.index : self.index + self.batch_size]
            self.index += self.batch_size
            return self.dataset[batch_indices]
        else:
            self.index = 0
            raise StopIteration

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.indices) // self.batch_size
        else:
            return (len(self.indices) + self.batch_size - 1) // self.batch_size
