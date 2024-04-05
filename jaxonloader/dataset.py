from abc import ABC, abstractmethod
from typing import Union

from jaxtyping import Int
from numpy import ndarray as NDArray


class JaxonDataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(
        self, idx: Int[NDArray, " batch_size"] | slice
    ) -> Union[NDArray, tuple[NDArray, ...], "JaxonDataset"]:
        raise NotImplementedError()


class SingleArrayDataset(JaxonDataset):
    def __init__(self, data: NDArray):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataTargetDataset(JaxonDataset):
    def __init__(self, data: NDArray, target: NDArray):
        self.data = data
        self.target = target
        if len(data) != len(target):
            raise ValueError("data and target must have the same length")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return DataTargetDataset(self.data[idx], self.target[idx])
        else:
            return self.data[idx], self.target[idx]
