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
        self, idx: Int[NDArray, " batch_size"] | slice | int
    ) -> Union[NDArray, tuple[NDArray, ...]]:
        raise NotImplementedError()

    def split(self, ratio: float) -> tuple["JaxonDataset", "JaxonDataset"]:
        raise NotImplementedError()


class SingleArrayDataset(JaxonDataset):
    def __init__(self, data: NDArray):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def split(self, ratio: float) -> tuple["SingleArrayDataset", "SingleArrayDataset"]:
        split = int(len(self.data) * ratio)
        return SingleArrayDataset(self.data[:split]), SingleArrayDataset(
            self.data[split:]
        )


class DataTargetDataset(JaxonDataset):
    def __init__(self, data: NDArray, target: NDArray):
        self.data = data
        self.target = target
        if len(data) != len(target):
            raise ValueError("data and target must have the same length")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

    def split(self, ratio: float) -> tuple["DataTargetDataset", "DataTargetDataset"]:
        split = int(len(self.data) * ratio)
        return DataTargetDataset(
            self.data[:split], self.target[:split]
        ), DataTargetDataset(self.data[split:], self.target[split:])
