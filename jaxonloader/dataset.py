from abc import ABC, abstractmethod

from numpy.typing import NDArray


class JaxonDataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, idx: NDArray) -> NDArray:
        raise NotImplementedError()


class SingleArrayDataset(JaxonDataset):
    def __init__(self, data: NDArray):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: NDArray) -> NDArray:
        return self.data[idx]
