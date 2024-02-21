from abc import ABC, abstractmethod

from jaxtyping import Array


class Dataset(ABC):
    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int):
        raise NotImplementedError


class StandardDataset(Dataset):
    def __init__(self, *columns: Array):
        self.columns = columns
        if not all(len(c) == len(columns[0]) for c in columns):
            raise ValueError("All columns must have the same length")
        if len(columns) == 0:
            raise ValueError("At least one column is required")

    def __len__(self):
        return len(self.columns[0])

    def __getitem__(self, idx: int):
        return tuple(c[idx] for c in self.columns)
