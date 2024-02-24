from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from jaxtyping import Array


@dataclass
class Dataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        raise NotImplementedError


@dataclass
class StandardDataset(Dataset):
    columns: tuple[Array, ...]

    def __init__(self, *columns: Array):
        self.columns = columns
        if len(columns) == 0:
            raise ValueError("At least one column is required")

    def __len__(self) -> int:
        return len(self.columns[0])

    def __getitem__(self, idx: int) -> tuple:
        res_tuple = ()
        for c in self.columns:
            if idx >= len(c):
                res_tuple += (0,)
            else:
                res_tuple += (c[idx],)

        return res_tuple
