from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
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
    combine_columns_to_row: bool = False

    def __init__(self, *columns: Array, combine_columns_to_row: bool = False):
        self.columns = columns
        self.combine_columns_to_row = combine_columns_to_row
        if len(columns) == 0:
            raise ValueError("At least one column is required")

    def __len__(self) -> int:
        return len(self.columns[0])

    def __getitem__(self, idx: int) -> tuple[Array, ...] | Array:
        res_tuple: tuple[Array, ...] = ()
        for c in self.columns:
            if idx >= len(c):
                res_tuple += (jnp.array([0]),)
            else:
                res_tuple += (c[idx],)
        if self.combine_columns_to_row:
            res = [t for t in res_tuple]
            return jnp.array(res)
        else:
            return res_tuple
