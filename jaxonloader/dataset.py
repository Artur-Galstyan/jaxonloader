import equinox as eqx
from jaxtyping import Array


class JaxonDataset(eqx.Module):
    data: Array

    def __init__(self, data: Array):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __call__(self, idx: int) -> Array:
        return self.data[idx]

    def __getitem__(self, idx: int) -> Array:
        return self(idx)
