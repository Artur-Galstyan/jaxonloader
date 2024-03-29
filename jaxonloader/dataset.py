import equinox as eqx
import jax
from jaxtyping import Array


cpus = jax.devices("cpu")
gpus = jax.devices("gpu")


class JaxonDataset(eqx.Module):
    data: Array

    def __init__(self, data: Array):
        self.data = jax.device_put(data, cpus[0])

    def __len__(self) -> int:
        return len(self.data)

    def __call__(self, idx: int) -> Array:
        data = jax.device_put(self.data, jax.devices())
        return data[idx]

    def __getitem__(self, idx: int) -> Array:
        return self(idx)
