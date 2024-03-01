import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from jaxonloader.dataset import JaxonDataset


class JaxonDataLoader(eqx.Module):
    dataset: JaxonDataset
    batch_size: int = eqx.field(static=True)
    shuffle: bool = eqx.field(static=True)
    drop_last: bool = eqx.field(static=True)
    indices: Array

    index: eqx.nn.StateIndex

    def __init__(
        self,
        dataset: JaxonDataset,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        *,
        key: PRNGKeyArray | None = None,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.dataset = dataset

        indices = jnp.array(list(range(len(dataset))))
        if self.shuffle and key is None:
            raise ValueError("key must be provided when shuffle is True")
        elif self.shuffle and key is not None:
            key, subkey = jax.random.split(key)
            indices = jax.random.permutation(subkey, indices)
        self.indices = indices
        self.index = eqx.nn.StateIndex(jnp.array(0))

    def __call__(self, state: eqx.nn.State) -> tuple[Array, eqx.nn.State, bool] | None:
        index = state.get(self.index)
        break_condition = (
            self.drop_last and index + self.batch_size > len(self.indices)
        ) or index >= len(self.indices)

        n_samples, n_dims = self.dataset.data.shape
        batch_indices = jax.lax.dynamic_slice_in_dim(
            self.indices, index, self.batch_size
        )
        batch = jax.vmap(lambda i: self.dataset(i))(batch_indices)
        new_state = state.set(self.index, index + self.batch_size)
        return batch, new_state, break_condition

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.indices) // self.batch_size
        else:
            return (len(self.indices) + self.batch_size - 1) // self.batch_size

    def reset(self, key: PRNGKeyArray) -> None:
        if self.shuffle:
            key, subkey = jax.random.split(key)
            self.indices = jax.random.permutation(subkey, self.indices)


def make(
    dataset: JaxonDataset,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    key: PRNGKeyArray | None = None,
) -> tuple[JaxonDataLoader, eqx.nn.State]:
    return eqx.nn.make_with_state(JaxonDataLoader)(
        dataset, batch_size, shuffle, drop_last, key=key
    )
