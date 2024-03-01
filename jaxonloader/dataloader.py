import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from jaxonloader import Dataset
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

        # Maybe one day, this will be possible with JAX :(
        break_condition = (
            self.drop_last and index + self.batch_size > len(self.indices)
        ) or index >= len(self.indices)
        # if break_condition:
        #     return None

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


class DataLoader:
    dataset: Dataset
    batch_size: int
    shuffle: bool
    drop_last: bool

    _index: int
    key: PRNGKeyArray | None

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        *,
        key: PRNGKeyArray | None = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        if self.shuffle and key is None:
            raise ValueError("key must be provided when shuffle is True")

        self.indices = jnp.array(list(range(len(dataset))))
        self.key = key

        if self.shuffle and self.key is not None:
            self.key, subkey = jax.random.split(self.key)
            self.indices = jax.random.permutation(subkey, self.indices)

        self._index = 0

    def reset(self):
        if self.shuffle and self.key is not None:
            self.key, subkey = jax.random.split(self.key)
            self.indices = jax.random.permutation(subkey, self.indices)
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self) -> Array | tuple[Array, ...]:
        if self.drop_last and self._index + self.batch_size > len(self.indices):
            self.reset()
            raise StopIteration
        elif self._index >= len(self.indices):
            self.reset()
            raise StopIteration

        batch_indices = self.indices[self._index : self._index + self.batch_size]

        if isinstance(self.dataset[0], tuple):
            dataset_return_length = len(self.dataset[0])
            batch = tuple(
                jnp.array([self.dataset[i][j] for i in batch_indices])
                for j in range(dataset_return_length)
            )
        else:
            batch = jnp.array([self.dataset[i] for i in batch_indices])
        self._index += self.batch_size
        return batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.indices) // self.batch_size
        else:
            return (len(self.indices) + self.batch_size - 1) // self.batch_size
