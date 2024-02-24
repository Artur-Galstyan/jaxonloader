import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from jaxonloader import Dataset


class DataLoader:
    dataset: Dataset
    batch_size: int
    shuffle: bool
    drop_last: bool

    _index: int

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

    def __iter__(self):
        return self

    def __next__(self) -> Array | tuple[Array, ...]:
        if self.drop_last and self._index + self.batch_size > len(self.indices):
            raise StopIteration
        elif self._index >= len(self.indices):
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
