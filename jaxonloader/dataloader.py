import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

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
        shuffle: bool,
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

    def __next__(self):
        if self._index >= len(self.indices):
            if self.drop_last:
                raise StopIteration
            else:
                raise StopIteration

        batch_indices = self.indices[self._index : self._index + self.batch_size]
        batch = jnp.array([self.dataset[i] for i in batch_indices])
        self._index += self.batch_size
        return batch
