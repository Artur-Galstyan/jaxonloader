## jaxonloader.dataloader.JaxonDataLoader

**Note, this class is not meant to be used directly. Instead, use the `jaxonloader.dataloader.make`
function to create a `JaxonDataLoader` with an `index`!**

Arguments:

- `dataset` (JaxonDataset): The dataset to load data from.
- `batch_size` (int): The size of the batch to load.
- `shuffle` (bool): Whether to shuffle the data.
- `drop_last` (bool): Whether to drop the last batch if it's smaller than `batch_size`.
- `key` (PRNGKey): The key to use for shuffling. If no shuffling is done, this key is not used.

## jaxonloader.dataloader.make

Creates a `JaxonDataLoader` or a `JITJaxonDataLoader` depending on the `jit` argument as well as an
index state.

Arguments:

- `dataset` (JaxonDataset): The dataset to load data from.
- `batch_size` (int): The size of the batch to load.
- `shuffle` (bool): Whether to shuffle the data.
- `drop_last` (bool): Whether to drop the last batch if it's smaller than `batch_size`.
- `key` (PRNGKeyArray): The key to use for shuffling. If no shuffling is done, this key is not used.
- `jit` (bool): Whether to JIT compile the data loader (recommended).

Returns:

- If `jit` is `True`, returns a tuple of a `JITJaxonDataLoader` and a `jaxonloader.dataloader.Index`.
- If `jit` is `False`, returns a tuple of a `JaxonDataLoader` and a `jaxonloader.dataloader.Index`.

```python
from jaxonloader.dataloader import make, from_dataframe
import pandas as pd

df = pd.read_csv('data.csv')
dataset = from_dataframe(df)
jaxon_dataloader, index = make(dataset, batch_size=32, shuffle=True, drop_last=True, key=jax.random.PRNGKey(0), jit=True)


while it := jaxon_dataloader(index):
    x, index, done = it
    if done:
        break

```
