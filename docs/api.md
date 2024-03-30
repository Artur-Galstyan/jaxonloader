## jaxonloader.dataloader.JaxonDataLoader

Arguments:

- `dataset` (JaxonDataset): The dataset to load data from.
- `batch_size` (int): The size of the batch to load.
- `shuffle` (bool): Whether to shuffle the data.
- `drop_last` (bool): Whether to drop the last batch if it's smaller than `batch_size`.
- `key` (int): The key to use for shuffling. If no shuffling is done, this key is not used.
