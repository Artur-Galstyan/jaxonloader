import jax  # noqa
from jaxonloader import get_tiny_shakespeare, get_kaggle_dataset  # noqa
from jaxonloader.dataloader import DataLoader  # noqa
from jaxonloader import StandardDataset  # noqa
import jax.numpy as jnp  # noqa


datasets = get_kaggle_dataset(
    "rashikrahmanpritom/heart-attack-analysis-prediction-dataset",
    combine_columns_to_row=True,
)
first_dataset, second_dataset = datasets

train_loader = DataLoader(
    first_dataset,
    batch_size=4,
    shuffle=False,
    drop_last=True,
)
x = next(iter(train_loader))

assert not isinstance(x, tuple)
assert x.shape == (4, 14)
