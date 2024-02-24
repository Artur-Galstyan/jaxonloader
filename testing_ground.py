import jax  # noqa
from jaxonloader import get_tiny_shakespeare, get_kaggle_dataset  # noqa
from jaxonloader.dataloader import DataLoader  # noqa
from jaxonloader import StandardDataset  # noqa
import jax.numpy as jnp  # noqa


key = jax.random.PRNGKey(0)

first_c = jnp.array([1, 2, 3, 4, 5])
second_c = jnp.array([1, 2, 3, 4])
dataset = StandardDataset(first_c, second_c)


last = dataset[4]
x, y = last
assert y == 0

dataloader = DataLoader(
    dataset,
    batch_size=5,
    shuffle=False,
    drop_last=True,
    key=key,
)

x = next(iter(dataloader))
print(x)
