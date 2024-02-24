import jax.numpy as jnp
from jaxonloader import StandardDataset


def test_none():
    first_c = jnp.array([1, 2, 3, 4, 5])
    second_c = jnp.array([1, 2, 3, 4])
    dataset = StandardDataset(first_c, second_c)

    last = dataset[4]
    x, y = last
    assert y == 0
