import jax.numpy as jnp
from jaxonloader import StandardDataset
from jaxonloader.dataloader import DataLoader


def test_drop_last():
    first_c = jnp.array([1, 2, 3, 4, 5])
    second_c = jnp.array([1, 2, 3, 4])
    dataset = StandardDataset(first_c, second_c)

    last = dataset[4]
    x, y = last
    assert y == 0

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        drop_last=False,
    )

    assert len(dataloader) == 3

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        drop_last=True,
    )

    assert len(dataloader) == 2
