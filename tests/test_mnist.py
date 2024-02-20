import jax
from jaxonloader import get_mnist
from jaxonloader.dataloader import DataLoader


def test_mnist():
    key = jax.random.PRNGKey(0)

    train, test = get_mnist()
    assert len(train) == 60000
    assert len(test) == 10000

    train_loader = DataLoader(
        train,
        batch_size=4,
        shuffle=False,
        drop_last=True,
        key=key,
    )
    x = next(iter(train_loader))
    assert x[0].shape == (4, 784)
    assert x[1].shape == (4,)

    assert x[1][0] == 5
