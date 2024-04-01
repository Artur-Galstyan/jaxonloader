import numpy as np
from jaxonloader.dataloader import JaxonDataLoader
from jaxonloader.datasets import get_mnist


def test_mnist():
    train, test = get_mnist()

    train_loader = JaxonDataLoader(
        train,
        batch_size=64,
        drop_last=True,
    )

    x = next(iter(train_loader))
    assert isinstance(x, np.ndarray)
    assert x.shape == (64, 785)


if __name__ == "__main__":
    test_mnist()
