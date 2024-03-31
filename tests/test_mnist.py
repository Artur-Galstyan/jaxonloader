from jaxonloader import get_mnist
from jaxonloader.dataloader import JaxonDataLoader
from numpy.typing import NDArray


def test_mnist():
    train, test = get_mnist()

    train_loader = JaxonDataLoader(
        train,
        batch_size=64,
        drop_last=True,
    )

    x = next(iter(train_loader))
    assert type(x) == NDArray
    assert x.shape == (64, 785)


if __name__ == "__main__":
    test_mnist()
