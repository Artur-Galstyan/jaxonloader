import equinox as eqx
from jaxonloader import get_mnist
from jaxonloader.dataloader import make


def test_mnist():
    train, test = get_mnist()

    train_loader, index = make(
        train,
        batch_size=64,
        drop_last=True,
    )
    train_loader = eqx.filter_jit(train_loader)
    x, _, _ = train_loader(index)
    assert x.shape == (64, 785)


if __name__ == "__main__":
    test_mnist()
