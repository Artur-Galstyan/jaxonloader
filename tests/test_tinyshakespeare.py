from jaxonloader import get_tiny_shakespeare
from jaxonloader.dataloader import JaxonDataLoader


def test_mnist():
    batch_size = 4
    block_size = 8

    train, test, vocab_size, encoder, decoder = get_tiny_shakespeare(
        block_size=8,
        train_ratio=0.8,
    )

    train_loader = JaxonDataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    x = next(train_loader)

    assert x.shape == (batch_size, block_size * 2)
