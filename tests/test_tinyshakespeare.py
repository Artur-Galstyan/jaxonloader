import jax
from jaxonloader import get_tiny_shakespeare
from jaxonloader.dataloader import make


def test_mnist():
    batch_size = 4
    block_size = 8
    key = jax.random.PRNGKey(0)

    train, test, vocab_size, encoder, decoder = get_tiny_shakespeare(
        block_size=8,
        train_ratio=0.8,
    )

    train_loader, index = make(
        train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        key=key,
    )
    x, index, breaking_cond = train_loader(index)

    assert x.shape == (batch_size, block_size * 2)
