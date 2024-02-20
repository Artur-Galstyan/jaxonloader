import jax
from jaxonloader import get_tiny_shakespeare
from jaxonloader.dataloader import DataLoader


def test_mnist():
    batch_size = 32
    block_size = 8
    key = jax.random.PRNGKey(0)

    train, test, vocab_size, encoder, decoder = get_tiny_shakespeare(
        block_size=8,
        train_ratio=0.8,
    )

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        key=key,
    )

    first_batch = next(iter(train_loader))
    assert len(first_batch) == 2
    assert first_batch[0].shape == (batch_size, block_size)
    assert first_batch[1].shape == (batch_size, block_size)
