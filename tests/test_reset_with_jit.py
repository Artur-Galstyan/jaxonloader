import jax.numpy as jnp
from jaxonloader import get_mnist
from jaxonloader.dataloader import make


def test_reset_with_jit():
    train, test = get_mnist()

    train_loader, index = make(train, batch_size=64, drop_last=True, jit=True)
    first_sample, _, _ = train_loader(index)

    while it := train_loader(index):
        x, index, done = it
        if done:
            break

    next_first_sample, _, _ = train_loader(index)
    assert jnp.allclose(first_sample, next_first_sample, atol=1e-6)


if __name__ == "__main__":
    test_reset_with_jit()
