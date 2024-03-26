import jax
import jax.numpy as jnp
from jaxonloader import get_mnist
from jaxonloader.dataloader import make


def test_reset():
    train, test = get_mnist()

    train_loader, index = make(train, batch_size=64, drop_last=True, jit=False)
    leaves, treedef = jax.tree.flatten(index)
    first_state_clone = jax.tree.unflatten(treedef, leaves)
    first_sample, _, _ = train_loader(first_state_clone)

    leaves, treedef = jax.tree.flatten(index)
    another_state_clone = jax.tree.unflatten(treedef, leaves)

    last_state_clone = None
    while it := train_loader(index):
        x, index, done = it
        if done:
            leaves, treedef = jax.tree.flatten(index)
            last_state_clone = jax.tree.unflatten(treedef, leaves)
            break

    next_first_sample, _, _ = train_loader(index)
    first = another_state_clone.get(train_loader.index)  # type: ignore
    assert last_state_clone is not None
    last = last_state_clone.get(train_loader.index)  # type: ignore
    assert first == 0
    assert last == 0
    assert jnp.allclose(first_sample, next_first_sample, atol=1e-6)


if __name__ == "__main__":
    test_reset()
