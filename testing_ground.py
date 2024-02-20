import jax
from jaxonloader import get_tiny_shakespeare
from jaxonloader.dataloader import DataLoader


key = jax.random.PRNGKey(0)

train, test, vocab_size, encoder, decoder = get_tiny_shakespeare(
    block_size=8,
    train_ratio=0.8,
)

train_loader = DataLoader(
    train,
    batch_size=4,
    shuffle=True,
    drop_last=True,
    key=key,
)

x = next(iter(train_loader))
print(x)
