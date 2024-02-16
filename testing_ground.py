import jax

from jaxonloader import get_tiny_shakespeare
from jaxonloader.dataloader import DataLoader

key = jax.random.PRNGKey(0)


train_dataset, test_dataset, vocab_size, encoder, decoder = get_tiny_shakespeare()

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, key=key)

x = next(train_dataloader)[0]

x, y = x[:-1], x[1:]

print(x)
print(y)
