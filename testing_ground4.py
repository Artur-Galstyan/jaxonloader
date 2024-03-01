import time

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxonloader.dataloader import JaxonDataLoader
from jaxonloader.dataset import JaxonDataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
train_loader = DataLoader(dataset1, batch_size=64, shuffle=True)
print("Starting to iterate through training data...")
start_time = time.time()

for data in train_loader:
    pass

print(f"Time to iterate through training data: {time.time() - start_time:.2f} seconds")


key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)

data = jnp.zeros(shape=(60000, 784))
jaxon_dataset = JaxonDataset(data)

jaxon_dataloader, state = eqx.nn.make_with_state(JaxonDataLoader)(
    jaxon_dataset, batch_size=64, shuffle=True, key=subkey
)

jaxon_dataloader = eqx.filter_jit(jaxon_dataloader)
start_time = time.time()
print("Starting to iterate through training data...")
while it := jaxon_dataloader(state):
    x, state, breaking_cond = it
    if breaking_cond:
        break
    # print(x)

print(f"Time to iterate through training data: {time.time() - start_time:.2f} seconds")
