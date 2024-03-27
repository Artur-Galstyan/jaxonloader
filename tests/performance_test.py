import time

import jax
import jax.numpy as jnp
import torch
from jaxonloader import make
from jaxonloader.dataset import JaxonDataset
from torch.utils.data import DataLoader


n_samples = 1_000_000  # 1 million samples

torch_data = torch.ones(n_samples, 784)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


train_loader = DataLoader(MyDataset(torch_data), batch_size=64, shuffle=True)
print("Starting to iterate through training data...")
start_time = time.time()
for data in train_loader:
    pass
total_time_torch = time.time() - start_time
print(f"Time to iterate through training data: {total_time_torch} seconds")

key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)

data = jnp.zeros(shape=(n_samples, 784))
jaxon_dataset = JaxonDataset(data)

jaxon_dataloader, state = make(
    jaxon_dataset, batch_size=64, shuffle=True, key=subkey, jit=True
)
start_time = time.time()
print("Starting to iterate through training data...")
while it := jaxon_dataloader(state):
    _, state, done = it
    if done:
        break

total_time_jax = time.time() - start_time
print(f"Time to iterate through training data: {total_time_jax} seconds")
print(
    f"Jax is {total_time_torch / total_time_jax:.2f} times "
    + f"{'slower' if total_time_jax > total_time_torch else 'faster'} than PyTorch"
)
