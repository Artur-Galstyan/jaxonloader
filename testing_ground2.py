import time

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxonloader.dataloader import JaxonDataLoader
from jaxonloader.dataset import JaxonDataset


key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)

data = jnp.zeros(shape=(60000, 784))
jaxon_dataset = JaxonDataset(data)

jaxon_dataloader, state = eqx.nn.make_with_state(JaxonDataLoader)(
    jaxon_dataset, batch_size=64, shuffle=True, key=subkey
)

max_its = len(jaxon_dataloader)
i = 0
jaxon_dataloader = eqx.filter_jit(jaxon_dataloader)
start_time = time.time()
print("Starting to iterate through training data...")
while it := jaxon_dataloader(state):
    x, state = it
    i += 1
    if i >= max_its:
        break
    # print(x)

print(f"Time to iterate through training data: {time.time() - start_time:.2f} seconds")
