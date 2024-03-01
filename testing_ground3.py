import jax
import jax.numpy as jnp


x = jnp.arange(100)
x = x.reshape((10, 10))
indices = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
random_indices = jax.random.permutation(subkey, indices)
print(x[random_indices[0 : 0 + 3]])
