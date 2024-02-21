# jaxonloader

A dataloader, but for Jax

The idea of this package is to have a DataLoader similar to the PyTorch one. To ensure that you don't have to learn anything new to use this package, the same API is chosen here (PyTorch's API actually a very good).

Unfortunately, this also means that this package does _not_ follow the functional programming paradigm, because neither does the PyTorch DataLoader API. While in that regard this DataLoader is not _functional_ per se, it still allows for reproducability since you provide a random key to shuffle the data (if you want to).

At the moment, this package is not yet a 1:1 mapping from PyTorch's DataLoader, but one day, it will be! \**holding up arm and clenching fist\**

## Installation

Install this package using pip like so:

```
pip install jaxonloader
```

## Usage

Pretty much exactly as you would use PyTorch's DataLoader. Create a dataset class by inheriting from the `jaxonloader` dataset and implement the `__len__` and `__getitem__` functions. Then simply pass that to the DataLoader class as argument.

On the other hand, you can also use some of the provided datasets, such as the MNIST dataset.

```python

import jax

from jaxonloader import get_mnist
from jaxonloader.dataloader import DataLoader
key = jax.random.PRNGKey(0)

train, test = get_mnist()

train_loader = DataLoader(
    train,
    batch_size=4,
    shuffle=False,
    drop_last=True,
    key=key,
)
x = next(iter(train_loader))
print(x[0].shape) # (4, 784)
print(x[1].shape) # (4,)


```

## Performing Transformations

As of now, transformations are not supported :(

But - since you can get a dataset from a `DataFrame` - you can first 
transform your data and then pass it to the `from_dataframe` function.

It's not ideal, but it works for now.
