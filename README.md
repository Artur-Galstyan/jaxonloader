# Jaxonloader

A dataloader that no one asked for, but here it is anyway.

## Documentation

Check out the docs [here](https://artur-galstyan.github.io/jaxonloader/)

## Installation

Install this package using pip like so:

```
pip install jaxonloader
```

## Quickstart

```python

from jaxonloader import get_mnist, JaxonDataLoader


train, test = get_mnist()
# these are JaxonDatasets

train_loader = JaxonDataLoader(
    train,
    batch_size=4,
    shuffle=True,
    drop_last=True,
    key=key,
    jit=True
)
for x in train_loader:
    pass
```

## Philosophy

The `jaxonloader` package is designed to be as lightweight as possible. It's effectively the same as the PyTorch dataloader except that it uses Numpy as the backend.

## Roadmap

The goal is to keep this package as lightweight as possible, while also providing as
many datasets as possible. The next steps are to gather as many datasets as possible
and to provide a simple API to load them.

If you have any datasets you want to see in here, just create an issue and I will look into it :)

## Other backends

Other backends are not supported and are not planned to be supported. There is already
a very good dataloader for PyTorch, and with all the support PyTorch has, it's not
needed to litter the world with yet another PyTorch dataloader. The same goes for TensorFlow as well.

If you really need one, which supports all backends, check out

[jax-dataloader](https://github.com/BirkhoffG/jax-dataloader)

## Then why does this package exist?

For one, I just like building things and don't really care if it's needed or not.
