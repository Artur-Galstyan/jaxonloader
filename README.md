# Jaxonloader

A blazingly fast ⚡️ dataloader for JAX that no one asked for, but here it is anyway.

## Documentation

Check out the docs [here](https://artur-galstyan.github.io/jaxonloader/)

## Installation

Install this package using pip like so:

```
pip install jaxonloader
```

## Quickstart

This package differs significantly from the PyTorch `DataLoader` class! In JAX,
there is no internal state, which means we have to keep track of it ourselves. Here's
a minimum example to setup MNIST:

```python

import jax
from jaxonloader import get_mnist, make

key = jax.random.PRNGKey(0)

train, test = get_mnist()
# these are JaxonDatasets

train_loader, index = make(
    train,
    batch_size=4,
    shuffle=True,
    drop_last=True,
    key=key,
    jit=True
)
while x:= train_loader(index):
    data, index, done = x
    processed_data = process_data(data)
    if done:
        break

```

## Philosophy

The `jaxonloader` package is designed to be as lightweight as possible. In fact, it's
only a very thin wrapper around JAX arrays! Under the hood, it's using
the [Equinox library](https://github.com/Patrick-Kidger/equinox) to handle the
stateful nature of the dataloader. Since the dataloader object is just a `eqx.Module`, it
can be JITted and can be used in other JAX transformations as well (although, I haven't tested this).

## Label & Target Handling

Due to it's lightweight nature, this package - as of now - doesn't perform any kinds of transformations. This means that you will have to transform your data first and then pass them to the dataloader.
This also goes for post-processing the data.

While in PyTorch, you would do something like this:

```python

for x, y in train_dataloader:
    # do something with x and y

```

In Jaxonloader, we don't split the row of the dataset into `x` and `y` and instead
simply return the whole row. This means that you will have to do the splitting (i.e. data post-processing) yourself.

```python
# MNIST example
while x:= train_loader(index):
    data, index, done = x
    print(data.shape) # (4, 785)
    x, y = data[:, :-1], data[:, -1] # split the data into x and y

    # do something with x and y
```

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

For one, I just like building things and don't really care if it's needed or not. Secondly,
I don't care about other backends (as they are already very well supported) and only want to
focus on JAX and I needed a lightweight, easy-to-handle package, which loads data in JAX.
