# jaxonloader

A dataloader, but for Jax

The idea of this package is to have a DataLoader similar to the PyTorch one. To ensure that you don't have to learn anything new to use this package, the same API is chosen here (PyTorch's API actually a very good).

Unfortunately, this also means that this package does _not_ follow the functional programming paradigm, because neither does the PyTorch DataLoader API. While in that regard this DataLoader is not _functional_ per se, it still allows for reproducability since you provide a random key to shuffle the data (if you want to).

At the moment, this package is not yet a 1:1 mapping from PyTorch's DataLoader, but one day, we will! \**holding up arm and clenching fist\**

## Installation

Install this package using pip like so:

```
pip install jaxonloader
```

## Usage

Pretty much exactly as you would use PyTorch's DataLoader. Create a dataset class by inheriting
from the `jaxonloader` dataset and implement the `__len__` and `__getitem__` functions. Then simply pass that to the DataLoader class as argument.

Examples coming soon.
