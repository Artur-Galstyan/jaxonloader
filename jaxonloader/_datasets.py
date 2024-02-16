import os
import pathlib

import jax.numpy as jnp
from beartype.typing import Callable
from jaxtyping import Array
from loguru import logger

from jaxonloader.dataset import Dataset
from jaxonloader.utils import JAXONLOADER_PATH, jaxonloader_cache


def get_mnist():
    raise NotImplementedError("get_mnist is not implemented yet.")


def get_cifar10():
    raise NotImplementedError("get_cifar10 is not implemented yet.")


def get_cifar100():
    raise NotImplementedError("get_cifar100 is not implemented yet.")


def get_fashion_mnist():
    raise NotImplementedError("get_fashion_mnist is not implemented yet.")


@jaxonloader_cache
def get_tiny_shakespeare(
    block_size: int = 8, train_ratio: float = 0.8
) -> tuple[Dataset, Dataset, int, Callable[[str], Array], Callable[[Array], str]]:
    """
    Get the tiny shakespeare dataset from Andrej Karpathy's char-rnn repository.

    Args:
        block_size: The number of tokens in a block in a sequence.
        train_ratio: The ratio of the dataset to be used for training.

    Returns:
        A tuple of (train_dataset, test_dataset, vocab_size, encoder, decoder).
        - train_dataset: The dataset for training.
        - test_dataset: The dataset for testing.
        - vocab_size: The size of the vocabulary.
        - encoder: A function that encodes a string into a sequence of integers.
        - decoder: A function that decodes a sequence of integers into a string.


    Example:
    ```python
    from jaxonloader import get_tiny_shakespeare

    train_dataset, test_dataset, vocab_size, encoder, decoder = get_tiny_shakespeare()
    ```
    """

    class MiniShakesPeare(Dataset):
        def __init__(self, data, block_size=block_size) -> None:
            self.block_size = block_size
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index: int):
            if index == -1:
                index = len(self.data) - 1
            x = self.data[index : index + self.block_size]
            y = self.data[index + 1 : index + self.block_size + 1]

            if index + self.block_size + 1 > len(self.data):
                diff = index + self.block_size + 1 - len(self.data)

                to_add_on_x = diff - 1
                to_add_on_y = diff

                x = jnp.concatenate((x, self.data[:to_add_on_x]))
                y = jnp.concatenate((y, self.data[:to_add_on_y]))

            return x, y

    def get_text():
        data_path = pathlib.Path(JAXONLOADER_PATH) / "tinyshakespeare/"
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        if not os.path.exists(data_path / "input.txt"):
            import urllib.request

            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"  # noqa
            logger.info(f"Downloading the dataset from {url}")
            urllib.request.urlretrieve(url, data_path / "input.txt")

        with open(data_path / "input.txt", "r") as f:
            text = f.read()
        return text

    text = get_text()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    def encode(string: str) -> Array:
        return jnp.array([char_to_idx[ch] for ch in string])

    def decode(latent: Array) -> str:
        return "".join([idx_to_char[idx] for idx in latent])

    encoder = encode
    decoder = decode
    data = jnp.array(encode(text))
    n = int(train_ratio * len(data))

    train_data = data[:n]
    test_data = data[n:]

    train_dataset = MiniShakesPeare(train_data, block_size=block_size)
    test_dataset = MiniShakesPeare(test_data, block_size=block_size)

    return train_dataset, test_dataset, vocab_size, encoder, decoder
