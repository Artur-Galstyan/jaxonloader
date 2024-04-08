import pathlib
import pickle
from collections.abc import Callable
from typing import Optional

import numpy as np
import polars as pl
from numpy import ndarray as NDArray

from jaxonloader.dataset import DataTargetDataset, JaxonDataset, SingleArrayDataset
from jaxonloader.datasets.download import (
    download_cifar10,
    download_cifar100,
    download_mnist,
    download_tinyshakespeare,
    download_titanic,
)
from jaxonloader.utils import (
    get_data_path,
    JAXONLOADER_PATH,
)


def get_mnist(
    *, target_path: Optional[str] = None
) -> tuple[JaxonDataset, JaxonDataset]:
    download_mnist(target_path=target_path)
    data_path = get_data_path("mnist", target_path)
    train_df = pl.read_csv(data_path / "mnist_train.csv")
    test_df = pl.read_csv(data_path / "mnist_test.csv")

    x_train = train_df.to_numpy()
    x_test = test_df.to_numpy()

    train_dataset = SingleArrayDataset(x_train)
    test_dataset = SingleArrayDataset(x_test)
    return train_dataset, test_dataset


def get_cifar10(target_path: Optional[str] = None) -> tuple[JaxonDataset, JaxonDataset]:
    download_cifar10(target_path=target_path)
    data_path = pathlib.Path(JAXONLOADER_PATH) / "cifar10"
    n_batches = 5
    train_data = []
    train_labels = []
    for i in range(1, n_batches + 1):
        with open(data_path / f"cifar-10-batches-py/data_batch_{i}", "rb") as f:
            data = pickle.load(f, encoding="bytes")
            train_data.append(data[b"data"])
            train_labels.append(data[b"labels"])
    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)

    with open(data_path / "cifar-10-batches-py/test_batch", "rb") as f:
        data = pickle.load(f, encoding="bytes")
        test_data = data[b"data"]
        test_labels = data[b"labels"]

    train_dataset = DataTargetDataset(train_data, train_labels)
    test_dataset = DataTargetDataset(test_data, test_labels)

    return train_dataset, test_dataset


def get_cifar100(
    target_path: Optional[str] = None,
) -> tuple[JaxonDataset, JaxonDataset]:
    download_cifar100(target_path=target_path)
    data_path = get_data_path("cifar100", target_path)

    with open(data_path / "cifar-100-python/train", "rb") as f:
        train_data = pickle.load(f, encoding="bytes")
    with open(data_path / "cifar-100-python/test", "rb") as f:
        test_data = pickle.load(f, encoding="bytes")

    class CiFar100Dataset(JaxonDataset):
        def __init__(
            self, data: NDArray, fine_labels: list[int], coarse_labels: list[int]
        ):
            self.data = data
            self.fine_labels = np.array(fine_labels)
            self.coarse_labels = np.array(coarse_labels)

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, idx) -> tuple[NDArray, NDArray, NDArray]:
            return self.data[idx], self.fine_labels[idx], self.coarse_labels[idx]

    train_dataset = CiFar100Dataset(
        train_data[b"data"], train_data[b"fine_labels"], train_data[b"coarse_labels"]
    )
    test_dataset = CiFar100Dataset(
        test_data[b"data"], test_data[b"fine_labels"], test_data[b"coarse_labels"]
    )

    return train_dataset, test_dataset


def get_fashion_mnist(target_path: Optional[str]):
    raise NotImplementedError("get_fashion_mnist is not implemented yet.")


def get_tiny_shakespeare(
    block_size: int = 8, train_ratio: float = 0.8, *, target_path: Optional[str] = None
) -> tuple[
    JaxonDataset, JaxonDataset, int, Callable[[str], NDArray], Callable[[NDArray], str]
]:
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
    download_tinyshakespeare(target_path=target_path)
    data_path = get_data_path("tinyshakespeare", target_path)

    def get_text():
        with open(data_path / "input.txt", "r") as f:
            text = f.read()
        return text

    text = get_text()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    def encode(string: str) -> NDArray:
        return np.array([char_to_idx[ch] for ch in string])

    def decode(latent: NDArray) -> str:
        return "".join([idx_to_char[idx] for idx in latent])

    encoder = encode
    decoder = decode
    data = np.array(encode(text))
    n = int(train_ratio * len(data))

    x_train = data[:n]
    remainder = len(x_train) % block_size
    x_train = x_train[:-remainder].reshape(-1, block_size)
    y_train = np.roll(x_train, -1)
    train_data = np.concatenate((x_train, y_train), axis=1)
    train_dataset = SingleArrayDataset(train_data)

    x_test = data[n:]
    remainder = len(x_test) % block_size
    x_test = x_test[:-remainder].reshape(-1, block_size)
    y_test = np.roll(x_test, -1)
    test_data = np.concatenate((x_test, y_test), axis=1)
    test_dataset = SingleArrayDataset(test_data)

    return train_dataset, test_dataset, vocab_size, encoder, decoder


def get_titanic(target_path: Optional[str] = None) -> JaxonDataset:
    download_titanic(target_path=target_path)
    data_path = pathlib.Path(JAXONLOADER_PATH) / "titanic"
    train_df = pl.read_csv(data_path / "train.csv")

    def _gender_to_int(df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            pl.when(pl.col("Sex") == "male").then(0).otherwise(1).alias("Sex")
        )
        return df

    def _embarked_to_int(df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            pl.when(pl.col("Embarked") == "S")
            .then(0)
            .when(pl.col("Embarked") == "C")
            .then(1)
            .otherwise(2)
            .alias("Embarked")
        )
        return df

    def _fill_nans(df: pl.DataFrame) -> pl.DataFrame:
        df = df.fill_nan(0)
        return df

    train = _gender_to_int(train_df)
    train = _embarked_to_int(train)
    train = _fill_nans(train)
    train_data = train.select(pl.exclude("Survived")).to_numpy()
    train_target = train.select(pl.col("Survived")).to_numpy()

    train_dataset = DataTargetDataset(train_data, train_target)

    return train_dataset
