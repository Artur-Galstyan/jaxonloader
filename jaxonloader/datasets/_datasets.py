import os
import pathlib
import pickle
from collections.abc import Callable

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from numpy import ndarray as NDArray

from jaxonloader.dataset import DataTargetDataset, JaxonDataset, SingleArrayDataset
from jaxonloader.utils import (
    download_and_extract_zip,
    jaxonloader_cache,
    JAXONLOADER_PATH,
)


@jaxonloader_cache(dataset_name="mnist")
def get_mnist() -> tuple[JaxonDataset, JaxonDataset]:
    data_url = "https://omnisium.eu-central-1.linodeobjects.com/mnist/mnist.zip"
    data_path = pathlib.Path(JAXONLOADER_PATH) / "mnist"
    download_and_extract_zip(data_url, data_path)

    train_df = pl.read_csv(data_path / "mnist_train.csv")
    test_df = pl.read_csv(data_path / "mnist_test.csv")

    x_train = train_df.to_numpy()
    x_test = test_df.to_numpy()

    train_dataset = SingleArrayDataset(x_train)
    test_dataset = SingleArrayDataset(x_test)
    return train_dataset, test_dataset


@jaxonloader_cache(dataset_name="cifar10")
def get_cifar10() -> tuple[JaxonDataset, JaxonDataset]:
    data_url = "https://omnisium.eu-central-1.linodeobjects.com/cifar10/cifar-10-batches-py.zip"
    data_path = pathlib.Path(JAXONLOADER_PATH) / "cifar10"
    download_and_extract_zip(data_url, data_path)
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


@jaxonloader_cache(dataset_name="cifar100")
def get_cifar100() -> tuple[JaxonDataset, JaxonDataset]:
    dataset_url = (
        "https://omnisium.eu-central-1.linodeobjects.com/cifar100/cifar-100-python.zip"
    )
    data_path = pathlib.Path(JAXONLOADER_PATH) / "cifar100"
    download_and_extract_zip(dataset_url, data_path)

    if os.path.exists(data_path) and not os.path.exists(data_path / "cifar-100-python"):
        raise FileNotFoundError(
            f"The data folder {data_path} exists but the dataset is missing. "
            + "If this error persists, please delete the data folder and try again."
        )

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

        def __getitem__(self, idx: NDArray) -> tuple[NDArray, NDArray, NDArray]:
            return self.data[idx], self.fine_labels[idx], self.coarse_labels[idx]

    train_dataset = CiFar100Dataset(
        train_data[b"data"], train_data[b"fine_labels"], train_data[b"coarse_labels"]
    )
    test_dataset = CiFar100Dataset(
        test_data[b"data"], test_data[b"fine_labels"], test_data[b"coarse_labels"]
    )

    return train_dataset, test_dataset


def get_fashion_mnist():
    raise NotImplementedError("get_fashion_mnist is not implemented yet.")


@jaxonloader_cache(dataset_name="tinyshakespeare")
def get_tiny_shakespeare(
    block_size: int = 8, train_ratio: float = 0.8
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

    def get_text():
        data_path = pathlib.Path(JAXONLOADER_PATH) / "tinyshakespeare/"
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


@jaxonloader_cache(dataset_name="titanic")
def get_titanic() -> JaxonDataset:
    data_url = "https://omnisium.eu-central-1.linodeobjects.com/titanic/titanic.zip"
    data_path = pathlib.Path(JAXONLOADER_PATH) / "titanic"
    download_and_extract_zip(data_url, data_path)
    train_df = pl.read_csv(data_path / "train.csv")

    def _gender_to_int(df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            pl.col("Sex")
            .apply(lambda gender: 0 if gender == "male" else 1)
            .alias("Sex")
        )
        return df

    train = _gender_to_int(train_df)
    train_data = train.select(pl.exclude("Survived")).to_numpy()
    train_target = train.select(pl.col("Survived")).to_numpy()

    train_dataset = DataTargetDataset(train_data, train_target)

    return train_dataset


def from_dataframe(dataframe: pl.DataFrame | pd.DataFrame) -> JaxonDataset:
    """
    Convert a polars.DataFrame (or pandas.DataFrame) to a JaxonDataset.

    Args:
    dataframe: A polars.DataFrame (or pandas.DataFrame).

    Returns:
    A JaxonDataset.
    """
    df: pl.DataFrame = (
        pl.from_pandas(dataframe) if isinstance(dataframe, pd.DataFrame) else dataframe
    )
    data = df.to_numpy()
    return SingleArrayDataset(data)


def from_dataframes(*dataframes: pl.DataFrame | pd.DataFrame) -> list[JaxonDataset]:
    """
    Convert a list of polars.DataFrame (or pandas.DataFrame) to a list of JaxonDataset.

    Args:
        dataframes: A list of polars.DataFrame (or pandas.DataFrame).

    Returns:
        A list of JaxonDataset.
    """
    datasets: list[JaxonDataset] = []
    for df in dataframes:
        dataframe: pl.DataFrame = (
            pl.from_pandas(df) if isinstance(df, pd.DataFrame) else df
        )
        data = dataframe.to_numpy()
        datasets.append(SingleArrayDataset(data))
    return datasets
