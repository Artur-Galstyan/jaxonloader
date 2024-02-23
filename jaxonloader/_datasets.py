import os
import pathlib
import urllib.request
import zipfile
from collections.abc import Callable

import jax.numpy as jnp
import pandas as pd
import polars as pl
from jaxtyping import Array
from loguru import logger

from jaxonloader.dataset import Dataset, StandardDataset
from jaxonloader.utils import jaxonloader_cache, JAXONLOADER_PATH


@jaxonloader_cache(dataset_name="kaggle")
def get_kaggle_dataset(
    dataset_name: str,
    force_redownload: bool = False,
    *,
    kaggle_json_path: str | None = None,
) -> list[Dataset]:
    """
    Get a dataset from Kaggle. You need to have the Kaggle
    API token in your home directory. Furthermore,
    only the CSV files in the dataset will be used as support
    for other file extensions is not implemented yet.

    Args:
        dataset_name: The name of the dataset in Kaggle.
        force_redownload: Whether to force redownload the dataset and
        overwrite the existing one.

    Returns:
        A list of datasets. Each dataset is of class StandardDataset.

    Raises:
        FileNotFoundError: If the dataset is not found in Kaggle.
        ValueError: If the Kaggle API token is not found.
    """
    try:
        dataframes = get_kaggle_dataset_dataframes(
            dataset_name, force_redownload, kaggle_json_path=kaggle_json_path
        )
    except Exception as e:
        logger.error(f"Failed to get the dataset {dataset_name} from Kaggle.")
        raise e

    return from_dataframes(*dataframes)


@jaxonloader_cache(dataset_name="kaggle")
def get_kaggle_dataset_dataframes(
    dataset_name: str,
    force_redownload: bool = False,
    *,
    kaggle_json_path: str | None = None,
) -> list[pl.DataFrame]:
    """
    Get a dataset from Kaggle. You need to have the Kaggle
    API token in your home directory. Furthermore,
    only the CSV files in the dataset will be used as support
    for other file extensions is not implemented yet.

    Args:
        dataset_name: The name of the dataset in Kaggle.
        force_redownload: Whether to force redownload the dataset and
        overwrite the existing one.
        kaggle_json_path: The path to the Kaggle API token. If not provided,
        the default path is used.

    Returns:
        A list of dataframes. Each dataframe is of class polars.DataFrame.

    Raises:
        FileNotFoundError: If the dataset is not found in Kaggle.
        ValueError: If the Kaggle API token is not found.
    """

    kaggle_path = (
        pathlib.Path.home() / ".kaggle/kaggle.json"
        if kaggle_json_path is None
        else kaggle_json_path
    )

    if not os.path.exists(kaggle_path):
        raise ValueError(
            f"Kaggle API token not found at {kaggle_path}. Please download it from"
            + "Kaggle and place it in your home directory, e.g."
            + f"under {pathlib.Path.home() / '.kaggle'}."
        )

    if force_redownload:
        if os.path.exists(JAXONLOADER_PATH / "kaggle" / dataset_name):
            logger.info(f"Removing the existing dataset {dataset_name}")
            os.rmdir(JAXONLOADER_PATH / "kaggle" / dataset_name)

    if not os.path.exists(JAXONLOADER_PATH / "kaggle" / dataset_name):
        logger.info(f"Downloading the dataset {dataset_name} from Kaggle")
        os.makedirs(JAXONLOADER_PATH / "kaggle" / dataset_name)
        logger.info(
            f"Storing the dataset in {JAXONLOADER_PATH / 'kaggle' / dataset_name}"
        )
        try:
            os.system(
                f"kaggle datasets download -d {dataset_name} "
                + f"-p {JAXONLOADER_PATH / 'kaggle' / dataset_name}"
            )
        except Exception as e:
            logger.error(f"Failed to download the dataset {dataset_name} from Kaggle.")
            raise e

    path_to_dataset = JAXONLOADER_PATH / "kaggle" / dataset_name
    assert os.path.exists(
        path_to_dataset
    ), f"Failed to download the dataset {dataset_name} or the dataset does not exist."

    _, trimmed_dataset_name = dataset_name.split("/")
    downloaded_zip_from_kaggle = (
        JAXONLOADER_PATH / "kaggle" / dataset_name / (trimmed_dataset_name + ".zip")
    )
    with zipfile.ZipFile(downloaded_zip_from_kaggle, "r") as zip_ref:
        logger.info(f"Extracting the dataset to {path_to_dataset}")
        zip_ref.extractall(path_to_dataset)

    dataframes: list[pl.DataFrame] = []
    for file in os.listdir(path_to_dataset):
        if file.endswith(".csv"):
            dataframes.append(pl.read_csv(path_to_dataset / file))
    len_files_without_zip = len(os.listdir(path_to_dataset)) - 1
    if len(dataframes) == 0:
        raise FileNotFoundError(
            f"No CSV file found in the dataset {dataset_name} from Kaggle."
        )
    elif len(dataframes) < len_files_without_zip:
        logger.warning(
            f"The folder contained {len(os.listdir(path_to_dataset))} files, but only"
            + f" {len(dataframes)} of them are CSV files. The remaining "
            + f"{len(os.listdir(path_to_dataset)) - len(dataframes)} files have "
            + "other file extensions, which are not supported yet :("
        )

    return dataframes


@jaxonloader_cache(dataset_name="mnist")
def get_mnist() -> tuple[Dataset, Dataset]:
    MNIST_TRAIN_URL = (
        "https://omnisium.eu-central-1.linodeobjects.com/mnist/mnist_train.csv.zip"
    )
    MNIST_TEST_URL = (
        "https://omnisium.eu-central-1.linodeobjects.com/mnist/mnist_test.csv.zip"
    )

    data_path = pathlib.Path(JAXONLOADER_PATH) / "mnist"
    if not os.path.exists(data_path / "mnist_train.csv"):
        logger.info(f"Downloading the dataset from {MNIST_TRAIN_URL}")
        urllib.request.urlretrieve(MNIST_TRAIN_URL, data_path / "mnist_train.csv.zip")
        with zipfile.ZipFile(data_path / "mnist_train.csv.zip", "r") as zip_ref:
            logger.info(f"Extracting the dataset to {data_path}")
            zip_ref.extractall(data_path)

    if not os.path.exists(data_path / "mnist_test.csv"):
        logger.info(f"Downloading the dataset from {MNIST_TEST_URL}")
        urllib.request.urlretrieve(MNIST_TEST_URL, data_path / "mnist_test.csv.zip")
        with zipfile.ZipFile(data_path / "mnist_test.csv.zip", "r") as zip_ref:
            logger.info(f"Extracting the dataset to {data_path}")
            zip_ref.extractall(data_path)

    assert os.path.exists(
        data_path / "mnist_train.csv"
    ), "Failed to download the dataset"
    assert os.path.exists(
        data_path / "mnist_test.csv"
    ), "Failed to download the dataset"

    train_df = pl.read_csv(data_path / "mnist_train.csv")
    test_df = pl.read_csv(data_path / "mnist_test.csv")

    x_train = jnp.array(train_df.drop("label").to_numpy())
    y_train = jnp.array(train_df["label"].to_numpy())

    x_test = jnp.array(test_df.drop("label").to_numpy())
    y_test = jnp.array(test_df["label"].to_numpy())

    train_dataset = StandardDataset(x_train, y_train)
    test_dataset = StandardDataset(x_test, y_test)

    return train_dataset, test_dataset


def get_cifar10():
    raise NotImplementedError("get_cifar10 is not implemented yet.")


def get_cifar100():
    raise NotImplementedError("get_cifar100 is not implemented yet.")


def get_fashion_mnist():
    raise NotImplementedError("get_fashion_mnist is not implemented yet.")


@jaxonloader_cache(dataset_name="tinyshakespeare")
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
        def __init__(self, data: Array, block_size: int = block_size) -> None:
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


def from_dataframes(*dataframes: pl.DataFrame | pd.DataFrame) -> list[Dataset]:
    """
    Convert a list of polars.DataFrame (or pandas.DataFrame) to a list of Dataset.

    Args:
        dataframes: A list of polars.DataFrame (or pandas.DataFrame).

    Returns:
        A list of Dataset.
    """
    datasets: list[Dataset] = []
    for df in dataframes:
        dataframe: pl.DataFrame = (
            pl.from_pandas(df) if isinstance(df, pd.DataFrame) else df
        )
        columns = [jnp.array(df[col].to_numpy()) for col in dataframe.columns]
        datasets.append(StandardDataset(*columns))

    return datasets
