import pathlib


JAXONLOADER_PATH = pathlib.Path.home() / ".jaxonloader"


def get_expected_files(dataset_name: str) -> list[str]:
    if dataset_name == "hms":
        return _get_hms_files()
    elif dataset_name == "titanic":
        return _get_titanic_files()
    elif dataset_name == "mnist":
        return _get_mnist_files()
    elif dataset_name == "cifar10":
        return _get_cifar10_files()
    elif dataset_name == "cifar100":
        return _get_cifar100_files()
    elif dataset_name == "tinyshakespeare":
        return _get_tiny_shakespeare_files()
    return []


def _get_tiny_shakespeare_files() -> list[str]:
    return ["input.txt"]


def _get_cifar100_files() -> list[str]:
    return ["cifar-100-python/test", "cifar-100-python/train"]


def _get_cifar10_files() -> list[str]:
    return [
        "cifar-10-batches-py/data_batch_1",
        "cifar-10-batches-py/data_batch_2",
        "cifar-10-batches-py/data_batch_3",
        "cifar-10-batches-py/data_batch_4",
        "cifar-10-batches-py/data_batch_5",
        "cifar-10-batches-py/test_batch",
    ]


def _get_mnist_files() -> list[str]:
    return ["mnist_train.csv", "mnist_test.csv"]


def _get_titanic_files() -> list[str]:
    return ["train.csv"]


def _get_hms_files() -> list[str]:
    return [
        "eegs.pkl",
        "spectrograms.pkl",
        "train.csv",
    ]
