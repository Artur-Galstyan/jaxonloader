from jaxonloader.dataset import JaxonDataset
from jaxonloader.datasets import get_titanic
from numpy import ndarray


def test_slicing():
    data = get_titanic()

    train_test_split = 0.8
    split = int(len(data) * train_test_split)
    train_data = data[:split]
    test_data = data[split:]

    assert isinstance(train_data, JaxonDataset)
    assert isinstance(test_data, JaxonDataset)

    assert isinstance(data[0], tuple)
    assert isinstance(data[0][0], ndarray)


if __name__ == "__main__":
    test_slicing()
