from jaxonloader import get_mnist, Index, make


def test_index_subclass():
    train, test = get_mnist()
    trainloader, index = make(train, batch_size=32)
    assert isinstance(index, Index)
