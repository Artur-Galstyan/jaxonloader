def _get_dataset(dataset: str):
    match dataset.lower():
        case "mnist":
            return _get_mnist()
        case "cifar10":
            return _get_cifar10()
        case "cifar100":
            return _get_cifar100()
        case "fashion_mnist":
            return _get_fashion_mnist()
        case "tinyshakespeare":
            return _get_tiny_shakespeare()
        case _:
            raise ValueError(f"Unknown dataset {dataset}.")


def _get_mnist():
    raise NotImplementedError("get_mnist is not implemented yet.")


def _get_cifar10():
    raise NotImplementedError("get_cifar10 is not implemented yet.")


def _get_cifar100():
    raise NotImplementedError("get_cifar100 is not implemented yet.")


def _get_fashion_mnist():
    raise NotImplementedError("get_fashion_mnist is not implemented yet.")


def _get_tiny_shakespeare():
    raise NotImplementedError("get_tiny_shakespeare is not implemented yet.")
