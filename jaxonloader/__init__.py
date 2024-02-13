from abc import ABC, abstractmethod

from jaxonloader.dataloader import DataLoader  # noqa


class Dataset(ABC):
    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError
