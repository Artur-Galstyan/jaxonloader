from abc import ABC, abstractmethod


class Dataset(ABC):
    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int):
        raise NotImplementedError
