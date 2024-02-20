import jax  # noqa
from jaxonloader import get_tiny_shakespeare, get_kaggle_dataset  # noqa
from jaxonloader.dataloader import DataLoader  # noqa


datasets = get_kaggle_dataset(
    "rashikrahmanpritom/heart-attack-analysis-prediction-dataset"
)
first_dataset, second_dataset = datasets

first_dataloader = DataLoader(first_dataset, batch_size=32)
print(len(next(iter(first_dataloader))))
