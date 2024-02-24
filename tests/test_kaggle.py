from jaxonloader import get_kaggle_dataset
from jaxonloader.dataloader import DataLoader


def test_heart_attack_analysis_dataset():
    datasets = get_kaggle_dataset(
        "rashikrahmanpritom/heart-attack-analysis-prediction-dataset"
    )
    first_dataset, second_dataset = datasets

    train_loader = DataLoader(
        first_dataset,
        batch_size=4,
        shuffle=False,
        drop_last=True,
    )
    x = next(iter(train_loader))
    first, *_ = x
    assert first.shape == (4,)


def test_heart_attack_analysis_dataset_combine():
    datasets = get_kaggle_dataset(
        "rashikrahmanpritom/heart-attack-analysis-prediction-dataset",
        combine_columns_to_row=True,
    )
    first_dataset, second_dataset = datasets

    train_loader = DataLoader(
        first_dataset,
        batch_size=4,
        shuffle=False,
        drop_last=True,
    )
    x = next(iter(train_loader))
    assert not isinstance(x, tuple)
    assert x.shape == (4, 14)
