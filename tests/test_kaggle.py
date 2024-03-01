from jaxonloader import get_kaggle_dataset
from jaxonloader.dataloader import JaxonDataLoader, make


def test_heart_attack_analysis_dataset():
    datasets = get_kaggle_dataset(
        "rashikrahmanpritom/heart-attack-analysis-prediction-dataset"
    )
    first_dataset, second_dataset = datasets

    train_loader, state = make(
        first_dataset,
        batch_size=4,
        shuffle=False,
        drop_last=True,
    )
    x, state, breaking_cond = train_loader(state)

    assert x.shape == (4, 14)


if __name__ == "__main__":
    test_heart_attack_analysis_dataset()
