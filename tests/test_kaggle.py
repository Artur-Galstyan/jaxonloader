import jax
from jaxonloader import get_kaggle_dataset
from jaxonloader.dataloader import DataLoader


def test_heart_attack_analysis_dataset():
    key = jax.random.PRNGKey(0)

    datasets = get_kaggle_dataset(
        "rashikrahmanpritom/heart-attack-analysis-prediction-dataset"
    )
    first_dataset, second_dataset = datasets

    train_loader = DataLoader(
        first_dataset,
        batch_size=4,
        shuffle=False,
        drop_last=True,
        key=key,
    )
    x = next(iter(train_loader))
    first, *_ = x
    assert first.shape == (4,)
