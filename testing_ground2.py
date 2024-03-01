from jaxonloader import get_kaggle_dataset, get_kaggle_dataset_dataframes


dataset_name = "rashikrahmanpritom/heart-attack-analysis-prediction-dataset"


dataframes = get_kaggle_dataset_dataframes(
    dataset_name,
)
first, second = dataframes
print(first.shape)
print(second.shape)

datasets = get_kaggle_dataset(
    "rashikrahmanpritom/heart-attack-analysis-prediction-dataset"
)
first_dataset, second_dataset = datasets
print(first_dataset.data.shape)
print(second_dataset.data.shape)
