import pathlib
import pickle
from typing import Any

import pandas as pd
from numpy import ndarray as NDArray

from jaxonloader.utils import (
    download_and_extract_zip,
    jaxonloader_cache,
    JAXONLOADER_PATH,
)


@jaxonloader_cache(dataset_name="hms")
def get_hms() -> tuple[dict[str, NDArray], dict[str, NDArray], dict[Any, Any]]:
    """
    Gets the raw HMS dataset from this Kaggle competition:
    https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/data?select=test.csv

    Returns:
    - `eegs: dict[str, NDArray]`: A dictionary of EEGs.
    - `spectrograms: dict[str, NDArray]`: A dictionary of spectrograms.
    - `train_df: dict[Any, Any]`: A dictionary of the training dataframe.

    """
    data_url = "https://omnisium.eu-central-1.linodeobjects.com/hms/hms.zip"
    data_path = pathlib.Path(JAXONLOADER_PATH) / "hms"
    download_and_extract_zip(data_url, data_path)

    with open(data_path / "eegs.pkl", "rb") as f:
        eegs = pickle.load(f)
    with open(data_path / "spectrograms.pkl", "rb") as f:
        spectrograms = pickle.load(f)

    with open(data_path / "train.csv", "rb") as f:
        train_df = pd.read_csv(f)
        train_df = train_df.to_dict()

    return eegs, spectrograms, train_df
