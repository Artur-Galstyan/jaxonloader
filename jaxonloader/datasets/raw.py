import os
import pathlib
import pickle
import zipfile
from typing import Any

import pandas as pd
from loguru import logger
from numpy import ndarray as NDArray

from jaxonloader.utils import (
    download,
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
    data_url = "https://omnisium.eu-central-1.linodeobjects.com/hms/"
    data_path = pathlib.Path(JAXONLOADER_PATH) / "hms"
    expected_files = [
        "eegs.pkl",
        "spectrograms.pkl",
        "train.csv",
    ]

    if not all((data_path / file).exists() for file in expected_files):
        parts = [
            "part_aa",
            "part_ab",
            "part_ac",
            "part_ad",
            "part_ae",
            "part_af",
            "part_ag",
        ]
        for part in parts:
            download(data_url + part, data_path)
        os.system(f"cat {data_path}/part_* > {data_path}/hms.zip")
        with zipfile.ZipFile(data_path / "hms.zip", "r") as zip_ref:
            logger.info(f"Extracting the dataset to {data_path}")
            zip_ref.extractall(data_path)
        logger.info("Cleaning up the dataset")
        os.remove(data_path / "hms.zip")
        for part in parts:
            os.remove(data_path / part)

    with open(data_path / "eegs.pkl", "rb") as f:
        eegs = pickle.load(f)
    with open(data_path / "spectrograms.pkl", "rb") as f:
        spectrograms = pickle.load(f)

    with open(data_path / "train.csv", "rb") as f:
        train_df = pd.read_csv(f)
        train_df = train_df.to_dict()

    return eegs, spectrograms, train_df
