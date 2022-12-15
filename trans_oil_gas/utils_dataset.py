"""Core functions for data processing."""
import logging
import os
from datetime import datetime
from typing import List, Tuple, Union, Iterable, Optional, Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


# --------------Generate intervals---------------


def generate_random_samples(
    random_seed: int,
    df: pd.DataFrame,
    slice_len: int = 100,
    results_len: int = 10000,
) -> List[pd.DataFrame]:
    """Generate random intervals with the specified length.

    :param random_seed: fixed random seed for reproducibility
    :param df: source dataframe
    :param slice_len: length of one interval
    :param results_len: total number of intervals
    :return: list of intervals
    """
    np.random.seed(random_seed)

    random_inds = np.round(
        np.random.uniform(0, len(df) - slice_len, results_len)
    ).astype(int)
    random_sample = [df.iloc[i : i + slice_len] for i in random_inds]
    return random_sample


# --------------Prepare data for modeling---------------


def generate_datasets(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    slice_len: int = 100,
    well_column: str = "WELLNAME",
    path_to_saves: str = "./saves/",
    results_len_train: int = 25000,
    results_len_test: int = 5000,
) -> Union[Tuple[Dataset, Dataset], Tuple[List[pd.DataFrame], pd.DataFrame]]:
    """Prepare datasets for any of our models.

    :param train_data: dataframe with training data
    :param test_data: dataframe with test data
    :param slice_len: length of one interval for slicing or aggregation
    :param well_column: name of column with wells' id
    :param path_to_saves: path for saving Scaler
    :param results_len_train: length of generated train slices
    :param results_len_test: length of generated test slices
    :return: tuple of train and test (val) data suitable for the model
    """
    features_for_scale = train_data.columns.copy().tolist()

    # do not scale well_column
    if well_column in features_for_scale:
        features_for_scale.pop(features_for_scale.index(well_column))

    # normalize data
    time_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    path_to_scaler = os.path.join(
        path_to_saves, "standard_scaler_{}.bin".format(time_str.replace(":", "_"))
    )
    scaler = StandardScaler()
    train_data_sc = pd.DataFrame(
        scaler.fit_transform(train_data[features_for_scale].values),
        columns=features_for_scale,
    )
    dump(scaler, path_to_scaler, compress=True)

    test_data_sc = pd.DataFrame(
        scaler.transform(test_data[features_for_scale].values),
        columns=features_for_scale,
    )

    # update features that form dataset
    features_for_dataset = train_data.columns.copy().tolist()
    if well_column not in features_for_dataset:
        features_for_dataset.append(well_column)

    # restore values for these features
    for col in list(set(features_for_dataset) - set(features_for_scale)):
        train_data_sc[col] = train_data[col].values
        test_data_sc[col] = test_data[col].values

    # generate stratified slices and prepare them to the model
    train_slice_dataset = TripletDatasetSlices(
        train_data_sc[features_for_dataset],
        slice_len=slice_len,
        results_len=results_len_train,
        well_column=well_column,
    )
    val_slice_dataset = TripletDatasetSlices(
        test_data_sc[features_for_dataset],
        slice_len=slice_len,
        results_len=results_len_test,
        well_column=well_column,
    )

    return train_slice_dataset, val_slice_dataset


# --------------Generate data for models that is training using triplet loss---------------


def generate_triplet_data_sample(
    tmp_anchors: pd.DataFrame,
    tmp_positives: pd.DataFrame,
    data: pd.DataFrame,
    target: int,
    random_seed: int = 123,
    slice_len: int = 100,
    results_len: int = 10000,
    well_column: str = "WELLNAME",
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
    """Generate samples for triplet models.

    :param tmp_anchors: source dataframe of anchors
    :param tmp_positives: source dataframe of elements with the same target as anchors
    :param data: source dataframe
    :param target: target value of anchors
    :param random_seed: fixed random seed for reproducibility
    :param slice_len: length of one interval
    :param results_len: total number of generated points
    :param well_column: name of column with wells' id
    :return: tuple of generated data
        - list of anchor elements
        - list of positive elements (slices with the same wells' id as anchor elements)
        - list of negative elements (slices with different wells' id from anchor elements)
    """
    anchor_sample = generate_random_samples(
        random_seed, tmp_anchors, slice_len, results_len
    )

    random_well = np.random.choice(tmp_positives[well_column].unique())
    positive_sample = generate_random_samples(
        random_seed + 1,
        tmp_positives[tmp_positives[well_column] == random_well],
        slice_len,
        results_len,
    )

    negative_sample = []
    neg_targets = [w for w in data[well_column].unique() if w != target]

    len_one_neg = results_len // len(neg_targets)
    len_last_neg = results_len - len_one_neg * len(neg_targets)

    for j, neg_target in enumerate(neg_targets):
        if j == len(neg_targets) - 1:
            len_one_negative = len_one_neg + len_last_neg
        else:
            len_one_negative = len_one_neg

        tmp_neg_target = data[data[well_column] == neg_target]

        negative_sample_1 = generate_random_samples(
            random_seed,
            tmp_neg_target,
            slice_len,
            len_one_negative,
        )

        negative_sample.extend(negative_sample_1)

    return anchor_sample, positive_sample, negative_sample


def generate_triplet_slices(
    data: pd.DataFrame,
    random_seed: int,
    slice_len: int = 100,
    results_len: int = 10000,
    well_column: str = "WELLNAME",
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
    """Generate data for triplet models.

    :param data: source dataframe
    :param random_seed: fixed random seed for reproducibility
    :param slice_len: length of one interval
    :param results_len: total number of generated points
    :param well_column: name of column with wells' id
    :return: tuple of generated data
        - list of anchor elements
        - list of positive elements (slices with the same wells' id as anchor elements)
        - list of negative elements (slices with different wells' id from anchor elements)
    """
    anchors = []
    positives = []
    negatives = []

    len_one_target = results_len // data[well_column].nunique()
    len_last_target = results_len - len_one_target * data[well_column].nunique()

    for i, target in enumerate(data[well_column].unique()):
        if i == data[well_column].nunique() - 1:
            len_one = len_one_target + len_last_target
        else:
            len_one = len_one_target

        tmp_target = data[data[well_column] == target]

        (
            anchor_sample,
            positive_sample,
            negative_sample,
        ) = generate_triplet_data_sample(
            tmp_target,
            tmp_target,
            data,
            target,
            random_seed=random_seed,
            slice_len=slice_len,
            results_len=len_one,
            well_column=well_column,
        )

        anchors.extend(anchor_sample)
        positives.extend(positive_sample)
        negatives.extend(negative_sample)

    return anchors, positives, negatives


class TripletDatasetSlices(Dataset):
    """Class implements Dataset for Triplet RNN models."""

    def __init__(
        self,
        wells_data: pd.DataFrame,
        slice_len: int,
        random_seed: int = 123,
        results_len: int = 10000,
        well_column: str = "WELLNAME",
    ) -> None:
        """Initialize Dataset for RNN models.

        :param wells_data: source dataframe
        :param slice_len: length of one interval
        :param random_seed: fixed random seed for reproducibility
        :param results_len: total number of generated points
        :param well_column: name of column with wells' id
        """
        self.slice_len = slice_len
        self.well_column = well_column

        self.anchors, self.positives, self.negatives = generate_triplet_slices(
            wells_data,
            random_seed,
            slice_len,
            results_len,
            well_column=well_column,
        )

    def __len__(self) -> int:
        """Get length of dataset.

        :return: length of dataset
        """
        return len(self.anchors)

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray, int, List[Tuple[Any, Any, Any]]],
        Tuple[np.ndarray, int, int, float, float, List[Any]],
    ]:
        """Get example from Dataset.

        :param idx: index of sample from Dataset
        :return: anchor interval, interval from the same well, interval from another well, anchor element well name
        """
        anchor = self.anchors[idx]
        well_anchor = anchor[self.well_column].iloc[0]

        positive = self.positives[idx]
        negative = self.negatives[idx]

        return (
            anchor.drop(self.well_column, axis=1).values.astype(np.float32),
            positive.drop(self.well_column, axis=1).values.astype(np.float32),
            negative.drop(self.well_column, axis=1).values.astype(np.float32),
            anchor[self.well_column].iloc[0],
        )
