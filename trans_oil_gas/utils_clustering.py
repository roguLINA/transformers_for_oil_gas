"""Tools for clustering."""

from typing import Union, List, Dict

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold
from sklearn.metrics import adjusted_rand_score
from trans_oil_gas import utils_dataset, utils_emb

import torch

CLUSTERING_MODELS = {
    "kmeans": KMeans,
    "agglom": AgglomerativeClustering,
    "gmm": GaussianMixture,
}


def _do_clustering(
    embeddings: np.ndarray,
    labels: Union[np.ndarray, List[int]],
    n_times: int,
    clustering_model_name: str,
) -> List[float]:
    """Clustering embeddings.

    :param embeddings: the inputs to cluster
    :param labels: ground truth labels to compare to
    :param n_times: the number of times to repeat clustering
    :param clustering_model_name: model name, one of 'kmeans', 'agglom' and 'gmm'
    :return: list of ARI
    """
    results = []
    n_clusters = len(np.unique(labels))
    for i in range(n_times):
        # select clustering model
        model = CLUSTERING_MODELS[clustering_model_name](n_clusters=n_clusters)

        # cluster
        preds = model.fit_predict(embeddings)

        results.append(round(adjusted_rand_score(labels, preds), 3))
    return results


def emb_clustering_wells(
    model: torch.nn.Module,
    df: pd.DataFrame,
    slice_len: int = 100,
    model_type: str = "classical_model",
    n_splits: int = 5,
    results_len_train: int = 25000,
    results_len_test: int = 5000,
    cv_column: str = "WELLNAME",
    well_column: str = "WELLNAME",
    path_to_saves="./saves/",
    **cluster_kwargs: Dict[str, int]
) -> np.ndarray:
    """Generate and cluster well-interval embeddings.

    :param model: model from which we want to obtain embeddings
    :param df: dataframe with initial well data
    :param slice_len: the length of well-interval
    :param model_type: type of model
    :param n_splits: the number of splits in cross-validation
    :param results_len_train: the number of well-intervals in train set
    :param results_len_test: the number of well-intervals in test set
    :param cv_column: target column name in group-k-fold cross-validation
    :param well_column: column name corresponds to well name
    :param path_to_saves: path to saving logs
    :return: list of ARI
    """
    if "triplet" in model_type:
        results_len_test = results_len_test // 2

    metrics = []
    group_kfold = GroupKFold(n_splits=n_splits)

    for train_index, test_index in group_kfold.split(X=df, groups=df[cv_column]):
        train_data = df.iloc[train_index]
        test_data = df.iloc[test_index]

        _, val_slice_dataset = utils_dataset.generate_datasets(
            train_data,
            test_data,
            well_column=well_column,
            slice_len=slice_len,
            path_to_saves=path_to_saves,
            results_len_train=results_len_train,
            results_len_test=results_len_test,
        )

        embeddings, wells = utils_emb.get_embeddings(
            model,
            model_type,
            val_slice_dataset,
            slice_len=slice_len,
            input_size=df.shape[1] - 1,
        )
        embeddings = embeddings.detach().cpu().numpy().reshape(embeddings.shape[0], -1)

        results = _do_clustering(
            labels=wells,
            n_times=cluster_kwargs["n_times"],
            clustering_model_name="agglom",
            embeddings=embeddings,
        )

        ari_mean = np.mean(results).round(3)
        ari_std = np.std(results).round(3)

        print("ARI = {} ± {}".format(ari_mean, ari_std))

        metrics.append(ari_mean)

    return np.array(metrics)
