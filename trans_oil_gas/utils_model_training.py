"""Tools for model training and testing."""

from typing import Tuple, Union, Dict

import numpy as np

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

from torch.utils.data import DataLoader, Dataset


def calculate_metrics(
    target: np.ndarray, y_pred: np.ndarray
) -> Tuple[float, float, float, np.ndarray]:
    """Calculate Accuracy, ROC_AUC, PR_AUC and confusion matrix.

    :param target: array with true labels
    :param y_pred: array with predictions
    :return: tuple of accuracy, ROC AUC, PR AUC, Confusion matrix
    """
    y_pred_cl = y_pred > 0.5
    acc = accuracy_score(target, y_pred_cl)
    conf_matrix = confusion_matrix(target, y_pred_cl)
    pr_auc = average_precision_score(target, y_pred)

    try:
        roc_auc = roc_auc_score(target, y_pred)
    except ValueError:
        roc_auc = acc

    return acc, roc_auc, pr_auc, conf_matrix


def test_nn_model(
    model: nn.Module,
    dataset: Dataset,
    return_metrics: bool = True,
) -> Union[
    Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, np.ndarray]],
    Tuple[np.ndarray, np.ndarray],
]:
    """Test Neural Network models (Siamese or Triplet).

    :param model: pre-trained NN model
    :param dataset: test dataset
    :param return_metrics: if True return also Accuracy, ROC_AUC, PR_AUC
    :return: print Accuracy, ROC_AUC, PR_AUC and return tuple of
        - predictions
        - targets
        - metrics (if return_metrics=True)
    """
    test_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    test_predictions = []
    test_targets = []

    # get predictions on each batch
    for anchor, positive, negative, well_anchor in test_loader:
        target_1_pred = model.forward([anchor, positive])
        target_0_pred = model.forward([anchor, negative])

        predictions = torch.cat((target_1_pred.squeeze(), target_0_pred.squeeze()))
        all_targets = torch.cat(
            (torch.ones(anchor.shape[0]).float(), torch.zeros(anchor.shape[0]).float())
        ).to(predictions.device)

        test_predictions.extend(predictions.flatten().detach().cpu().numpy())
        test_targets.extend(all_targets.flatten().detach().cpu().numpy())

    test_predictions = np.array(test_predictions)
    test_targets = np.array(test_targets)

    metrics = calculate_metrics(test_targets, test_predictions)

    print(
        "Accuracy:{:.3}, ROC_AUC:{:.3}, PR_AUC:{:.3}".format(
            metrics[0], metrics[1], metrics[2]
        )
    )

    if return_metrics:
        return test_predictions, test_targets, metrics

    return test_predictions, test_targets


def print_metrics(result: Dict[str, np.array]) -> None:
    """Print mean and std of metrics.

    :param result: dict with lists of metrics
    :return:
    """
    for key in result.keys():
        print(
            "{}: {:.3f} \u00B1 {:.3f}".format(
                key, np.mean(result[key]), np.std(result[key])
            )
        )
