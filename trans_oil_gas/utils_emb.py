"""Tools for embeddings' quality experiment."""

from typing import Dict, Tuple, List, Optional, Any

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, TensorDataset

import datetime

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
)

from sklearn.preprocessing import label_binarize

import numpy as np
from tqdm import tqdm

import pytorch_lightning as pl


def softmax(X: np.array, theta: float = 1.0, axis: Optional[int] = None) -> np.array:
    """Compute the softmax of each element along an axis of X.

    :param X: initial numpy array
    :param theta: multiplier prior to exponentiation
    :param axis: axis to compute values along

    :return: array (of softmax) the same size as X, the result will sum to 1 along the specified axis.
    """
    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1:
        p = p.flatten()

    return p


class ClassificationModel(pl.LightningModule):
    """PyTorch Lightning wrapper for convenient training models."""

    def __init__(
        self,
        model: nn.Module,
        train_data: Dataset,
        test_data: Dataset,
        n_classes: int,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
    ) -> None:
        """Initialize PyTorch Wrapper.

        :param model: base model
        :param train_data: dataset with training data
        :param test_data: dataset with test data
        :param n_classes: the number of wells in the data
        :param batch_size: batch size for DataLoader
        :param learning_rate: hyperparameter that controls how much to change the model in response to
            the estimated error each time the model weights are updated
        """
        super(ClassificationModel, self).__init__()
        self.model = model

        self.batch_size = batch_size
        self.loss_function = nn.CrossEntropyLoss()

        self.train_data = train_data
        self.val_data = test_data
        self.n_classes = n_classes

        self.learning_rate = learning_rate

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Get model's output.

        :param inputs: input data for model
        :return: base model's data
        """
        return self.model(inputs)

    @staticmethod
    def calculate_metrics(
        target: torch.Tensor, y_pred: torch.Tensor, n_classes: int
    ) -> Tuple[float, float, float]:
        """Calculate accuracy, ROC_AUC and PR_AUC.

        :param target: data with true labels
        :param y_pred: model's predictions
        :param n_classes: the number of wells in target
        :return: tuple of accuracy, ROC_AUC, PR_AUC
        """
        target = target.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        acc = accuracy_score(target, y_pred.argmax(axis=1))

        # Normalization
        y_pred = softmax(y_pred, axis=1)

        # Roc-Auc score
        roc_auc = roc_auc_score(
            y_true=target.astype(np.int),
            y_score=y_pred,
            labels=np.arange(n_classes).astype(np.int),
            multi_class="ovo",
            average="macro",
        )

        # Use label_binarize to be multi-label like settings
        new_target = label_binarize(target, classes=[i for i in range(n_classes)])

        # Precision - recall score
        pr_auc = average_precision_score(new_target, y_pred)

        return acc, roc_auc, pr_auc

    def _step(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Do step.

        :param batch: input data for model
        :param batch_idx: batch index (need for PL)
        :return: dict of log names and logs
        """
        interval_data, target = batch
        pred = self.forward(interval_data)

        loss = self.loss_function(pred[:, 0], target.long())
        acc = (target == pred.squeeze().argmax()).float().mean()

        logs = {
            "loss": loss,
            "acc": acc,
            "preds": pred[:, 0],
            "targets": target,
        }
        return logs

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Do training step.

        :param batch: input data for model
        :param batch_idx: batch index (need for PL)
        :return: loss on training data
        """
        logs = self._step(batch)
        self.log(
            "train_loss", logs["loss"], on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("train_acc", logs["acc"], on_step=False, on_epoch=True, prog_bar=True)
        return logs

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Do validation step.

        :param batch: input data for model
        :param batch_idx: batch index (need for PL)
        :return: dict with loss, accuracy, target and predictions
        """
        logs = self._step(batch)
        self.log("val_loss", logs["loss"], prog_bar=True)
        self.log("val_acc", logs["acc"], prog_bar=True)
        return logs

    def validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        """Calculate validation accuracy, ROC_AUC, PR_AUC after epoch end.

        :param outputs: dict with loss, accuracy, target and predictions
        """
        preds = torch.cat([x["preds"] for x in outputs])
        target = torch.cat([x["targets"] for x in outputs])

        acc, roc_auc, pr_auc = self.calculate_metrics(
            target.squeeze(), preds.squeeze(), self.n_classes
        )

        for k, v in zip(
            ["mean_accuracy", "mean_roc_auc", "mean_pr_auc"],
            [acc, roc_auc, pr_auc],
        ):
            print("{} = {}".format(k, v))
            self.log(k, v, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Set parameters for optimizer.

        :return: optimizer
        """
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return opt

    def train_dataloader(self) -> DataLoader:
        """Set DataLoader for training data.

        :return: dataLoader for training data
        """
        train_dataloader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=4
        )
        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        """Set DataLoader for validation data.

        :return: dataLoader for validation data
        """
        val_dataloader = DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=4
        )
        return val_dataloader


def calculate_metrics_for_final(
    target: np.ndarray, y_pred: np.ndarray
) -> Tuple[float, float, float, np.ndarray]:
    """Calculate Accuracy, ROC_AUC, PR_AUC and confusion matrix.

    :param target: array with true labels
    :param y_pred: array with predictions
    :return: tuple of Accuracy, ROC AUC, PR_AUC, Precision, Recall
    """
    target = np.array(target)

    acc = accuracy_score(target, y_pred.argmax(axis=1))

    y_pred = softmax(y_pred, axis=1)

    # Roc-Auc score
    roc_auc = roc_auc_score(
        y_true=target.astype(np.int),
        y_score=y_pred,
        labels=np.arange(len(np.unique(target))).astype(np.int),
        multi_class="ovo",
        average="macro",
    )

    # Use label_binarize to be multi-label like settings
    new_target = label_binarize(
        target, classes=[i for i in range(len(np.unique(target)))]
    )

    pr_auc = average_precision_score(new_target, y_pred)

    precision = precision_score(
        y_true=target.astype(np.int), y_pred=y_pred.argmax(axis=1), average="macro"
    )

    recall = recall_score(
        target.astype(np.int), y_pred=y_pred.argmax(axis=1), average="macro"
    )

    return acc, roc_auc, pr_auc, precision, recall


def train_test_linear_classifier(
    train: List[List[float]],
    test: List[List[float]],
    train_wells: List[int],
    test_wells: List[int],
    model: nn.Module,
    experiment_name: str,
    batch_size: int = 64,
    epochs: int = 100,
    gpu: List[int] = [0],
    log_dir: str = "../logs",
) -> Tuple[
    nn.Module,
    Optional[pl.loggers.TensorBoardLogger],
    Tuple[float, float, float, float, float],
]:
    """Train and test a model to classify embeddings on wells.

    :param train: train well-intervals
    :param test: test well-intervals
    :param train_wells: train wells
    :param test_wells: test wells
    :param model: model we want to train
    :param experiment_name: name of the experiment for logging
    :param batch_size: batch size
    :param epochs: the number of epochs we want to train our classifier
    :param gpu: indicate used gpu index
    :param log_dir: directory to store all logs
    :return: tuple of trained classifier, logger, tuple of Accuracy, ROC AUC, PR AUC, Precision, Recall
    """
    train_dataset = TensorDataset(train, torch.tensor(train_wells))
    test_dataset = TensorDataset(test, torch.tensor(test_wells))

    clf = ClassificationModel(
        model=model,
        train_data=train_dataset,
        test_data=test_dataset,
        n_classes=len(np.unique(train_wells)),
        batch_size=batch_size,
    )

    current_time = datetime.datetime.now().strftime("%m%d%Y_%H:%M:%S")
    experiment_name = experiment_name + "_" + current_time

    logger = pl.loggers.TensorBoardLogger(save_dir="./logs/", name=experiment_name)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"{log_dir}/{experiment_name}",
        filename="{epoch:02d}-{val_loss:.3f}",
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=gpu,  # set 1 to use one GPU
        benchmark=True,
        check_val_every_n_epoch=1,
        logger=logger,
        callbacks=[
            checkpoint_callback,
            pl.callbacks.early_stopping.EarlyStopping(
                monitor="val_loss", patience=40, mode="min"
            ),
        ],
    )
    trainer.fit(clf)

    y_pred = model(test)
    output = (
        y_pred.detach()
        .numpy()
        .reshape(y_pred.detach().numpy().shape[0], y_pred.detach().numpy().shape[-1])
    )

    acc, roc_auc, pr_auc, precision, recall = calculate_metrics_for_final(
        target=torch.tensor(test_wells), y_pred=torch.tensor(output)
    )
    print(
        "Accuracy: {}, ROC AUC: {}, PR AUC: {}, Precision: {}, Recall: {}".format(
            np.round(acc, 4),
            np.round(roc_auc, 4),
            np.round(pr_auc, 4),
            np.round(precision, 4),
            np.round(recall, 4),
        )
    )

    return model, logger, (acc, roc_auc, pr_auc, precision, recall)


def get_embeddings(
    model: nn.Module,
    model_type: str,
    slice_dataset: Dataset,
    **data_kwargs: Dict[str, Any],
) -> Tuple[List[List[float]], List[int]]:
    """Obtain model's embeddings.

    :param model: model from which embedding would be obtained
    :param model_type: type of model
    :param slice_dataset: dataset of well-intervals
    :param data_kwargs: arguments for data processing
    :return: tuple of embeddings and wells they belong to
    """
    embeddings = torch.tensor([])
    wells = []

    for i, (anchor, positive, negative, anchor_well) in tqdm(enumerate(slice_dataset)):
        if "xgb" in model_type:
            agg_slice = []
            for k in range(anchor.shape[1]):
                agg_slice.extend([anchor[:, k].mean(), anchor[:, k].std()])

            embeddings = torch.cat(
                [
                    embeddings,
                    torch.tensor(
                        [
                            [
                                model.predict_proba(
                                    np.array(agg_slice + agg_slice).reshape(
                                        (1, 2 * len(agg_slice))
                                    )
                                )[:, 1]
                            ]
                        ]
                    ),
                ],
                dim=0,
            )
        else:
            slice_1 = torch.tensor(
                anchor.reshape((1, data_kwargs["slice_len"], data_kwargs["input_size"]))
            )
            if "rnn" in model_type:
                embeddings = torch.cat(
                    [
                        embeddings,
                        model.rnn(slice_1.float())[1][0],
                    ],
                    dim=0,
                )
            elif "transformer_triplet" in model_type:
                embeddings = torch.cat(
                    [
                        embeddings,
                        model.forward(slice_1),
                    ],
                    dim=0,
                )
            else:
                embeddings = torch.cat(
                    [
                        embeddings,
                        model.encode(slice_1),
                    ],
                    dim=0,
                )
        wells.append(anchor_well)

    return embeddings, wells
