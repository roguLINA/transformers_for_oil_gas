"""Cross-validation tools."""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from sklearn.model_selection import GroupKFold
from scipy.spatial.distance import cosine

from datetime import datetime

import numpy as np

import os
import json

from trans_oil_gas import (
    utils_dataset,
    utils_model,
    utils_model_training,
)  
import optuna

from typing import List, Tuple, Optional, Union, Dict, Any

from trans_oil_gas.utils_dataset import TripletDatasetSlices


def calculate_distance(
    model: nn.Module,
    dataset: Dataset,
    distance_type: str = "euclidean",
) -> Tuple[float, float, float, float, float]:
    """Calculate classification metrics via distance between embeddings.

    :param model: model from which embeddings are obtained
    :param dataset: dataset with prepared well-intervals
    :param distance_type: type of calculated distance (euclidean or cosine)
    :return: tuple of ROC AUC, PR AUC, Accuracy, Precision, Recall
    """
    test_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    test_targets = []
    test_predictions = []
    for element in test_loader:
        anchor, positive, negative, wells = element

        pred_1 = model.encode(anchor).detach().cpu().numpy()
        pred_2 = model.encode(positive).detach().cpu().numpy()
        pred_3 = model.encode(negative).detach().cpu().numpy()

        if distance_type == "euclidean":
            dist_12 = np.linalg.norm(pred_1 - pred_2, ord=2, axis=1)
            pred_12 = 1 / (1 + dist_12)

            dist_13 = np.linalg.norm(pred_1 - pred_3, ord=2, axis=1)
            pred_13 = 1 / (1 + dist_13)
        elif distance_type == "cosine":
            dist_12 = np.array(
                [cosine(pred_1[i], pred_2[i]) for i in range(len(pred_1))]
            )
            pred_12 = 1 - dist_12

            dist_13 = np.array(
                [cosine(pred_1[i], pred_3[i]) for i in range(len(pred_1))]
            )
            pred_13 = 1 - dist_13

        test_predictions.extend(
            [*pred_12.flatten().tolist(), *pred_13.flatten().tolist()]
        )
        test_targets.extend([1 for _ in range(len(pred_12))])
        test_targets.extend([0 for _ in range(len(pred_13))])

    metrics = utils_model_training.calculate_metrics(
        np.array(test_targets), np.array(test_predictions)
    )
    roc_auc, pr_auc, accuracy, conf_matrix = metrics

    tn, fp, fn, tp = conf_matrix.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    metrics = (roc_auc, pr_auc, accuracy, precision, recall)

    return metrics


def prepare_interval_model(
    df: pd.DataFrame,
    train_index: np.ndarray,
    test_index: np.ndarray,
    well_column: str,
    slice_len: int,
    save_dir: str,
    results_len_train: int,
    results_len_test: int,
    model: nn.Module,
    model_type: str,
    batch_size: int,
    log_dir: str,
    epochs: int,
    gpu: Union[str, int, List[int]],
) -> Tuple[
    TripletDatasetSlices,
    TripletDatasetSlices,
    nn.Module,
    pl.loggers.TensorBoardLogger,
    str,
]:
    """Prepare datasets and initialize, train, and test model.

    :param df: initial dataframe
    :param train_index: indices of training elements
    :param test_index: indices of testing elements
    :param well_column: name of column with names of wells
    :param slice_len: the length of well-interval
    :param save_dir: directory for saving scaler
    :param results_len_train: the number of well-intervals for training
    :param results_len_test: the number of well-intervals for testing
    :param model: model for training and testing
    :param model_type: type of model
    :param batch_size: batch size
    :param log_dir: directory for saving logs
    :param epochs: the number of epochs for model training
    :param gpu: if is not 0 or None use GPU (else CPU)
    :return: tuple of
             - training and validation datasets
             - trained model
             - logger (for further results plotting)
             - experiment name (for necessary information saving)
    """
    train_data = df.iloc[train_index]
    test_data = df.iloc[test_index]

    (train_slice_dataset, val_slice_dataset,) = utils_dataset.generate_datasets(
        train_data,
        test_data,
        well_column=well_column,
        slice_len=slice_len,
        path_to_saves=save_dir,
        results_len_train=results_len_train,
        results_len_test=results_len_test,
    )

    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()

    similarity_model = utils_model.IntervalModel(
        model=model,
        model_type=model_type,
        train_data=train_slice_dataset,
        test_data=val_slice_dataset,
        batch_size=batch_size,
    )

    current_time = datetime.now().strftime("%m%d%Y_%H:%M:%S")
    experiment_name = model_type + "_" + current_time.replace(":", "_")

    logger = pl.loggers.TensorBoardLogger(save_dir=log_dir, name=experiment_name)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"{log_dir}/{experiment_name}",
        filename="{epoch:02d}-{val_loss:.3f}",
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=gpu,
        benchmark=True,
        check_val_every_n_epoch=1,
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(similarity_model)

    return (
        train_slice_dataset,
        val_slice_dataset,
        similarity_model,
        logger,
        experiment_name,
    )


def cv_model(
    model: nn.Module,
    df: pd.DataFrame,
    slice_len: int = 100,
    model_type: str = "siamese",
    n_splits: int = 5,
    results_len_train: int = 25000,
    results_len_test: int = 5000,
    cv_column: str = "WELLNAME",
    well_column: str = "WELLNAME",
    epochs: Optional[int] = None,
    batch_size: int = 64,
    gpu: List[int] = [0],
    emb_distance: bool = False,
    save_dir: str = "../saves/",
    log_dir: str = "../logs/",
) -> Union[
    Tuple[
        nn.Module,
        Tuple[Dict[str, float], Dict[str, float], Dict[str, float]],
        Optional[pl.loggers.TensorBoardLogger],
    ],
    Tuple[nn.Module, Dict[str, float], Optional[pl.loggers.TensorBoardLogger]],
]:
    """Train and evaluate model via cross-validation.

    :param model: model we want ot train
    :param df: initial dataframe
    :param slice_len: the length of well-interval
    :param model_type: type of model
    :param n_splits: the number of splits in cross-validation
    :param results_len_train: the number of well-intervals in train data
    :param results_len_test: the number of well-intervals in test data
    :param cv_column: target column name in group-k-fold cross-validation
    :param well_column: column name corresponds to well name
    :param epochs: the number of epochs for model training
    :param batch_size: batch size
    :param gpu: indicate used gpu index
    :param emb_distance: if True calculate metrics with embeddings' distance
    :param save_dir: path to save useful things
    :param log_dir: path to save logs
    :return: tuple of trained model, metrics (or a tuple of metrics with and without usage of distance), and logger
    """
    if "triplet" in model_type:
        results_len_test = results_len_test // 2

    keys = ["Accuracy", "Precision", "Recall", "ROC AUC", "PR AUC"]
    keys_cluster = ["ARI", "AMI", "V-measure"]

    metrics_eucl = {
        "Eucl_" + keys[k]: [] for k in range(len(keys)) if keys[k] not in keys_cluster
    }
    metrics_cos = {
        "Cos_" + keys[k]: [] for k in range(len(keys)) if keys[k] not in keys_cluster
    }

    metrics = {keys[k]: [] for k in range(len(keys))}

    if "siamese" in model_type:
        results_len_train = results_len_train // 2
        results_len_test = results_len_test // 2

    group_kfold = GroupKFold(n_splits=n_splits)
    for train_index, test_index in group_kfold.split(X=df, groups=df[cv_column]):
        (
            train_slice_dataset,
            val_slice_dataset,
            similarity_model,
            logger,
            experiment_name,
        ) = prepare_interval_model(
            df,
            train_index,
            test_index,
            well_column,
            slice_len,
            save_dir,
            results_len_train,
            results_len_test,
            model,
            model_type,
            batch_size,
            log_dir,
            epochs,
            gpu,
        )
        if "performer" in model_type:
            similarity_model.model.encoder.multi_head_attention.device = "cpu"

        if emb_distance:
            metrics_eucl_nn = calculate_distance(
                similarity_model.model,
                val_slice_dataset,
                distance_type="euclidean",
            )

            metrics_cosine_nn = calculate_distance(
                similarity_model.model,
                val_slice_dataset,
                distance_type="cosine",
            )

            (
                roc_auc_eucl,
                pr_auc_eucl,
                accuracy_eucl,
                precision_eucl,
                recall_eucl,
            ) = metrics_eucl_nn
            (
                roc_auc_cos,
                pr_auc_cos,
                accuracy_cos,
                precision_cos,
                recall_cos,
            ) = metrics_cosine_nn

        if "siamese" in model_type:
            metrics_nn = utils_model_training.test_nn_model(
                similarity_model, val_slice_dataset, return_metrics=True
            )

            accuracy, roc_auc, pr_auc, conf_matrix = metrics_nn[2]
            tn, fp, fn, tp = conf_matrix.ravel()
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            print(
                "TN: {}, FP: {}, FN: {}, TP: {}\nprecision = tp / (tp + fp); recall = tp / (tp + fn)".format(
                    tn, fp, fn, tp
                )
            )

        if "triplet" not in model_type:
            metrics["ROC AUC"].append(roc_auc)
            metrics["PR AUC"].append(pr_auc)
            metrics["Accuracy"].append(accuracy)
            metrics["Precision"].append(precision)
            metrics["Recall"].append(recall)

        if emb_distance:
            metrics_eucl["Eucl_ROC AUC"].append(roc_auc_eucl)
            metrics_eucl["Eucl_PR AUC"].append(pr_auc_eucl)
            metrics_eucl["Eucl_Accuracy"].append(accuracy_eucl)
            metrics_eucl["Eucl_Precision"].append(precision_eucl)
            metrics_eucl["Eucl_Recall"].append(recall_eucl)

            metrics_cos["Cos_ROC AUC"].append(roc_auc_cos)
            metrics_cos["Cos_PR AUC"].append(pr_auc_cos)
            metrics_cos["Cos_Accuracy"].append(accuracy_cos)
            metrics_cos["Cos_Precision"].append(precision_cos)
            metrics_cos["Cos_Recall"].append(recall_cos)

        if "performer" in model_type and len(gpu) == 1:
            similarity_model = similarity_model.to("cuda:{}".format(gpu[0]))
            similarity_model.model.encoder.multi_head_attention.device = (
                "cuda:{}".format(gpu[0])
            )

    if emb_distance:
        return model, (metrics, metrics_eucl, metrics_cos), logger

    return model, metrics, logger


def cv_model_hyperopt(
    model: nn.Module,
    df: pd.DataFrame,
    slice_len: int = 100,
    model_type: str = "siamese",
    n_splits: int = 5,
    results_len_train: int = 25000,
    results_len_test: int = 5000,
    cv_column: str = "WELLNAME",
    well_column: str = "WELLNAME",
    epochs: Optional[int] = None,
    batch_size: int = 64,
    gpu: List[int] = [0],
    emb_distance: bool = False,
    save_dir: str = "../saves/",
    log_dir: str = "./logs/",
) -> float:
    """Train and evaluate model via cross-validation. Used for hyperparameter optimization.

    :param model: model we want ot train
    :param df: initial dataframe
    :param slice_len: the length of well-interval
    :param model_type: type of model
    :param n_splits: the number of splits in cross-validation
    :param results_len_train:
    :param results_len_test:
    :param results_len_train: the number of well-intervals in train data
    :param results_len_test: the number of well-intervals in test data
    :param cv_column: target column name in group-k-fold cross-validation
    :param well_column: column name corresponds to well name
    :param epochs: the number of epochs for model training
    :param batch_size: batch size
    :param gpu: indicate used gpu index
    :param emb_distance: if True calculate metrics with embeddings' distance
    :param save_dir: path to save useful things
    :param log_dir: path to save logs
    :return: mean ROC AUC score calculated through all folds
    """
    roc_aucs = []
    if "triplet" in model_type:
        results_len_test = results_len_test // 2

    group_kfold = GroupKFold(n_splits=n_splits)
    for train_index, test_index in group_kfold.split(X=df, groups=df[cv_column]):
        (
            train_slice_dataset,
            val_slice_dataset,
            similarity_model,
            logger,
            experiment_name,
        ) = prepare_interval_model(
            df,
            train_index,
            test_index,
            well_column,
            slice_len,
            save_dir,
            results_len_train,
            results_len_test,
            model,
            model_type,
            batch_size,
            log_dir,
            epochs,
            gpu,
        )
        if "performer" in model_type:
            similarity_model.model.encoder.multi_head_attention.device = "cpu"

        if emb_distance:
            metrics_cosine_nn = calculate_distance(
                similarity_model.model,
                val_slice_dataset,
                distance_type="cosine",
            )

            (
                roc_auc,
                pr_auc_cos,
                accuracy_cos,
                precision_cos,
                recall_cos,
            ) = metrics_cosine_nn

        if "siamese" in model_type:
            metrics_nn = utils_model_training.test_nn_model(
                similarity_model, val_slice_dataset, return_metrics=True
            )
            accuracy, roc_auc, pr_auc, conf_matrix = metrics_nn[2]

        roc_aucs.append(roc_auc)

        if "performer" in model_type and len(gpu) == 1:
            similarity_model = similarity_model.to("cuda:{}".format(gpu[0]))
            similarity_model.model.encoder.multi_head_attention.device = (
                "cuda:{}".format(gpu[0])
            )

    torch.save(
        model.state_dict(),
        f"{save_dir}{model_type}_{experiment_name}_{np.mean(roc_aucs)}.pth",
    )
    return np.mean(roc_aucs)


def set_model_specific_params(
    model_type: str, specific_params: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Specify some of the parameters for siamese and triplet models.

    :param model_type: type of model
    :param specific_params: parameters we want to preprocess
    :return: tuple of preprocessed specific_params and some necessary other parameters
    """
    kwargs = dict()
    if "siamese" in model_type:
        if "transformer" in model_type:
            fc_hidden_size = specific_params["hidden_size"]
            specific_params.pop("hidden_size")
        elif "performer" in model_type:
            fc_hidden_size = 16
        elif "informer" in model_type:
            fc_hidden_size = 64

        kwargs = {
            "fc_hidden_size": fc_hidden_size,
            "fc_output_size": 1,
            "output_transform": "sigmoid",
        }
    elif "triplet" in model_type:
        if "transformer" in model_type and "hidden_size" in specific_params:
            specific_params["embedding_size"] = specific_params["hidden_size"]
            specific_params.pop("hidden_size")
    return specific_params, kwargs


def optuna_hpo_and_best_model_evaluation(
    model_type: str,
    device: torch.device,
    gpu: List[int],
    n_trials: int,
    fixed_params: Dict[str, Any],
    default_params: Dict[str, Any],
    specific_params: Dict[str, Any],
    data_kwargs: Dict[str, Any],
) -> Tuple[nn.Module, Dict[str, float], List[float]]:
    """Select hyperparameters with Optuna and train and evaluate model with the best set of hyperparameters.

    :param model_type: type of model
    :param device: calculation device ('cpu' or 'cuda')
    :param gpu: indicate used gpu index
    :param n_trials: the number of optuna iterations
    :param fixed_params: parameters that would not be changed
    :param default_params: default values of hyperparameters
    :param specific_params: hyperparameter space
    :param data_kwargs: arguments for data processing
    :return: tuple of trained model, metrics, and ROC AUC scores from all Optuna trials
    """
    # adaptation for optuna optimization
    print("model_type:", model_type)

    def objective(trial):
        def get_specific_params(specific_params):
            ans = dict()
            for k, (suggest_type, suggest_param) in specific_params.items():
                if suggest_type == "cat":
                    ans[k] = trial.suggest_categorical(k, suggest_param)
                elif suggest_type == "int":
                    ans[k] = trial.suggest_int(k, *suggest_param)
                elif suggest_type == "float":
                    ans[k] = trial.suggest_float(k, *suggest_param)
            return ans

        trial_specific_params = get_specific_params(specific_params)
        trial_specific_params, model_specific_kwargs = set_model_specific_params(
            model_type, trial_specific_params
        )

        if "siamese" in model_type:
            clf_model = utils_model.SiameseArchitecture(
                encoder_type=model_type.split("_")[1],
                **model_specific_kwargs,
                **fixed_params,
                **trial_specific_params,
            )
        elif "triplet" in model_type:
            clf_model = utils_model.TripletArchitecture(
                encoder_type=model_type.split("_")[1],
                **fixed_params,
                **trial_specific_params,
            ).float()

        clf_model = clf_model.to(device)

        emb_distance_hpo = False if "siamese" in model_type else True

        roc_auc = cv_model_hyperopt(
            clf_model,
            data_kwargs["df"],
            slice_len=data_kwargs["slice_len"],
            well_column=data_kwargs["well_column"],
            results_len_train=data_kwargs["results_len_train"],
            results_len_test=data_kwargs["results_len_test"],
            model_type=model_type,
            n_splits=data_kwargs["n_splits_hpo"],
            epochs=data_kwargs["epochs"],
            emb_distance=emb_distance_hpo,
            gpu=gpu,
            save_dir=data_kwargs["save_dir"],
            log_dir=data_kwargs["log_dir"],
        )

        return roc_auc

    study = optuna.create_study(direction="maximize")
    study.enqueue_trial({**fixed_params, **default_params})
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    with open(
        os.path.join(data_kwargs["save_dir"], "best_params_{}.json".format(model_type)),
        "w",
    ) as f:
        json.dump(best_params, f)
    print("best_params:", best_params)

    roc_auc_all_trials = [trial.value for trial in study.trials]

    best_params, model_specific_kwargs = set_model_specific_params(
        model_type, best_params
    )

    if "siamese" in model_type:
        clf_model = utils_model.SiameseArchitecture(
            encoder_type=model_type.split("_")[1],
            **model_specific_kwargs,
            **fixed_params,
            **best_params,
        )

    elif "triplet" in model_type:
        clf_model = utils_model.TripletArchitecture(
            encoder_type=model_type.split("_")[1], **fixed_params, **best_params
        ).float()

    model, res_model, logger_model = cv_model(
        clf_model,
        data_kwargs["df"],
        slice_len=data_kwargs["slice_len"],
        well_column=data_kwargs["well_column"],
        results_len_train=data_kwargs["results_len_train"],
        results_len_test=data_kwargs["results_len_test"],
        model_type=model_type,
        n_splits=data_kwargs["n_splits_res"],
        epochs=data_kwargs["epochs"],
        emb_distance=True,
        gpu=gpu,
        save_dir=data_kwargs["save_dir"],
        log_dir=data_kwargs["log_dir"],
    )

    torch.save(
        model.state_dict(),
        os.path.join(data_kwargs["save_dir"], "best_{}.pth".format(model_type)),
    )

    for i in range(len(res_model)):
        try:
            utils_model_training.print_metrics(res_model[i])
            print("-" * 50)
        except:
            print("Smth went wrong in utils_model_training.print_metrics()")

    print("\nall_results:", res_model)
    print("*" * 100)

    return model, res_model, roc_auc_all_trials
