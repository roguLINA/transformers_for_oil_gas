"""Script for attention analysis investigation."""

import numpy as np
import torch
import torch.nn as nn

from copy import deepcopy

import warnings

warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score
from typing import Tuple, Union


def get_inputs_for_calc_grad(
    batch: torch.Tensor, delta: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Obtain batches with substracted delta and added delta.

    :param batch: initial batch (well-interval)
    :param delta: hyperparameter that should be substracted or added
    :return: a tuple of changed versions of the initial batch
    """
    batch_plus, batch_minus = [], []
    for i in range(batch.size()[-1]):
        a, b = deepcopy(batch.detach()), deepcopy(batch.detach())
        a[:, :, i] += delta
        batch_plus.append(a)

        b[:, :, i] -= delta
        batch_minus.append(b)

    return torch.stack(batch_plus), torch.stack(batch_minus)


def get_gradients(
    slice_plus: torch.Tensor,
    slice_minus: torch.Tensor,
    model: nn.Module,
    delta: float,
    agg: bool = True,
) -> torch.Tensor:
    """Calculate gradients in each point of well-interval.

    :param slice_plus: batch with added delta
    :param slice_minus: batch with substracted delta
    :param model: model from which embeddings are obtained
    :param delta: hyperparameter that should be substracted or added
    :param agg: if True aggregate the obtained results
    :return: model's gradients
    """
    grads = torch.stack(
        [
            torch.abs(
                model.encoder(slice_plus[f, :, :, :])
                - model.encoder(slice_minus[f, :, :, :])
            )
            / 2
            / delta
            for f in range(slice_plus.shape[0])
        ]
    ).sum(dim=0)
    if agg:
        return grads.sum(dim=-1)
    return grads


def get_attn_score(el: torch.Tensor) -> torch.Tensor:
    """Calculate attention score.

    :param el: attention matrix
    :return: attention score calculated via summarizing elements on the diagonal of attention matrix
    """
    ans = []
    for i in range(el.shape[0]):
        t = torch.diag(el[i, 0, :, :])
        for j in range(1, el.shape[1]):
            t += torch.diag(el[i, j, :, :])
        ans.append(t.detach().cpu().numpy().tolist())
    return torch.tensor(ans, device=el.device)


def get_attention_scores(x: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """Calculate attention scores for each element in well-interval.

    :param x: well-interval
    :param model: model from which embeddings and attention matrix are obtained
    :return: attention scores
    """
    attn = model.get_attention_maps(x)
    attn_sum = torch.tensor([])

    for el in attn:
        attn_sum = torch.cat([attn_sum, get_attn_score(el)[:, :, None]], dim=-1)

    return attn_sum.sum(dim=-1)


def calc_corr(x1: torch.Tensor, x2: torch.Tensor) -> float:
    """Calculate correlation coefficient of two tensors.

    :param x1: the first tensor
    :param x2: the second tensor
    :return: pearson correlation coefficient of two tensors
    """
    return np.corrcoef(
        x1.detach().cpu().numpy().reshape(-1), x2.detach().cpu().numpy().reshape(-1)
    )[0][1]


def largest_smallest_indices(
    ary: np.array, n: int, mode: str = "largest"
) -> Tuple[np.array, np.array]:
    """Return the n largest (or smallest) indices from a numpy array.

    :param ary: initial array
    :param n: the number of the biggest (or smallest) elements we want to find
    :param mode: indicate which element should be found: the biggest or the smallest
    :return: tuple of indices in the following format: [i-ths (rows) indices], [j-ths (columns) indices]
    """
    assert mode in [
        "largest",
        "smallest",
    ], "mode should be either 'largest' or 'smallest'"
    flat = ary.flatten()

    if mode == "largest":
        indices = np.argpartition(flat, -n)[-n:]
    elif mode == "smallest":
        indices = np.argpartition(flat, n)[:n]

    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


def change_interval_part_with_highest_score(
    x: torch.Tensor,
    model: nn.Module,
    delta: float,
    fill_with: str = "zeros",
    p: float = 0.2,
    add_baseline: bool = False,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
]:
    """Replace p % of elements in well-interval with the highest gradients and the lowest attention scores.

    :param x: initial well-interval
    :param model: transforer model
    :param delta: hyperparameter which would be substracted or added to the initial interval
    :param fill_with: elements with which elements from well-interval would be masked
    :param p: the percentage of eliminated values
    :param add_baseline: if True add baseline where random elements from well-interval are masked
    :return: tuple of intervals with masked elements with the highest gradients, the smallest attention scores
             (if add_baseline True also return interval with masked random parts)
    """
    assert fill_with in [
        "zeros",
        "rand",
    ], "Filling type should be either 'zeros' or 'rand'"

    x_plus, x_minus = get_inputs_for_calc_grad(x, delta)
    gr = get_gradients(x_plus, x_minus, model, delta, agg=False).detach().cpu().numpy()

    a = get_attention_scores(x, model)

    new_batch_g, new_batch_a = [], []

    if add_baseline:
        new_batch_r = []

    n_g, n_a = int(p * x.shape[1] * x.shape[2]), int(p * x.shape[1])

    for i in range(x.shape[0]):

        new_x_g, new_x_a, new_gr, new_a = (
            deepcopy(x[i, :, :]),
            deepcopy(x[i, :, :]),
            deepcopy(gr[i, :, :]),
            deepcopy(a[i, :]),
        )

        if add_baseline:
            new_x_r = deepcopy(x[i, :, :])

        i_g, j_g = largest_smallest_indices(new_gr, n=n_g, mode="largest")
        i_a = largest_smallest_indices(new_a, n=n_a, mode="smallest")

        if fill_with == "rand":
            rand_g = torch.randn(n_g)
            rand_a = torch.randn(n_a)

            if add_baseline:
                rand_r = torch.randn(n_g)

        for k, (ii_g, jj_g) in enumerate(zip(i_g, j_g)):
            new_x_g[ii_g, jj_g] = 0 if fill_with == "zeros" else rand_g[k]

        new_batch_g.append(new_x_g[None, :, :])

        for k, ii_a in enumerate(i_a):
            new_x_a[ii_a, :] = 0 if fill_with == "zeros" else rand_a[k]

        new_batch_a.append(new_x_a[None, :, :])

        if add_baseline:
            i_r, j_r = torch.randint(
                low=0, high=x.shape[1], size=(n_g,)
            ), torch.randint(low=0, high=x.shape[2], size=(n_g,))
            for k, (ii_r, jj_r) in enumerate(zip(i_r, j_r)):
                new_x_r[ii_r, jj_r] = 0 if fill_with == "zeros" else rand_r[k]

            new_batch_r.append(new_x_r[None, :, :])

    if add_baseline:
        return torch.cat(new_batch_g), torch.cat(new_batch_a), torch.cat(new_batch_r)

    return torch.cat(new_batch_g), torch.cat(new_batch_a)


def get_acc(
    s1: torch.Tensor, s2: torch.Tensor, model: nn.Module, target: torch.Tensor
) -> float:
    """Calculate model's accuracy.

    :param s1: the first well-interval
    :param s2: the second well-interval
    :param model: transformer model
    :param target: target values: 1 if s1 and s2 belong to the same well otherwise 0
    :return: model's accuracy
    """
    return accuracy_score(
        target.detach().cpu().numpy(),
        (model((s1, s2)).detach().cpu().numpy() > 0.5).astype(int),
    )
