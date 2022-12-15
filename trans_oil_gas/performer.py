"""Performer adaptation inspired by TODO ADD LINK."""


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.activations import get_activation


def iid_gaussian(m: int, d: int) -> torch.Tensor:
    """Generate IID Gaussian random features.

    :param m:
    :param d:
    :return: tensor of elements from standard distribution of size m x d
    """
    return torch.normal(0.0, 1.0, size=(m, d))


def orthogonal_gaussian_random_feature(m, d) -> torch.Tensor:
    """Generate orthogonal Gaussian random features.
    :param m:
    :param d:
    :return:
    """

    def orthogonal_square() -> None:
        """Create orthogonal square matrix using Gram-Schmidt."""
        q, _ = torch.qr(iid_gaussian(d, d))
        return q.T

    num_squares = int(m / d)
    blocks = [orthogonal_square() for _ in range(num_squares)]

    remainder = m - d * num_squares
    if remainder:
        blocks.append(orthogonal_square()[:remainder])

    matrix = torch.cat(blocks)
    matrix /= torch.sqrt(torch.tensor(num_squares + remainder / d))

    return matrix


def relu_kernel_transformation(
    data, is_query, projection_matrix=None, numerical_stabilizer=0.000001
):
    """Computes features for the ReLU-kernel.
    ReLU kernel에 대한 Random Features를 계산 from https://arxiv.org/pdf/2009.14794.pdf.
    Args:
    data: 입력데이터 텐서, the shape [B, L, H, D], where: B - batch
                       dimension, L - attention dimensions, H - heads, D - features.
    is_query: 입력데이터가 쿼리인지 아닌지, 쿼리 또는 키인지를 나타낸다. indicates whether input data is a query oor key tensor.
    projection_matrix: [M,D] 모양을 가진 랜덤 가우시안 매트릭스 random Gaussian matrix of shape [M, D], where M stands
      for the number of random features and each D x D sub-block has pairwise
      orthogonal rows.
    numerical_stabilizer: 수치 안정성을 위한 작은 값의 양의 상수
    Returns:
    대응되는 kernel feature map
    """
    del is_query
    relu = nn.ReLU()
    if projection_matrix is None:
        return nn.relu(data) + numerical_stabilizer
    else:
        ratio = 1.0 / math.sqrt(projection_matrix.shape[0])
        data_dash = ratio * torch.einsum("blhd,md->blhm", data, projection_matrix)
        return relu(data_dash) + numerical_stabilizer


def softmax_kernel_transformation(
    data, is_query, projection_matrix=None, numerical_stabilizer=0.000001
):
    """
    FAVOR+ 메커니즘을 사용하여 softmax kernel에 대한 random feature 계산
    :param data: 입력 텐서. [B,L,H,D] B-batch dimension, L- attention dimensions, H- Heads, D- features
    :param is_query: 입력 값이 쿼리 또는 인지 나타내는 값
    :param projection_matrix: [M, D]의 모양을 가진 랜덤 가우시안 매트릭스
               M - M은 Random Feature 수를 의미하며,
               각각의 [D,D] 서브 블록은 pairwise orthogonal rows를 가진다.
    :param numerical_stabilizer: 수치 안정성을 위한 작은 값의 양의 상수
    :return:
    """
    data_normalizer = data.shape[-1] ** -0.25
    data = data_normalizer * data
    ratio = projection_matrix.shape[0] ** -0.5

    data_dash = torch.einsum("blhd,md->blhm", data, projection_matrix)

    diag_data = torch.square(data)
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = diag_data / 2.0
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(
                data_dash
                - diag_data
                - torch.max(data_dash, dim=-1, keepdims=True).values
            )
            + numerical_stabilizer
        )
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash))
            + numerical_stabilizer
        )

    return data_dash


def noncausal_numerator(qs, ks, vs):
    """Computes not-normalized FAVOR noncausal attention AV.
    Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].
    vs: value tensor of the shape [L,B,H,D].
    Returns:
    Not-normalized FAVOR noncausal attention AV.
    """
    kvs = torch.einsum("lbhm,lbhd->bhmd", ks, vs)
    return torch.einsum("lbhm,bhmd->lbhd", qs, kvs)


def noncausal_denominator(qs, ks, device):
    """Computes FAVOR normalizer in noncausal attention.
    Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].
    Returns:
    FAVOR normalizer in noncausal attention.
    """
    qs, ks = qs.to(device), ks.to(device)
    all_ones = torch.ones([ks.shape[0]]).to(device)
    ks_sum = torch.einsum("lbhm,l->bhm", ks, all_ones)
    return torch.einsum("lbhm,bhm->lbh", qs, ks_sum)


def causal_numerator(qs, ks, vs, device):
    """Computes not-normalized FAVOR causal attention A_{masked}V.
    Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].
    vs: value tensor of the shape [L,B,H,D].
    Returns:
    Not-normalized FAVOR causal attention A_{masked}V.
    """

    result = []
    sums = torch.zeros_like(torch.einsum("ijk,ijl->ijkl", ks[0], vs[0])).to(device)

    for index in range(qs.shape[0]):
        sums = sums + torch.einsum("ijk,ijl->ijkl", ks[index], vs[index])
        result.append(torch.einsum("ijkl,ijk->ijl", sums, qs[index])[None, Ellipsis])

    result = torch.cat(result, dim=0)

    return result


def causal_denominator(qs, ks, device):
    """Computes FAVOR normalizer in causal attention.
    Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].
    Returns:
    FAVOR normalizer in causal attention.
    """

    result = []
    sums = torch.zeros_like(ks[0]).to(device)

    for index in range(qs.shape[0]):
        sums = sums + ks[index]
        result.append(torch.sum(qs[index] * sums, dim=2)[None, Ellipsis])

    result = torch.cat(result, dim=0)

    return result


def favor_attention(
    query,
    key,
    value,
    kernel_transformation,
    causal=False,
    projection_matrix=None,
    device=None,
):
    """
    favor_attention 계산
    :param query: 쿼리
    :param key: 키
    :param value: 밸류
    :param kernel_transformation: kernel feature를 얻기 위한 tranformation.
         relu_kernel_transformation나 softmax_kernel_transformation 사용.
    :param causal: causl or not
    :param projection_matrix: 사용될 projection matrix
    :return: Favor+ normalized attention
    """
    query, key, value = query.to(device), key.to(device), value.to(device)
    projection_matrix = projection_matrix.to(device)

    # Kernel Transformation
    query_prime = kernel_transformation(query, True, projection_matrix)  # [B, L, H, M]
    key_prime = kernel_transformation(key, False, projection_matrix)  # [B,L,H,M]

    # Transpose
    query_prime = torch.transpose(query_prime, 0, 1)  # [L,B,H,M]
    key_prime = torch.transpose(key_prime, 0, 1)  # [L,B,H,M]
    value = torch.transpose(value, 0, 1)  # [L,B,H,D]

    # Causal or Not
    if causal:
        av_attn = causal_numerator(query_prime, key_prime, value, device=device)
        attn_normalizer = causal_denominator(query_prime, key_prime, device=device)
    else:
        av_attn = noncausal_numerator(query_prime, key_prime, value)
        attn_normalizer = noncausal_denominator(query_prime, key_prime, device=device)

    av_attn = torch.transpose(av_attn, 0, 1)
    attn_normalizer = torch.transpose(attn_normalizer, 0, 1)
    attn_normalizer = attn_normalizer.unsqueeze(-1)

    return av_attn / attn_normalizer


class MultiHeadFAVORAttention(nn.Module):
    def __init__(
        self,
        head_num=8,
        dim=512,
        dropout=0.1,
        nb_random_features=256,
        causal=False,
        use_relu_kernel=False,
        device=None,
    ):
        super(MultiHeadFAVORAttention, self).__init__()

        self.head_num = head_num
        self.dim = dim
        self.d_k = self.d_v = dim // head_num
        self.nb_random_features = nb_random_features
        self.device = device
        self.kernel_transformation = (
            softmax_kernel_transformation
            if not use_relu_kernel
            else relu_kernel_transformation
        )

        random_features = orthogonal_gaussian_random_feature(
            self.nb_random_features, self.d_k
        )
        random_features = random_features.to(self.device)

        self.register_buffer("projection_matrix", random_features)

        self.causal = causal

        self.w_q = nn.Linear(dim, dim)
        self.w_k = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)
        self.w_o = nn.Linear(dim, dim)

        self.favor_attention = favor_attention
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):

        batche_num = query.size(0)

        query = self.w_q(query).view(batche_num, -1, self.head_num, self.d_k)
        key = self.w_k(key).view(batche_num, -1, self.head_num, self.d_k)
        value = self.w_v(value).view(batche_num, -1, self.head_num, self.d_k)

        attention_result = self.favor_attention(
            query,
            key,
            value,
            kernel_transformation=self.kernel_transformation,
            causal=self.causal,
            projection_matrix=self.projection_matrix,
            device=self.device,
        )
        attention_result = attention_result.contiguous().view(
            batche_num, -1, self.head_num * self.d_k
        )

        return self.dropout(self.w_o(attention_result))


class FeedForward(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(dim, dim * 4)
        self.w_2 = nn.Linear(dim * 4, dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):

        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualConnection(nn.Module):
    def __init__(self, size, dropout):
        super(ResidualConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout((sublayer(self.norm(x))))


class PerformerEncoder(nn.Module):
    def __init__(
        self, dim, head_num, dropout, nb_random_features, device, use_relu_kernel=False
    ):
        super(PerformerEncoder, self).__init__()
        self.multi_head_attention = MultiHeadFAVORAttention(
            dim=dim,
            head_num=head_num,
            nb_random_features=nb_random_features,
            device=device,
            use_relu_kernel=use_relu_kernel,
        )
        self.residual_1 = ResidualConnection(dim, dropout=dropout)

        self.feed_forward = FeedForward(dim)
        self.residual_2 = ResidualConnection(dim, dropout=dropout)

    def forward(self, input):
        x = self.residual_1(input, lambda x: self.multi_head_attention(x, x, x))
        x = self.residual_2(x, lambda x: self.feed_forward(x))
        return x
