import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from .metrics import *


def weighted_mse(X, Z, weight):
    """Overcomplicated to remove nans with div by 0 ... """
    loss = ((X - Z)**2 * weight)
    loss = loss * weight.expand_as(loss)
    return loss.mean()


def weighted_mean(v, weight):
    batch = v.shape[0]
    weigth_sum = weight.view(batch, -1).sum(1)
    loss = (v * weight).view(batch, -1).sum(1)
    weigth_sum, loss = weigth_sum[weigth_sum != 0], loss[weigth_sum != 0]
    loss /= weigth_sum
    return loss[torch.isnan(loss)]


class AccLoss(nn.Module):
    def __init__(self):
        super(AccLoss, self).__init__()

    def forward(self, X, Z, mask):
        acc = torch_angular_corr_coeff(X + 1e-8, Z + 1e-8)
        acc = 1 - weighted_mean(acc, mask.squeeze()).mean()
        return acc


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, X, Z, mask):
        return weighted_mse(X, Z, mask)


class DWIMSELoss(nn.Module):
    def __init__(self, B, mean=None, std=None, voxels_to_take="center"):
        """B is the matrix to pass from sh to dwi
        voxels_to_take : "center", "all", or a matrix of indices
        """
        super(DWIMSELoss, self).__init__()
        self.B = torch.FloatTensor(B).to("cuda")
        self.mini = .001
        self.maxi = .999
        self.voxels_to_take = voxels_to_take

        self.mean = torch.FloatTensor(mean) if mean is not None else None
        self.std = torch.FloatTensor(std) if std is not None else None

    def forward(self, X, Z, mask):
        if torch.is_tensor(self.voxels_to_take):
            X = X[:, self.voxels_to_take]
            Z = Z[:, self.voxels_to_take]
        elif self.voxels_to_take == "center":
            c_x, c_y, cz = np.array(X.shape[1:4]) // 2
            X = X[:, c_x, c_y, c_z]
            Z = Z[:, c_x, c_y, c_z]

        if self.mean is not None and self.std is not None:
            self.mean = self.mean.to(X.device)
            self.std = self.std.to(X.device)
            X = X * self.std + self.mean
            Z = Z * self.std + self.mean

        X, X_b0 = X[..., 1:], X[..., :1]
        Z, Z_b0 = Z[..., 1:], Z[..., :1]

        X = X * X_b0
        Z = Z * Z_b0

        self.B = self.B.to(X.device)
        X = torch.einsum("...i,ij", X, self.B.T).clamp(self.mini, self.maxi)
        Z = torch.einsum("...i,ij", Z, self.B.T).clamp(self.mini, self.maxi)
        mse = weighted_mse(X, Z, mask)
        return mse


class GFAMSELoss(nn.Module):
    def __init__(self):
        super(GFAMSELoss, self).__init__()

    def forward(self, X, Z, mask):
        return torch_mse_gfa(X, Z, mask).mean()


class APMSELoss(nn.Module):
    def __init__(self):
        super(APMSELoss, self).__init__()

    def forward(self, X, Z, mask):
        return torch_mse_anisotropic_power(X, Z, mask).mean()


class RISMSELoss(nn.Module):
    def __init__(self):
        super(RISMSELoss, self).__init__()

    def forward(self, X, Z, mask):
        return torch_mse_RIS(X, Z, mask).mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int, long)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,H,W,D,C => N*H*W*D,C
            input = input.view(-1, input.size(-1))
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def gram_matrix(input):
    shape = np.prod(list(input.size()))
    if len(input.size()) > 2:
        features = input.view(input.size()[0], input.size()[1], -1)
    else:
        features = torch.unsqueeze(input, -1)

    G = torch.matmul(features, features.transpose(1, 2))

    return G.div(shape)


class GramLoss(nn.Module):
    """MSE of Gramm Matrices"""

    def __init__(self, target_features, layers_coeff=None):
        """target_features: dict {layer_name: layer_features}
        """
        super(GramLoss, self).__init__()
        self.target_G = nn.Parameter(
            gram_matrix(torch.FloatTensor(target_features)),
            requires_grad=False)
        self.layers_coeff = layers_coeff if layers_coeff is not None else 1.

    def forward(self, inputs):
        G = gram_matrix(inputs)
        loss = self.layers_coeff * F.mse_loss(G, self.target_G.expand_as(G))
        return loss.mean()


class FeatureLoss(nn.Module):
    """MSE of each feature vector"""

    def __init__(self, target_features, layers_coeff=None):
        """target_features: dict {layer_name: layer_features}
        """
        super(FeatureLoss, self).__init__()
        self.target_features = nn.Parameter(
            torch.FloatTensor(target_features).mean(0),
            requires_grad=False)

        self.layers_coeff = layers_coeff if layers_coeff is not None else 1.

    def forward(self, inputs):
        loss = self.layers_coeff * \
            F.mse_loss(inputs, self.target_features.expand_as(inputs))
        return loss.mean()


class SoftHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(SoftHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * \
            (torch.arange(bins).float() + 0.5)
        self.centers = nn.Parameter(self.centers, requires_grad=False)

    def forward(self, x):
        x = torch.unsqueeze(x, 1) - torch.unsqueeze(self.centers, 2)
        x = torch.sigmoid(self.sigma * (x + self.delta / 2)) - \
            torch.sigmoid(self.sigma * (x - self.delta / 2))
        x = x.sum(dim=2)
        return x


class HistLoss(nn.Module):
    """MSE of histogram"""

    def __init__(self, data, bins, min, max, scale=1., sigma=None):
        """target_features: dict {layer_name: layer_features}
        """
        super(HistLoss, self).__init__()
        data *= scale
        self.target_hist, _ = np.histogram(data,
                                           bins=bins,
                                           range=[min * scale, max * scale])
        self.target_hist = nn.Parameter(torch.FloatTensor(target_hist),
                                        requires_grad=False)
        self.bins = bins
        self.max = max * scale
        self.min = min * scale
        self.scale = scale

        if sigma is not None:
            self.sigma = sigma
        else:
            self.sigma = 3 * np.std(data)

        self.hist_fun = SoftHistogram(bins=bins, min=min, max=max, sigma=sigma)

    def forward(self, inputs):
        inputs = inputs.view(inputs.shape[0], -1)
        inputs *= self.scale
        hist = hist_fun(inputs)
        loss = F.mse_loss(hist, self.target_hist.expand_as(hist))
        return loss.mean()


def get_loss_dict():
    return {
        'acc': AccLoss,
        'mse': WeightedMSELoss,
        'dwi_mse': DWIMSELoss,
        'mse_RIS': RISMSELoss,
        'mse_ap': APMSELoss,
        'mse_gfa': GFAMSELoss,

        'gram': GramLoss,
        'feature': FeatureLoss,
        'hist': HistLoss,

        'bce': nn.BCELoss,
        'cross_entropy': nn.CrossEntropyLoss,
        'focal': FocalLoss,
    }


def get_loss_fun(loss_specs, device):
    loss_dict = get_loss_dict()
    losses = []
    for specs in loss_specs:
        d = dict()
        d['fun'] = loss_dict[specs["type"]](**specs["parameters"]).to(device)
        d['inputs'] = specs['inputs']
        d['coeff'] = specs['coeff']
        d['type'] = specs['type']
        losses.append(d)
    return losses
