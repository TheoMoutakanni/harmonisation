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


def loss_acc():
    def loss(X, Z, mask):
        acc = torch_angular_corr_coeff(X + 1e-8, Z + 1e-8)
        acc = 1 - weighted_mean(acc, mask.squeeze()).mean()
        return acc
    return loss


def loss_mse():
    def loss(X, Z, mask):
        mse = weighted_mse(X, Z, mask)
        return mse
    return loss


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


def loss_acc_mse(alpha=0.5):
    def loss(X, Z, mask):
        acc = torch_angular_corr_coeff(X + 1e-8, Z + 1e-8)
        acc = 1 - weighted_mean(acc, mask.squeeze()).mean()
        mse = weighted_mse(X, Z, mask)
        loss = alpha * acc + mse
        return loss
    return loss


def loss_mse_gfa():
    def loss(X, Z, mask):
        mse_gfa = torch_mse_gfa(X, Z, mask).mean()
        return mse_gfa
    return loss


def loss_mse_anisotropic_power():
    def loss(X, Z, mask):
        mse_gfa = torch_mse_anisotropic_power(X, Z, mask).mean()
        return mse_gfa
    return loss


def loss_mse_RIS():
    def loss(X, Z, mask):
        mse_RIS = torch_mse_RIS(X, Z, mask).mean()
        return mse_RIS
    return loss


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


class StyleLoss(nn.Module):
    def __init__(self, target_features, layers_coeff=None):
        """target_features: dict {layer_name: layer_features}
        """
        super(StyleLoss, self).__init__()
        self.target_features = {name: gram_matrix(torch.FloatTensor(feature))
                                for name, feature in target_features.items()}

        if layers_coeff is None:
            layers_coeff = {name: 1. for name in self.target_features.keys()}
        self.layers_coeff = layers_coeff

    def forward(self, input):
        loss = 0
        for layer_name, target_G in self.target_features.items():
            G = gram_matrix(input[layer_name])
            target_G = target_G.to(G.device).expand_as(G)
            loss += self.layers_coeff[layer_name] * F.mse_loss(G, target_G)
        loss /= float(len(self.target_features))
        return loss.mean()


def get_loss_dict():
    return {
        'acc': loss_acc,
        'mse': loss_mse,
        'dwi_mse': DWIMSELoss,
        'mse_gfa': loss_acc_mse,
        'mse_RIS': loss_mse_RIS,
        'mse_ap': loss_mse_anisotropic_power,

        'style': StyleLoss,

        'bce': nn.BCELoss,
        'cross_entropy': nn.CrossEntropyLoss,
        'focal': FocalLoss,
    }


def get_loss_fun(loss_specs):
    loss_dict = get_loss_dict()
    losses_fun = [loss_dict[loss["type"]](**loss["parameters"])
                  for loss in loss_specs]
    losses_coeff = [loss['coeff'] for loss in loss_specs]

    def loss(*args, **kwargs):
        losses = torch.stack([coeff * fun(*args, **kwargs)
                              for fun, coeff in zip(losses_fun, losses_coeff)],
                             dim=0)
        return losses.sum()
    return loss
