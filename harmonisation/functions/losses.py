import torch
import torch.nn as nn

from .metrics import *


def weighted_mse_loss(X, Z, weight):
    """Overcomplicated to remove nans with div by 0 ... """
    batch = X.shape[0]
    weigth_sum = weight.view(batch, -1).sum(1)
    loss = ((X - Z)**2 * weight).view(batch, -1).sum(1)
    weigth_sum, loss = weigth_sum[weigth_sum != 0], loss[weigth_sum != 0]
    loss /= weigth_sum
    return loss


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
        mse = weighted_mse_loss(X, Z, mask).mean()
        return mse
    return loss


def loss_acc_mse(alpha=0.5):
    def loss(X, Z, mask):
        acc = torch_angular_corr_coeff(X + 1e-8, Z + 1e-8)
        acc = 1 - weighted_mean(acc, mask.squeeze()).mean()
        mse = weighted_mse_loss(X, Z, mask).mean()
        loss = alpha * acc + mse
        return loss
    return loss


def loss_mse_gfa():
    def loss(X, Z, mask):
        mse_gfa = torch_mse_gfa(X, Z, mask)
        return mse_gfa
    return loss


def loss_mse_RIS():
    def loss(X, Z, mask):
        mse_RIS = torch_mse_RIS(X, Z, mask)
        return mse_RIS
    return loss


def get_loss_fun(loss_specs):
    if loss_specs["type"] == "acc":
        return loss_acc(**loss_specs["parameters"])
    elif loss_specs["type"] == "mse":
        return loss_mse(**loss_specs["parameters"])
    elif loss_specs["type"] == "acc_mse":
        return loss_acc_mse(**loss_specs["parameters"])
    elif loss_specs["type"] == "bce":
        return nn.BCELoss(**loss_specs["parameters"])
    elif loss_specs["type"] == "mse_gfa":
        return loss_mse_gfa(**loss_specs["parameters"])
    elif loss_specs["type"] == "mse_RIS":
        return loss_mse_RIS(**loss_specs["parameters"])
    elif loss_specs["type"] == "cross_entropy":
        return nn.CrossEntropyLoss(**loss_specs["parameters"])
