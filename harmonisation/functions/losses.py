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


def loss_dwi_mse(B, voxels_to_take="center"):
    """B is the matrix to pass from sh to dwi
    voxels_to_take : "center", "all", or a matrix of indices
    """
    B = torch.FloatTensor(B).to("cuda")
    mini = .001
    maxi = .999

    def loss(X, Z, mask):
        if torch.is_tensor(voxels_to_take):
            X = X[:, voxels_to_take]
            Z = Z[:, voxels_to_take]
        elif voxels_to_take == "center":
            c_x, c_y, cz = np.array(X.shape[1:4]) // 2
            X = X[:, c_x, c_y, c_z]
            Z = Z[:, c_x, c_y, c_z]

        B = B.to(X.device)
        X = torch.einsum("...i,ij", X, B.T).clamp(mini, maxi)
        Z = torch.einsum("...i,ij", Z, B.T).clamp(mini, maxi)
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


def get_loss_dict():
    return {
        'acc': loss_acc,
        'mse': loss_mse,
        'dwi_mse': loss_dwi_mse,
        'mse_gfa': loss_acc_mse,
        'mse_RIS': loss_mse_RIS,
        'mse_ap': loss_mse_anisotropic_power,

        'bce': nn.BCELoss,
        'cross_entropy': nn.CrossEntropyLoss
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
