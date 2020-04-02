import numpy as np
import torch

from dipy.reconst.shm import sph_harm_ind_list, order_from_ncoef


def torch_norm(vect):
    return torch.sqrt(torch.sum(vect**2, axis=-1))


def torch_angular_corr_coeff(sh_U, sh_V):
    assert sh_U.shape[-1] == sh_V.shape[-1], "SH orders must be the same"

    acc = torch.sum(sh_U * sh_V, axis=-1) / \
        (torch_norm(sh_U) * torch_norm(sh_V))

    return acc


def weighted_mse_loss(X, Z, weight):
    return torch.sum(weight * (X - Z) ** 2, axis=-1) / torch.sum(weight)


def torch_accuracy(labels, proba):
    predicted = (proba > 0.5).float()
    accuracy = (predicted == labels).float().mean()
    return accuracy


def torch_RIS(X):
    sh_order = order_from_ncoef(X.shape[-1])
    m, n = sph_harm_ind_list(sh_order)
    RIS = []
    for i in np.arange(sh_order // 2 + 1) * 2:
        RIS.append(X[..., n == i].sum(-1))
    return torch.stack(RIS).permute(*range(1, len(X.shape)), 0)


def torch_mse_RIS(X, Z, weight):
    X_RIS = torch_RIS(X)
    Z_RIS = torch_RIS(Z)
    mse_RIS = weighted_mse_loss(X_gra, Z_gfa, weight)
    return mse_RIS.mean()


def torch_gfa(X):
    sh0_index = 0

    X2 = X**2
    numer = X2[..., sh0_index]
    denom = X2.sum(-1)
    # The sum of the square of the coefficients being zero is the same as all
    # the coefficients being zero
    allzero = denom == 0
    # By adding 1 to numer and denom where both and are 0, we prevent 0/0
    numer = numer + allzero
    denom = denom + allzero
    return torch.sqrt(1. - (numer / denom))


def torch_mse_gfa(X, Z, weight):
    X_gfa = torch_gfa(X)
    Z_gfa = torch_gfa(Z)
    mse_gfa = weighted_mse_loss(X_gfa, Z_gfa, weight.squeeze())
    return mse_gfa.mean()


def get_metrics_fun():
    """Return a dict with all the metrics to be computed during an epoch
    Metrics must be greater when better, so -mse instead of mse for example"""
    return {
        'acc': lambda x, z, mask: torch_angular_corr_coeff(x * mask, z * mask),
        'mse': lambda x, z, mask: -weighted_mse_loss(x, z, mask),
        'mse_RIS': lambda x, z, mask: -torch_mse_RIS(x, z, mask),
        'mse_gfa': lambda x, z, mask: -torch_mse_gfa(x, z, mask),
        'accuracy': lambda labels, proba, mask: torch_accuracy(labels, proba)
    }


def compute_metrics_dataset(data_true, data_pred, metrics):
    metrics_fun = get_metrics_fun()

    metrics_output = [
        {metric: np.nanmean(metrics_fun[metric](
            data_true[name]['sh'],
            data_pred[name],
            data_true[name]['mask']))
            for metric in metrics}
        for name in list(set(data_pred.keys()) & set(data_true.keys()))]

    return metrics_output
