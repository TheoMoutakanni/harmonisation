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


def get_metrics_fun():
    return {
        'acc': lambda x, z, mask: torch_angular_corr_coeff(x * mask, z * mask),
        'mse': lambda x, z, mask: -weighted_mse_loss(x, z, mask),
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
