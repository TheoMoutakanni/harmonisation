import numpy as np
import torch
import torch.nn as nn

import dipy.reconst.dti as dti

from dipy.data import get_sphere
from dipy.reconst.shm import (sph_harm_ind_list, cart2sphere, real_sph_harm,
                              sh_to_rh, forward_sdeconv_mat)

from harmonisation.functions import shm


def nanmean(v, inplace=False, *args, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


def torch_norm(vect):
    return torch.sqrt(torch.sum(vect**2, axis=-1))


class AngularCorrCoeff(nn.Module):
    def __init__(self):
        super(AngularCorrCoeff, self).__init__()

    def forward(self, sh_U, sh_V, mask=None):
        if mask is not None:
            sh_U = sh_U * mask
            sh_V = sh_V * mask
        acc = torch.sum(sh_U * sh_V, axis=-1) / \
            (torch_norm(sh_U) * torch_norm(sh_V))
        return acc


def weighted_mse(X, Z, weight):
    mse = (X - Z)**2
    mse = mse * weight.expand_as(mse)
    return torch.sum(mse) / torch.sum(weight)


class NegativeWeightedMSE(nn.Module):
    def __init__(self):
        super(NegativeWeightedMSE, self).__init__()

    def forward(self, X, Z, mask):
        return -weighted_mse(X, Z, mask)


class Accuracy(nn.Module):
    def __init__(self, force_label=None):
        super(Accuracy, self).__init__()
        self.force_label = force_label

    def forward(self, logits, labels=None):
        if self.force_label is not None:
            labels = self.force_label
        elif len(labels.shape) == 1:
            labels = labels[None]
        if logits.shape[1] > 1:
            predicted = torch.argmax(logits, dim=1)
        else:
            proba = torch.sigmoid(logits)
            predicted = (proba >= 0.5).float()
        accuracy = (predicted == labels).float().mean()
        return accuracy


def torch_RIS(X):
    sh_order = shm.order_from_ncoef(X.shape[-1])
    m, n = shm.sph_harm_ind_list(sh_order)
    RIS = []
    for i in np.arange(sh_order // 2 + 1) * 2:
        RIS.append(X[..., n == i].sum(-1))
    return torch.stack(RIS).permute(*range(1, len(X.shape)), 0)


class NegativeRISMSE(nn.Module):
    def __init__(self):
        super(NegativeRISMSE, self).__init__()

    def forward(self, X_sh, Z_sh, mask):
        X_RIS = torch_RIS(X_sh)
        Z_RIS = torch_RIS(Z_sh)
        mse_RIS = weighted_mse_loss(X_RIS, Z_RIS, weight)
        return -mse_RIS


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


class NegativeGfaMSE(nn.Module):
    def __init__(self):
        super(NegativeGfaMSE, self).__init__()

    def forward(self, X_sh, Z_sh, mask):
        X_gfa = torch_gfa(X_sh)
        Z_gfa = torch_gfa(Z_sh)
        mse_gfa = weighted_mse_loss(X_gfa, Z_gfa, weight.squeeze())
        return -mse_gfa


def torch_anisotropic_power(sh_coeffs, norm_factor=0.00001, power=2,
                            non_negative=True):
    """Calculates anisotropic power map with a given SH coefficient matrix
    """

    dim = sh_coeffs.shape[:-1]
    n_coeffs = sh_coeffs.shape[-1]
    max_order = shm.calculate_max_order(n_coeffs)
    ap = torch.zeros(dim).to(sh_coeffs.device)
    n_start = 1
    for L in range(2, max_order + 2, 2):
        n_stop = n_start + (2 * L + 1)
        ap_i = torch.mean(
            torch.abs(sh_coeffs[..., n_start:n_stop]) ** power, -1)
        ap += ap_i
        n_start = n_stop

    # Shift the map to be mostly non-negative,
    # only applying the log operation to positive elements
    # to avoid getting numpy warnings on log(0).
    # It is impossible to get ap values smaller than 0.
    # Also avoids getting voxels with -inf when non_negative=False.

    if ap.ndim < 1:
        # For the off chance we have a scalar on our hands
        ap = torch.reshape(ap, (1, ))
    log_ap = torch.zeros_like(ap).to(ap.device)
    log_ap[ap > 0] = torch.log(ap[ap > 0]) - np.log(norm_factor)

    # Deal with residual negative values:
    if non_negative:
        if torch.is_tensor(log_ap):
            # zero all values < 0
            log_ap[log_ap < 0] = 0
        else:
            # assume this is a singleton float (input was 1D):
            if log_ap < 0:
                return 0
    return log_ap


class NegativeAPMSE(nn.Module):
    def __init__(self):
        super(NegativeAPMSE, self).__init__()

    def forward(self, X_sh, Z_sh, mask):
        X_ap = torch_anisotropic_power(X_sh)
        Z_ap = torch_anisotropic_power(Z_sh)
        mse_ap = weighted_mse_loss(X_ap, Z_ap, weight.squeeze())
        return -mse_ap


def torch_fa(evals):
    all_zero = (evals == 0).all(axis=-1)
    ev1, ev2, ev3 = evals[..., 0], evals[..., 1], evals[..., 2]
    fa = torch.sqrt(0.5 * ((ev1 - ev2) ** 2 +
                           (ev2 - ev3) ** 2 +
                           (ev3 - ev1) ** 2) /
                    ((evals * evals).sum(-1) + all_zero))

    return fa


class NumberSmallfODF(nn.Module):
    def __init__(self, gtab, response, sh_order, lambda_=1, tau=0.1):
        super(NumberSmallfODF, self).__init__()

        m, n = sph_harm_ind_list(sh_order)

        # x, y, z = gtab.gradients[~gtab.b0s_mask].T
        # r, theta, phi = cart2sphere(x, y, z)
        # self.B_dwi = real_sph_harm(m, n, theta[:, None], phi[:, None])
        self.B_dwi = shm.get_B_matrix(gtab, sh_order)

        self.sphere = get_sphere('symmetric362')

        r, theta, phi = cart2sphere(
            self.sphere.x,
            self.sphere.y,
            self.sphere.z
        )
        self.B_reg = real_sph_harm(m, n, theta[:, None], phi[:, None])

        S_r = shm.estimate_response(gtab, response[0:3], response[3])
        r_sh = np.linalg.lstsq(self.B_dwi, S_r[~gtab.b0s_mask], rcond=-1)[0]
        n_response = n
        m_response = m
        r_rh = sh_to_rh(r_sh, m_response, n_response)
        R = forward_sdeconv_mat(r_rh, n)

        # scale lambda_ to account for differences in the number of
        # SH coefficients and number of mapped directions
        # This is exactly what is done in [4]_
        lambda_ = (lambda_ * R.shape[0] * r_rh[0] /
                   (np.sqrt(self.B_reg.shape[0]) * np.sqrt(362.)))
        self.B_reg = self.B_reg * lambda_
        self.B_reg = nn.Parameter(torch.FloatTensor(self.B_reg),
                                  requires_grad=False)

        self.tau = tau

    def forward(self, fodf_sh, mask):
        threshold = self.B_reg[0, 0] * self.tau * fodf_sh[..., 0:1]
        fodf = torch.einsum("...i,ij", fodf_sh, self.B_reg.T)
        p = torch.sum((fodf < threshold) * mask) / torch.sum(mask)
        return p


def get_metric_dict():
    """Return a dict with all the metric functions
    Metrics must be greater when better, so -mse instead of mse for example"""
    return {
        'acc': AngularCorrCoeff,
        'mse': NegativeWeightedMSE,
        'mse_RIS': NegativeRISMSE,
        'mse_gfa': NegativeGfaMSE,
        'mse_ap': NegativeAPMSE,

        'accuracy': Accuracy,
        'small_fodf': NumberSmallfODF,
    }


def get_metric_fun(metric_specs, device):
    """Take a dictionnary of metrics specs and a device et return a dict
    with a metric module and its paramters.

    Attributes:
        metric_specs (list[dict]): a list of all metrics specifications:
        {"type" (float): the metric function name found in "get_metric_dict",
         "parameters" (dict): dict of parameters to initialize the module,
         "inputs" (list[str]): the list of inputs name IN ORDER for the forward
                               method of the module,}
        device (str or torch.device): the device of the module

    Returns:
        A dict of dict where each key is the concatenation of the metric name
        and the first input:
        {"fun" (nn.Module): the metric module,
         "inputs" (list[str]): the list of inputs IN ORDER,
         "type" (float): the metric name}
    """
    metric_dict = get_metric_dict()
    metrics = {}
    for specs in metric_specs:
        d = dict()
        d['fun'] = metric_dict[specs["type"]](**specs["parameters"]).to(device)
        d['inputs'] = specs['inputs']
        for input_params in d['inputs']:
            if "net" not in input_params:
                input_params["net"] = "dataset"
            if "detach" not in input_params:
                input_params["detach"] = False
            if "recompute" not in input_params:
                input_params["recompute"] = False
            if "from" not in input_params:
                input_params["from"] = "dataset"
        d['type'] = specs['type']

        input_name = d['inputs'][0]['name']
        if d['inputs'][0]['from'] != "dataset":
            input_name += '_' + (d['inputs'][0]['from'])
        metrics[d['type'] + '_' + input_name] = d
    return metrics
