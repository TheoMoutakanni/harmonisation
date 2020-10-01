import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from dipy.data import get_sphere
from dipy.reconst.shm import (sph_harm_ind_list, cart2sphere, real_sph_harm,
                              sh_to_rh, forward_sdeconv_mat)
from harmonisation.functions import metrics, shm


def weighted_mean(v, weight):
    batch = v.shape[0]
    weigth_sum = weight.view(batch, -1).sum(1)
    loss = (v * weight).view(batch, -1).sum(1)
    weigth_sum, loss = weigth_sum[weigth_sum != 0], loss[weigth_sum != 0]
    loss /= weigth_sum
    return loss[torch.isnan(loss)]


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, X, Z, mask):
        return metrics.weighted_mse(X, Z, mask)


class AccLoss(nn.Module):
    """Angular Correlation Coefficient Loss"""

    def __init__(self):
        super(AccLoss, self).__init__()

        self.acc_module = metrics.AngularCorrCoeff()

    def forward(self, X, Z, mask):
        acc = self.acc_module(X + 1e-8, Z + 1e-8, mask)
        acc = 1 - weighted_mean(acc, mask.squeeze()).mean()
        return acc


class DWIMSELoss(nn.Module):
    """MSE Loss between a true DWI signal and a predicted SH signal remapped
    on the DWI directions"""

    def __init__(self, B, where_b0, use_b0=False,
                 mean=None, std=None, b0_mean=None, b0_std=None):
        """
        Attributes:
            B (np.array(nb_dwi_directions, nb_sh_coeff)): matrix to fit SH->DWI
            where_b0 (array[bool]): where are the b0 values in the DWI
            use_b0 (bool): do the SH inputs have b0 values as 1st channel?
            mean (array[float]): the mean to unnormalize the SH coeffs
            std (array[float]): the std to unnormalize the SH coeffs
            mean_b0 (array[float]): the mean to unnormalize the b0 values
            std_b0 (array[float]): the std to unnormalize the b0 values
        """
        super(DWIMSELoss, self).__init__()
        self.B = nn.Parameter(torch.FloatTensor(B), requires_grad=False)
        self.mini = .001
        self.maxi = .999
        self.mean = nn.Parameter(torch.FloatTensor(
            mean), requires_grad=False) if mean is not None else None
        self.std = nn.Parameter(torch.FloatTensor(
            std), requires_grad=False) if std is not None else None
        self.b0_mean = nn.Parameter(torch.FloatTensor(
            b0_mean), requires_grad=False) if b0_mean is not None else None
        self.b0_std = nn.Parameter(torch.FloatTensor(
            b0_std), requires_grad=False) if b0_std is not None else None
        self.use_b0 = use_b0
        self.where_b0 = where_b0

    def forward(self, X, dwi, mask):
        """Return the MSE between the fitted DWI signal on the predicted
        SH coeffs of X and the true DWI signal

        Attributes:
            X (torch.tensor(batch, X, Y, Z, C)): Tensor of SH coeffs, if use_b0
                                                 is True, the first channel is
                                                 the b0 value.
            dwi (torch.tensor(batch, X, Y, Z, C)): Tensor of DWI coeffs
        Returns:
            mse (torch.tensor): the computed MSE
        """
        if self.use_b0:
            X_sh, X_b0 = X[..., 1:], X[..., :1]

            if self.mean is not None and self.std is not None:
                X_sh = X_sh * self.std + self.mean
            if self.b0_mean is not None and self.b0_std is not None:
                X_b0 = X_b0 * self.b0_std + self.b0_mean

            X = X_sh * X_b0
        else:
            if self.mean is not None and self.std is not None:
                X = X * self.std + self.mean

        X = torch.einsum("...i,ij", X, self.B.T).clamp(self.mini, self.maxi)
        dwi = dwi[..., ~self.where_b0]
        mse = metrics.weighted_mse(X, dwi, mask)
        return mse


class NegativefODFLoss(nn.Module):
    """Compute the norm of the negative fODF values on the sphere
    Only a subsample of the original 3D matrices is taken to fasten the
    computation and reduce memory consumption (the sphere has 362 directions)
    """

    def __init__(self, gtab, response, sh_order, lambda_=1, tau=0.1,
                 size=3, method='center'):
        """
        Attributes:
            gtab (dipy GradientTable): the gradient of the dwi in wich the
                                       response is represented.
            response (float[4]): first 3 elements: eigenvalues
                                 4th one: mean b0 value
            sh_order (int): the sh_order of the DWI used for the deconvolution
            lambda_ (float): the coefficient to rescale the loss
                             better to keep to 1. and change the 'coeff' param
                             when defining the loss
            tau (float): the coefficient to rescale the threshold used
            size (int): the size of the subsample taken to compute the loss
                        take size**3 voxels
            method (str): the method to take the subsample (random or center)
        """
        super(NegativefODFLoss, self).__init__()
        m, n = sph_harm_ind_list(sh_order)

        self.sphere = get_sphere('symmetric362')
        r, theta, phi = cart2sphere(
            self.sphere.x,
            self.sphere.y,
            self.sphere.z
        )
        # B_reg = real_sph_harm(m, n, theta[:, None], phi[:, None])
        B_reg = shm.get_B_matrix(theta=theta, phi=phi, sh_order=sh_order)

        R, r_rh, B_dwi = shm.get_deconv_matrix(gtab, response, sh_order)

        # scale lambda_ to account for differences in the number of
        # SH coefficients and number of mapped directions
        # This is exactly what is done in [4]_
        lambda_ = (lambda_ * R.shape[0] * r_rh[0] /
                   (np.sqrt(B_reg.shape[0]) * np.sqrt(362.)))
        B_reg = torch.FloatTensor(B_reg * lambda_)
        self.B_reg = nn.Parameter(B_reg, requires_grad=False)
        self.tau = tau

        self.size = size
        self.method = method

    def forward(self, fodf_sh, mask):
        px, py, pz = fodf_sh.shape[1:4]
        if self.method == 'center':
            s = self.size
            fodf_sh = fodf_sh[:,
                              px - s:px + s,
                              py - s:py + s,
                              pz - s:pz + s, :]
        else:
            number = self.size**3
            idx = np.array([(i, j, k) for i in range(px)
                            for j in range(py) for k in range(pz)])
            idx = tuple(zip(*idx[
                np.random.choice(range(px * py * pz), number, replace=False)]
            ))
            fodf_sh = fodf_sh[(slice(None),) + idx]
        fodf = torch.einsum("...i,ij", fodf_sh, self.B_reg.T)
        threshold = self.B_reg[0, 0] * fodf_sh[..., 0:1] * self.tau
        where_fodf_small = torch.nonzero(fodf < threshold, as_tuple=True)
        fodf = fodf[where_fodf_small]

        loss = torch.mean((fodf**2).sum(-1))

        return loss


class GFAMSELoss(nn.Module):
    """MSE of the gfa values computed on X and Z"""

    def __init__(self):
        super(GFAMSELoss, self).__init__()

    def forward(self, X, Z, mask):
        return metrics.torch_mse_gfa(X, Z, mask).mean()


class APMSELoss(nn.Module):
    """MSE of the anisotropic power values computed on X and Z"""

    def __init__(self):
        super(APMSELoss, self).__init__()

    def forward(self, X, Z, mask):
        return metrics.torch_mse_anisotropic_power(X, Z, mask).mean()


class RISMSELoss(nn.Module):
    """MSE of the Rotational Invariant Spherical features values
    computed on X and Z"""

    def __init__(self):
        super(RISMSELoss, self).__init__()

    def forward(self, X, Z, mask):
        return metrics.torch_mse_RIS(X, Z, mask).mean()


class onlyones_BCEWithLogitsLoss(nn.Module):
    """BCE with logits but only for values with label=1
    Used to ease the construction of the training scheme in settings.py"""

    def __init__(self, weight=None, pos_weight=None, reduction='mean'):
        super(onlyones_BCEWithLogitsLoss, self).__init__()
        self.weight = weight
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits):
        target = torch.ones(logits.shape, device=logits.device)
        return F.binary_cross_entropy_with_logits(
            logits, target,
            self.weight,
            pos_weight=self.pos_weight,
            reduction=self.reduction)


class onlyzeros_BCEWithLogitsLoss(nn.Module):
    """BCE with logits but only for values with label=0
    Used to ease the construction of the training scheme in settings.py"""

    def __init__(self, weight=None, pos_weight=None, reduction='mean'):
        super(onlyzeros_BCEWithLogitsLoss, self).__init__()
        self.weight = weight
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits):
        target = torch.zeros(logits.shape, device=logits.device)
        return F.binary_cross_entropy_with_logits(
            logits, target,
            self.weight,
            pos_weight=self.pos_weight,
            reduction=self.reduction)


class SmoothCrossEntropyLoss(nn.modules.loss._WeightedLoss):
    """Cross-entropy with label smoothing"""

    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                .fill_(smoothing / (n_classes - 1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(
            targets, inputs.size(-1), self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss


class MultiHingeLoss(nn.Module):
    def __init__(self, margin: float=1., p: int=1):
        super(MultiHingeLoss, self).__init__()
        self.margin = margin
        self.p = p

    def forward(self, logits, classes):
        mask_false = torch.ones_like(logits).bool()
        mask_false[..., classes] = False
        subset_idx = torch.argmax(logits[mask_false], dim=-1)
        max_false_idx = torch.arange(logits.shape[0], device=logits.device)
        max_false_idx = max_logits_fake[mask_false][subset_idx]

        loss = torch.max(
            0, self.margin - logits[..., classes] + logits[..., max_false_idx])
        loss = loss**p

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

    def forward(self, inputs, target):
        if inputs.dim() > 2:
            # N,H,W,D,C => N*H*W*D,C
            inputs = inputs.view(-1, inputs.size(-1))
        target = target.view(-1, 1)

        logpt = F.log_softmax(inputs)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class L2Reg(nn.Module):
    def __init__(self):
        super(L2Reg, self).__init__()

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        return (inputs**2).sum() / batch_size


class SmoothReg(nn.Module):
    def __init__(self):
        super(SmoothReg, self).__init__()

    def forward(self, y):
        dx = torch.sum(torch.abs(y[:, :-1, :, :, :] - y[:, 1:, :, :, :]))
        dy = torch.sum(torch.abs(y[:, :, :-1, :, :] - y[:, :, 1:, :, :]))
        dz = torch.sum(torch.abs(y[:, :, :, :-1, :] - y[:, :, :, 1:, :]))
        return (dx + dy + dz) / np.prod(y.shape[:-1])


def gram_matrix(input):
    shape = np.prod(list(input.size()))
    if len(input.size()) > 2:
        features = input.view(input.size()[0], input.size()[1], -1)
    else:
        features = torch.unsqueeze(input, -1)

    G = torch.matmul(features, features.transpose(1, 2))

    return G.div(shape)


class GramLoss(nn.Module):
    """MSE of Gramm Matrices of features vector"""

    def __init__(self, target_features):
        """
        Attributes:
            target_features (torch.tensor): A tensor of features to converge to
        """
        super(GramLoss, self).__init__()
        self.target_G = nn.Parameter(
            gram_matrix(torch.FloatTensor(target_features)),
            requires_grad=False)
        self.layer_coeff = layer_coeff if layer_coeff is not None else 1.

    def forward(self, inputs):
        G = gram_matrix(inputs)
        loss = F.mse_loss(G, self.target_G.expand_as(G))
        return loss.mean()


class FeatureLoss(nn.Module):
    """MSE of each feature vector"""

    def __init__(self, target_features):
        """
        Attributes:
            target_features (torch.tensor): A tensor of features to converge to
        """
        super(FeatureLoss, self).__init__()
        self.target_features = nn.Parameter(
            torch.FloatTensor(target_features).mean(0),
            requires_grad=False)

    def forward(self, inputs):
        loss = F.mse_loss(inputs, self.target_features.expand_as(inputs))
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
    """Return a dict with all the loss functions"""
    return {
        'acc': AccLoss,
        'mse': WeightedMSELoss,
        'mse_dwi': DWIMSELoss,
        'negative_fodf': NegativefODFLoss,
        'mse_RIS': RISMSELoss,
        'mse_ap': APMSELoss,
        'mse_gfa': GFAMSELoss,

        'gram': GramLoss,
        'feature': FeatureLoss,
        'hist': HistLoss,

        'bce': nn.BCELoss,
        'bce_logits': nn.BCEWithLogitsLoss,
        'bce_logits_ones': onlyones_BCEWithLogitsLoss,
        'bce_logits_zeros': onlyzeros_BCEWithLogitsLoss,

        'cross_entropy': SmoothCrossEntropyLoss,
        'multi_hinge': MultiHingeLoss,
        'multi_margin': nn.MultiMarginLoss,
        'focal': FocalLoss,

        'l2_reg': L2Reg,
        'smooth_reg': SmoothReg,
    }


def get_loss_fun(loss_specs, device):
    """Take a dictionnary of loss specs and a device et return a dict
    with a loss module and its paramters.

    Attributes:
        loss_specs (list[dict]): a list of all loss specifications:
        {"type" (float): the loss function name found in "get_metric_dict",
         "parameters" (dict): dict of parameters to initialize the module,
         "inputs" (list[str]): the list of inputs name IN ORDER for the forward
                               method of the module,}
        device (str or torch.device): the device of the module

    Returns:
        A list of dict:
        {"fun" (nn.Module): the loss module,
         "inputs" (list[str]): the list of inputs IN ORDER,
         "coeff" (float): the coefficient to multiply the loss with,
         "type" (float): the loss name}
    """
    loss_dict = get_loss_dict()
    losses = []
    for specs in loss_specs:
        d = dict()
        d['fun'] = loss_dict[specs["type"]](**specs["parameters"]).to(device)
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
        if 'detach_input' not in specs:
            d['detach_input'] = False
        else:
            d['detach_input'] = specs['detach_input']
        d['coeff'] = specs['coeff']
        d['type'] = specs['type']

        input_name = d['inputs'][0]['name']
        if d['inputs'][0]['from'] != "dataset":
            input_name += '_' + (d['inputs'][0]['from'])
        d['name'] = d['type'] + '_' + input_name
        losses.append(d)
    return losses
