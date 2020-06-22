import torch
import torch.nn as nn
import numpy as np

from dipy.reconst import dti
from dipy.core.gradients import gradient_table

from harmonisation.functions.shm import get_B_matrix

from torch_sym3eig import Sym3Eig


class DWIModule(nn.Module):
    def __init__(self, gtab, sh_order,
                 mean=None, std=None,
                 b0_mean=None, b0_std=None):
        super(DWIModule, self).__init__()

        B, _ = get_B_matrix(gtab, sh_order)
        self.gtab = gtab
        self.B = torch.FloatTensor(B)
        self.mini = .001
        self.maxi = .999
        if mean is not None:
            self.mean = torch.FloatTensor(mean)
            self.mean = nn.Parameter(self.mean, requires_grad=False)
        else:
            self.mean = None
        if std is not None:
            self.std = torch.FloatTensor(std)
            self.std = nn.Parameter(self.std, requires_grad=False)
        else:
            self.std = None
        if b0_mean is not None:
            self.b0_mean = torch.FloatTensor(b0_mean)
            self.b0_mean = nn.Parameter(self.b0_mean, requires_grad=False)
        else:
            self.b0_mean = None
        if b0_std is not None:
            self.b0_std = torch.FloatTensor(b0_std)
            self.b0_std = nn.Parameter(self.b0_std, requires_grad=False)
        else:
            self.b0_std = None

    def add_b0(self, dwi, mean_b0):
        b0_before, b0_after = 0, 0
        while self.gtab.b0s_mask[b0_before]:
            b0_before += 1
        while self.gtab.b0s_mask[-b0_after - 1]:
            b0_after += 1
        if b0_before > 0:
            b0_before = torch.ones(*dwi.shape[:-1], b0_before)
            b0_before = b0_before.to(dwi.device)
            dwi = torch.cat([b0_before, dwi], dim=-1)
        if b0_after > 0:
            b0_after = torch.ones(*dwi.shape[:-1], b0_after)
            b0_after = b0_after.to(dwi.device)
            dwi = torch.cat([dwi, b0_after], dim=-1)

        dwi = dwi * mean_b0

        return dwi

    def forward(self, X, mean_b0):
        if self.mean is not None and self.std is not None:
            X = X * self.std + self.mean
        if self.b0_mean is not None and self.b0_std is not None:
            mean_b0 = mean_b0 * self.b0_std + self.b0_mean

        self.B = self.B.to(X.device)
        dwi = torch.einsum("...i,ij", X, self.B.T).clamp(self.mini, self.maxi)
        dwi = self.add_b0(dwi, mean_b0)
        return dwi


class FAModule(nn.Module):
    def __init__(self, gtab):
        super(FAModule, self).__init__()
        tol = 1e-6

        design_matrix = dti.design_matrix(gtab)

        self.design_matrix_inv = torch.FloatTensor(
            np.linalg.pinv(design_matrix))
        self.design_matrix_inv = nn.Parameter(self.design_matrix_inv,
                                              requires_grad=False)

        self.min_diffusivity = tol / -design_matrix.min()
        self._lt_indices = np.array([[0, 1, 3],
                                     [1, 2, 4],
                                     [3, 4, 5]])

        self.symeig_module = SymEig()

    def forward(self, dwi, mask):
        # self.design_matrix_inv = self.design_matrix_inv.to(dwi.device)

        dwi = dwi.clamp(min=self.min_diffusivity)

        fit_result = torch.einsum('ij,...j',
                                  self.design_matrix_inv,
                                  torch.log(dwi))

        fit_result = fit_result[..., self._lt_indices]

        # eigenvals, eigenvectors = Sym3Eig.apply(fit_result)
        eigenvals, eigenvectors = self.symeig_module(fit_result)

        # shape = eigenvals.shape[:-1]
        # eigenvals = eigenvals.reshape(-1, 3)
        # size = eigenvals.shape[0]
        # order = torch.flip(eigenvals.argsort(), dims=(1,))  # [:, ::-1]
        # xi = np.ogrid[:size, :3][0]
        # eigenvals = eigenvals[xi, order]
        # eigenvals = eigenvals.reshape(shape + (3, ))
        eigenvals = eigenvals.clamp(min=self.min_diffusivity)
        eigenvals[torch.isnan(eigenvals)] = 0

        all_zero = (eigenvals == 0).all(axis=-1)
        ev1, ev2, ev3 = eigenvals[..., 0], eigenvals[..., 1], eigenvals[..., 2]
        fa = torch.sqrt(
            0.5 * ((ev1 - ev2) ** 2 + (ev2 - ev3) ** 2 + (ev3 - ev1) ** 2) / (
                (eigenvals * eigenvals).sum(-1) + all_zero))

        fa = fa.unsqueeze(-1)
        fa = fa * mask

        return fa


class SymEig(nn.Module):
    def __init__(self, clip_value=100, eps=1e-6):
        super(SymEig, self).__init__()

        self.clip_value = clip_value
        self.eps = eps
        # self.register_backward_hook(self.hook)

    def hook(self, grad):
        return torch.clamp(grad, -self.clip_value, self.clip_value)

    def forward(self, x):
        shape = x.shape[:-2]
        x = x.reshape(-1, 3, 3)
        if x.is_cuda:
            noise = torch.cuda.FloatTensor(x.shape)
        else:
            noise = torch.FloatTensor(x.shape)
        torch.randn(x.shape, out=noise)
        noise *= self.eps

        x = x + noise
        x = (x + torch.transpose(x, -1, -2)) / 2.0

        eigenvals, eigenvectors = Sym3Eig.apply(x)
        if x.requires_grad:
            eigenvals.register_hook(self.hook)
            # eigenvectors.register_hook(self.hook)

        eigenvals = eigenvals.reshape(*shape, 3)
        eigenvectors = eigenvectors.reshape(*shape, 3, 3)

        return eigenvals, eigenvectors
