import torch
import torch.nn as nn
import numpy as np

from dipy.reconst import dti
from dipy.core.gradients import gradient_table

from harmonisation.functions.shm import get_B_matrix

from torch_sym3eig import Sym3Eig


class DWIModule(nn.Module):
    def __init__(self, gtab, sh_order, mean=None, std=None):
        super(DWIModule, self).__init__()

        B, _ = get_B_matrix(gtab, sh_order)
        bvals, bvecs = gtab.bvals, gtab.bvecs
        bvals, bvecs = bvals[bvals != 0], bvecs[bvals != 0]
        bvals = np.concatenate(([0], bvals), axis=0)
        bvecs = np.concatenate(([[0, 0, 0]], bvecs), axis=0)
        self.gtab = gradient_table(bvals, bvecs)
        self.B = torch.FloatTensor(B)
        self.mini = .001
        self.maxi = .999

        self.mean = torch.FloatTensor(mean) if mean is not None else None
        self.std = torch.FloatTensor(std) if std is not None else None

    def add_b0(self, dwi, mean_b0):
        # b0_before, b0_after = 0, 0
        # while self.gtab.b0s_mask[b0_before]:
        #     b0_before += 1
        # while self.gtab.b0s_mask[-b0_after - 1]:
        #     b0_after += 1
        # if b0_before > 0:
        #     b0_before = torch.ones(*dwi.shape[:-1], b0_before)
        #     b0_before = b0_before.to(dwi.device)
        #     dwi = torch.cat([b0_before, dwi], dim=-1)
        # if b0_after > 0:
        #     b0_after = torch.ones(*dwi.shape[:-1], b0_after)
        #     b0_after = b0_after.to(dwi.device)
        #     dwi = torch.cat([dwi, b0_after], dim=-1)
        dwi *= mean_b0
        dwi = torch.cat([mean_b0, dwi], dim=0)

        return dwi

    def forward(self, X, mean_b0):
        if self.mean is not None and self.std is not None:
            self.mean = self.mean.to(X.device)
            self.std = self.std.to(X.device)
            X = X * self.std + self.mean

        self.B = self.B.to(X.device)
        dwi = torch.einsum("...i,ij", X, self.B.T).clamp(self.mini, self.maxi)
        dwi = self.add_b0(dwi, mean_b0)
        return dwi


class FAModule(nn.Module):
    def __init__(self, gtab=None, design_matrix=None, design_matrix_inv=None):
        super(FAModule, self).__init__()
        tol = 1e-6

        assert (gtab is not None) or (
            design_matrix is not None), "must give gtab or design_matrix"

        if design_matrix is None:
            design_matrix = dti.design_matrix(gtab)

        if design_matrix_inv is None:
            design_matrix_inv = torch.FloatTensor(
                np.linalg.pinv(design_matrix))
            design_matrix_inv = design_matrix_inv

        self.min_diffusivity = tol / -design_matrix.min()
        self.design_matrix_inv = design_matrix_inv

        # self.symeig_module = SymEig()

    def forward(self, data):
        self.design_matrix_inv = self.design_matrix_inv.to(data.device)

        data = data.clamp(min=self.min_diffusivity)

        fit_result = torch.einsum('ij,...j',
                                  self.design_matrix_inv,
                                  torch.log(data))

        _lt_indices = np.array([[0, 1, 3],
                                [1, 2, 4],
                                [3, 4, 5]])

        # eigenvals, _ = torch.symeig(fit_result[..., _lt_indices],
        #                             eigenvectors=True,
        #                             upper=True)

        fit_result = fit_result[..., _lt_indices]

        eigenvals, eigenvectors = Sym3Eig.apply(fit_result)
        # eigenvals, eigenvectors = self.symeig_module(fit_result)

        # print(torch.isnan(fit_result).any())
        # print((torch.isnan(eigenvals).float().sum(-1).sum(-1).sum(-1).sum(-1) * 100 / (32 * 32 * 32 * 3)).int())
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

        return fa


class SymEig(nn.Module):
    def __init__(self, clip_value=1000, eps=1e-6):
        super(SymEig, self).__init__()

        self.clip_value = clip_value
        self.eps = eps
        self.register_backward_hook(self.hook)

    def hook(self, module, grad_input, grad_output):
        return torch.clamp(grad_output, -self.clip_value, self.clip_value)

    def forward(self, x):
        noise = x = torch.cuda.FloatTensor(x.shape)
        torch.randn(x.shape, out=noise)
        noise *= self.eps

        x = x + noise
        x = (x + torch.transpose(x, -1, -2)) / 2.0

        eigenvals, eigenvectors = Sym3Eig.apply(x)

        return eigenvals, eigenvectors
