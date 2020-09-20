import torch
import torch.nn as nn
import numpy as np

from dipy.reconst import dti
from dipy.core.gradients import gradient_table

from harmonisation.functions import shm

from torch_sym3eig import Sym3Eig


class APModule(nn.Module):
    def __init__(self, norm_factor=0.00001, power=2, non_negative=True):
        super(FAMoAPModuledule, self).__init__()
        self.norm_factor = norm_factor
        self.power = power
        self.non_negative = non_negative

        self.inputs = ['sh']

    def forward(sh_coeffs):
        dim = sh_coeffs.shape[:-1]
        n_coeffs = sh_coeffs.shape[-1]
        max_order = calculate_max_order(n_coeffs)
        ap = torch.zeros(dim, device=sh_coeffs.device)
        n_start = 1
        for L in range(2, max_order + 2, 2):
            n_stop = n_start + (2 * L + 1)
            ap_i = torch.mean(
                torch.abs(sh_coeffs[..., n_start:n_stop]) ** self.power, -1)
            ap += ap_i
            n_start = n_stop

        # Shift the map to be mostly non-negative,
        # only applying the log operation to positive elements
        # to avoid getting numpy warnings on log(0).
        # It is impossible to get ap values smaller than 0.
        # Also avoids getting voxels with -inf when non_negative=False.

        log_ap = torch.zeros_like(ap)
        log_ap[ap > 0] = torch.log(ap[ap > 0]) - torch.log(self.norm_factor)

        # Deal with residual negative values:
        if self.non_negative:
            log_ap[log_ap < 0] = 0
        return log_ap


class DWIModule(nn.Module):
    def __init__(self, gtab, sh_order,
                 mean=None, std=None,
                 b0_mean=None, b0_std=None):
        super(DWIModule, self).__init__()

        B, _ = shm.get_B_matrix(gtab, sh_order)
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

        self.inputs = ['sh', 'mean_b0']

    def add_b0(self, dwi, mean_b0):
        dwi = dwi * mean_b0

        b0_before, b0_after = 0, 0
        while self.gtab.b0s_mask[b0_before]:
            b0_before += 1
        while self.gtab.b0s_mask[-b0_after - 1]:
            b0_after += 1
        if b0_before > 0:
            dwi = torch.cat(
                [mean_b0.expand(*mean_b0.shape[:-1], b0_before), dwi],
                dim=-1)
        if b0_after > 0:
            dwi = torch.cat(
                [mean_b0.expand(*mean_b0.shape[:-1], b0_after), dwi],
                dim=-1)
        return dwi

    def forward(self, sh, mean_b0):
        if self.mean is not None and self.std is not None:
            sh = sh * self.std + self.mean
        if self.b0_mean is not None and self.b0_std is not None:
            mean_b0 = mean_b0 * self.b0_std + self.b0_mean

        self.B = self.B.to(sh.device)
        dwi = torch.einsum("...i,ij", sh, self.B.T).clamp(self.mini, self.maxi)
        dwi = self.add_b0(dwi, mean_b0)
        return dwi


class SymEig(nn.Module):
    def __init__(self, clip_value=100, eps=1e-6):
        super(SymEig, self).__init__()

        self.clip_value = clip_value
        self.eps = eps

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
            # eigenvals.register_hook(self.hook)
            # eigenvectors.register_hook(self.hook)
            pass

        eigenvals = eigenvals.reshape(*shape, 3)
        eigenvectors = eigenvectors.reshape(*shape, 3, 3)

        return eigenvals, eigenvectors


class EigenModule(nn.Module):
    def __init__(self, gtab):
        super(EigenModule, self).__init__()
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

        self.inputs = ['dwi']

    def forward(self, dwi):
        # self.design_matrix_inv = self.design_matrix_inv.to(dwi.device)

        dwi = dwi.clamp(min=self.min_diffusivity)

        fit_result = torch.einsum('ij,...j',
                                  self.design_matrix_inv,
                                  torch.log(dwi))

        fit_result = fit_result[..., self._lt_indices]

        eigenvals, eigenvectors = self.symeig_module(fit_result)

        eigenvals = eigenvals.clamp(min=self.min_diffusivity)
        eigenvals[torch.isnan(eigenvals)] = 0

        return eigenvals  # , eigenvectors


class FAModule(nn.Module):
    def __init__(self):
        super(FAModule, self).__init__()

        self.inputs = ['evals', 'mask']

    def forward(self, eigenvals, mask):
        all_zero = (eigenvals == 0).all(axis=-1)
        ev1, ev2, ev3 = eigenvals[..., 0], eigenvals[..., 1], eigenvals[..., 2]
        fa = torch.sqrt(
            0.5 * ((ev1 - ev2) ** 2 + (ev2 - ev3) ** 2 + (ev3 - ev1) ** 2) / (
                (eigenvals * eigenvals).sum(-1) + all_zero))

        fa = fa.unsqueeze(-1)

        return fa * mask


class MDModule(nn.Module):
    def __init__(self):
        super(MDModule, self).__init__()
        self.inputs = ['evals', 'mask']

    def forward(self, eigenvals, mask):
        md = eigenvals.mean(-1, keepdim=True)
        return md * mask


class ADModule(nn.Module):
    def __init__(self):
        super(ADModule, self).__init__()
        self.inputs = ['evals', 'mask']

    def forward(self, eigenvals, mask):
        return eigenvals[..., 0] * mask


class RDModule(nn.Module):
    def __init__(self):
        super(RDModule, self).__init__()
        self.inputs = ['evals', 'mask']

    def forward(self, eigenvals, mask):
        rd = eigenvals[..., 1:].mean(-1, keepdim=True)
        return rd * mask


class fODFModule(nn.Module):
    def __init__(self, gtab, response, sh_order,
                 lambda_=1, tau=0.1, convergence=50,
                 size=3, method='random'):
        super(fODFModule, self).__init__()

        self.sh_order = sh_order
        self.tau = tau
        self.convergence = convergence
        self.size = size
        self.method = method

        m, n = sph_harm_ind_list(sh_order)
        self._where_b0s = lazy_index(gtab.b0s_mask)
        self._where_dwi = lazy_index(~gtab.b0s_mask)

        # x, y, z = gtab.gradients[self._where_dwi].T
        # r, theta, phi = cart2sphere(x, y, z)
        # self.B_dwi = real_sph_harm(m, n, theta[:, None], phi[:, None])
        # self.B_dwi, _ = shm.get_B_matrix(gtab, sh_order)

        # S_r = estimate_response(gtab, response[0:3], response[3])
        # r_sh = np.linalg.lstsq(self.B_dwi, S_r[self._where_dwi], rcond=-1)[0]
        # n_response = n
        # m_response = m
        # r_rh = sh_to_rh(r_sh, m_response, n_response)
        # R = forward_sdeconv_mat(r_rh, n)
        R, r_rh, self.B_dwi = shm.get_deconv_matrix(gtab, response, sh_order)

        # for the sphere used in the regularization positivity constraint
        self.sphere = get_sphere('symmetric362')

        r, theta, phi = cart2sphere(
            self.sphere.x,
            self.sphere.y,
            self.sphere.z
        )
        # self.B_reg = real_sph_harm(m, n, theta[:, None], phi[:, None])
        B_reg = shm.get_B_matrix(theta=theta, phi=phi, sh_order=sh_order)

        # scale lambda_ to account for differences in the number of
        # SH coefficients and number of mapped directions
        # This is exactly what is done in [4]_
        lambda_ = (lambda_ * R.shape[0] * r_rh[0] /
                   (np.sqrt(self.B_reg.shape[0]) * np.sqrt(362.)))
        self.B_reg = self.B_reg * lambda_

        mu = 1e-5
        self.X = R.diagonal() * self.B_dwi
        self.P = torch.dot(X.T, X)
        self.P = self.P + mu * torch.eye(self.P.shape[0])

        self.inputs = ['dwi', 'mask']

    def _solve_cholesky(self, A, b):
        u = torch.cholesky(A)
        return torch.cholesky_solve(b, u)

    def csdeconv(self, dwi):
        """
        Deconvolves the axially symmetric single fiber response function `r_rh`
        in rotational harmonics coefficients from the diffusion weighted signal
        in `dwi`.
        """
        z = torch.einsum("...i,ij", dwi, self.X.T)

        fodf_sh = self._solve_cholesky(self.P, z)

        fodf = torch.einsum("...i,ij", fodf_sh[..., :15], self.B_reg[:, :15])
        threshold = self.B_reg[0, 0] * fodf_sh[0] * self.tau
        where_fodf_small = (fodf < threshold).nonzero()[..., -1]

        if len(where_fodf_small) == 0 and self.B_reg.shape[-1] > 15:
            fodf = torch.einsum("...i,ij", fodf_sh, self.B_reg.T)
            where_fodf_small = (fodf < threshold).nonzero()[..., -1]
            if len(where_fodf_small) == 0:
                return fodf_sh, 0

        for num_it in range(1, convergence + 1):
            H = self.B_reg.take(where_fodf_small, axis=0)

            Q = self.P + torch.dot(H.T, H)
            fodf_sh = self._solve_cholesky(Q, z)

            fodf = torch.bmm(self.B_reg, fodf_sh)
            where_fodf_small_last = where_fodf_small
            where_fodf_small = (fodf < threshold).nonzero()[0]

            if (len(where_fodf_small) == len(where_fodf_small_last) and
                    (where_fodf_small == where_fodf_small_last).all()):
                break
        else:
            msg = 'maximum number of iterations exceeded - failed to converge'
            warnings.warn(msg)

        return fodf_sh, num_it

    def forward(self, dwi, mask):
        px, py, pz = dwi.shape[1:4]
        if self.method == 'center':
            s = self.size
            dwi = dwi[:,
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
            dwi = dwi[(slice(None),) + idx]

        # Computing CSD fit
        dwi = dwi[self._where_dwi]
        fodf, _ = csdeconv(dwi)

        return fodf
