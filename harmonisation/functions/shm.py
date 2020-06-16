from dipy.reconst.shm import cart2sphere, smooth_pinv
from dipy.reconst.shm import real_sym_sh_basis, order_from_ncoef

from scilpy.reconst.raw_signal import compute_sh_coefficients

import numpy as np
import torch


def normalize_data(data, where_b0, min_signal=1e-5, out=None):
    """Normalizes the data with respect to the mean b0
    """
    if out is None:
        out = np.array(data, dtype='float32', copy=True)
    else:
        if out.dtype.kind != 'f':
            raise ValueError("out must be floating point")
        out[:] = data

    out.clip(min_signal, out=out)
    b0 = out[..., where_b0].mean(-1)
    out /= b0[..., None]
    out.clip(0, 1, out=out)
    return out


def get_B_matrix(gtab, sh_order, smooth=0.006):
    x, y, z = gtab.gradients[~gtab.b0s_mask].T
    r, theta, phi = cart2sphere(x, y, z)
    B, m, n = real_sym_sh_basis(sh_order, theta[:, None], phi[:, None])
    L = -n * (n + 1)
    invB = smooth_pinv(B, np.sqrt(smooth) * L)

    return B, invB


def dwi_to_sh(data_dwi, gtab,
              sh_order=4, mask=None, use_attenuation=True, smooth=0.006,
              * args, **kwargs):

    B, invB = get_B_matrix(gtab, sh_order, smooth=smooth)

    mini = .001
    maxi = .999

    if torch.is_tensor(data_dwi):
        invB = torch.FloatTensor(invB)
        data_dwi = data_dwi[..., ~gtab.b0s_mask].clamp(mini, maxi)
        data_sh = torch.einsum("...i,ij", data_dwi, invB.T)
    else:
        data_dwi = data_dwi[..., ~gtab.b0s_mask].clip(mini, maxi)
        data_sh = np.einsum("...i,ij", data_dwi, invB.T)

    if mask is not None:
        data_sh *= mask

    return data_sh


def sh_to_dwi(data_sh, gtab, mask=None, add_b0=True, smooth=0.006):
    sh_order = order_from_ncoef(data_sh.shape[-1])

    B, invB = get_B_matrix(gtab, sh_order, smooth=smooth)

    mini = .001
    maxi = .999

    if torch.is_tensor(data_sh):
        B = torch.FloatTensor(B).to(data_sh.device)
        data_dwi = torch.einsum("...i,ij", data_sh, B.T).clamp(mini, maxi)

        if add_b0:
            b0_like = torch.ones(*data_dwi.shape[:-1], gtab.b0s_mask.sum())
            b0_like = b0_like.to(data_dwi.device)
            data_dwi = torch.cat([b0_like, data_dwi], dim=-1)
    else:
        data_dwi = np.einsum("...i,ij", data_sh, B.T).clip(mini, maxi)

        if add_b0:
            shape = tuple(data_dwi.shape[:-1]) + (gtab.b0s_mask.sum(),)
            b0_like = np.ones(shape)
            data_dwi = np.concatenate([b0_like, data_dwi], axis=-1)

    if mask is not None:
        data_dwi *= mask

    return data_dwi
