from dipy.reconst.shm import cart2sphere, smooth_pinv
from dipy.reconst.shm import real_sym_sh_basis, order_from_ncoef

from scilpy.reconst.raw_signal import compute_sh_coefficients

import numpy as np


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


def dwi_to_sh(data_dwi, gtab, mask=None, use_attenuation=True,
              *args, **kwargs):
    # x, y, z = gtab.gradients[~gtab.b0s_mask].T
    # r, theta, phi = cart2sphere(x, y, z)
    # B, m, n = real_sym_sh_basis(sh_order, theta[:, None], phi[:, None])
    # L = -n * (n + 1)
    # invB = smooth_pinv(B, np.sqrt(smooth) * L)

    # mini = .001
    # maxi = .999
    # data_dwi = data_dwi[..., ~gtab.b0s_mask].clip(mini, maxi)
    # data_sh = np.dot(data_dwi, invB.T)

    # if mask is not None:
    #     mask = np.asarray(mask, dtype=bool)
    #     data_sh *= mask[..., None]

    # return data_sh

    data_sh = compute_sh_coefficients(data_dwi, gtab,
                                      mask=mask,
                                      use_attenuation=use_attenuation,
                                      * args, **kwargs)

    return data_sh


def sh_to_dwi(data_sh, gtab, mask=None):
    sh_order = order_from_ncoef(data_sh.shape[-1])
    x, y, z = gtab.gradients[~gtab.b0s_mask].T
    r, theta, phi = cart2sphere(x, y, z)
    B, m, n = real_sym_sh_basis(sh_order, theta[:, None], phi[:, None])

    mini = .001
    maxi = .999
    data_dwi = np.dot(data_sh, B.T).clip(mini, maxi)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool).squeeze()
        data_dwi *= mask[..., None]

    return data_dwi
