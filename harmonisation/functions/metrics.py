import numpy as np
import torch

from dipy.reconst import shm, dti


def nanmean(v, inplace=False, *args, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


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
    predicted = torch.argmax(proba, dim=1)
    accuracy = (predicted == labels).float().mean()
    print(predicted)
    print(labels)
    return accuracy


def torch_RIS(X):
    sh_order = shm.order_from_ncoef(X.shape[-1])
    m, n = shm.sph_harm_ind_list(sh_order)
    RIS = []
    for i in np.arange(sh_order // 2 + 1) * 2:
        RIS.append(X[..., n == i].sum(-1))
    return torch.stack(RIS).permute(*range(1, len(X.shape)), 0)


def torch_mse_RIS(X, Z, weight):
    X_RIS = torch_RIS(X)
    Z_RIS = torch_RIS(Z)
    mse_RIS = weighted_mse_loss(X_gra, Z_gfa, weight)
    return mse_RIS


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
    return mse_gfa


def torch_anisotropic_power(sh_coeffs, norm_factor=0.00001, power=2,
                            non_negative=True):
    """Calculates anisotropic power map with a given SH coefficient matrix
    """

    dim = sh_coeffs.shape[:-1]
    n_coeffs = sh_coeffs.shape[-1]
    max_order = shm.calculate_max_order(n_coeffs)
    ap = torch.zeros(dim)
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
    log_ap = torch.zeros_like(ap)
    log_ap[ap > 0] = torch.log(ap[ap > 0]) - torch.log(norm_factor)

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


def ols_fit_tensor(data, gtab=None, design_matrix=None,
                   design_matrix_inv=None):
    r"""
    Computes ordinary least squares (OLS) fit to calculate self-diffusion
    tensor using a linear regression model [1]_.
    Parameters
    ----------
    design_matrix : array (g, 7)
        Design matrix holding the covariants used to solve for the regression
        coefficients.
    data : array ([X, Y, Z, ...], g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data. It makes no copies of data.
    return_S0_hat : bool
        Boolean to return (True) or not (False) the S0 values for the fit.
    Returns
    -------
    eigvals : array (..., 3)
        Eigenvalues from eigen decomposition of the tensor.
    eigvecs : array (..., 3, 3)
        Associated eigenvectors from eigen decomposition of the tensor.
        Eigenvectors are columnar (e.g. eigvecs[:,j] is associated with
        eigvals[j])
    """

    assert (gtab is not None) or (
        design_matrix is not None), "must give gtab or design_matrix"

    if design_matrix is None:
        design_matrix = dti.design_matrix(gtab)
        design_matrix = torch.FloatTensor(design_matrix)

    if design_matrix_inv is None:
        design_matrix_inv = torch.FloatTensor(np.linalg.pinv(design_matrix))
        design_matrix_inv = design_matrix_inv

    design_matrix = design_matrix.to(data.device)
    design_matrix_inv = design_matrix_inv.to(data.device)

    tol = 1e-6
    fit_result = torch.einsum('ij,...j',
                              design_matrix_inv,
                              torch.log(data))

    min_diffusivity = tol / -design_matrix.min()

    _lt_indices = np.array([[0, 1, 3],
                            [1, 2, 4],
                            [3, 4, 5]])
    eigenvals, eigenvecs = torch.symeig(fit_result[..., _lt_indices],
                                        eigenvectors=True,
                                        upper=True)

    # need to sort the eigenvalues and associated eigenvectors
    if eigenvals.ndim == 1:
        # this is a lot faster when dealing with a single voxel
        order = torch.flip(eigenvals.argsort(), dims=(0,))  # [::-1]
        eigenvecs = eigenvecs[:, order]
        eigenvals = eigenvals[order]
    else:
        # temporarily flatten eigenvals and eigenvecs to make sorting easier
        shape = eigenvals.shape[:-1]
        eigenvals = eigenvals.reshape(-1, 3)
        eigenvecs = eigenvecs.reshape(-1, 3, 3)
        size = eigenvals.shape[0]
        order = torch.flip(eigenvals.argsort(), dims=(1,))  # [:, ::-1]
        xi, yi = np.ogrid[:size, :3, :3][:2]
        eigenvecs = eigenvecs[xi, yi, order[:, None, :]]
        xi = np.ogrid[:size, :3][0]
        eigenvals = eigenvals[xi, order]
        eigenvecs = eigenvecs.reshape(shape + (3, 3))
        eigenvals = eigenvals.reshape(shape + (3, ))
    eigenvals = eigenvals.clamp(min=min_diffusivity)
    # eigenvecs: each vector is columnar

    dti_params = torch.cat((eigenvals[..., None, :], eigenvecs), dim=-2)
    eigs = dti_params.reshape(data.shape[:-1] + (12, ))

    return eigs


def torch_fa(evals):
    all_zero = (evals == 0).all(axis=-1)
    ev1, ev2, ev3 = evals[..., 0], evals[..., 1], evals[..., 2]
    fa = torch.sqrt(0.5 * ((ev1 - ev2) ** 2 +
                           (ev2 - ev3) ** 2 +
                           (ev3 - ev1) ** 2) /
                    ((evals * evals).sum(-1) + all_zero))

    return fa


def get_metrics_fun():
    """Return a dict with all the metrics to be computed during an epoch
    Metrics must be greater when better, so -mse instead of mse for example"""
    return {
        'acc': lambda x, z, mask: torch_angular_corr_coeff(x * mask, z * mask),
        'mse': lambda x, z, mask: -weighted_mse_loss(x, z, mask),
        'mse_RIS': lambda x, z, mask: -torch_mse_RIS(x, z, mask),
        'mse_gfa': lambda x, z, mask: -torch_mse_gfa(x, z, mask),
        'accuracy': lambda labels, proba: torch_accuracy(labels, proba)
    }
