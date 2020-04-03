from dipy.viz import window, actor, has_fury
from dipy.data import default_sphere
from dipy.reconst.shm import sph_harm_lookup, order_from_ncoef

import matplotlib.pyplot as plt

import numpy as np
import torch

from harmonisation.functions.metrics import get_metrics_fun
from harmonisation.settings import SIGNAL_PARAMETERS


def print_peaks(sh_signal, mask=None):
    if has_fury:
        data_small = sh_signal[:, :, 28:29]
        ren = window.Renderer()

        sh_order = order_from_ncoef(data_small.shape[-1])
        theta = default_sphere.theta
        phi = default_sphere.phi
        sh_params = SIGNAL_PARAMETERS['processing_params']['sh_params']
        basis_type = sh_params['basis_type']
        sph_harm_basis = sph_harm_lookup.get(basis_type)
        sampling_matrix, m, n = sph_harm_basis(sh_order, theta, phi)
        odfs = np.dot(data_small, sampling_matrix.T)

        odfs = np.clip(odfs, 0, np.max(odfs, -1)[..., None])
        odfs_actor = actor.odf_slicer(odfs, sphere=default_sphere,
                                      colormap='plasma', scale=0.4)
        odfs_actor.display(z=0)

        ren.add(odfs_actor)
        print('Saving illustration as csa_odfs.png')
        window.record(ren, n_frames=1,
                      out_path='csa_odfs.png', size=(600, 600))
        window.show(ren)


def print_diff(sh_true, sh_pred, mask, metric_name,
               normalize=False, fig_name=None):
    metric = get_metrics_fun()[metric_name](sh_true[:, :, 28:29],
                                            sh_pred[:, :, 28:29],
                                            mask[:, :, 28:29])

    if normalize:
        m = metric[~torch.isnan(metric)].min()
        M = metric[~torch.isnan(metric)].max()
        metric = (metric - m) / (M - m)

    im = plt.imshow(np.squeeze(metric), cmap='hot_r', interpolation='nearest')
    plt.colorbar(im)
    if fig_name is not None:
        plt.savefig('./{}.png'.format(fig_name))
    plt.suptitle(metric_name)
    plt.show()


def print_data(sh_true, sh_pred, mask, fig_name=None):
    mask = mask[:, :, 28:29, 0]

    R_true = np.squeeze(sh_true[..., 28:29] * mask)
    R_pred = np.squeeze(sh_pred[..., 28:29] * mask)
    M = R_true.max()
    m = R_true.min()

    print(R_true.shape)

    fig, axarr = plt.subplots(2, 1)
    axarr[0].imshow(R_true, vmin=m, vmax=M)
    axarr[0].set_title('True value')
    axarr[1].imshow(R_pred, vmin=m, vmax=M)
    axarr[1].set_title('Predicted value')
    if fig_name is not None:
        plt.suptitle(fig_name)
        plt.savefig('./{}.png'.format(fig_name))
    plt.show()


def print_RIS(RIS_true, RIS_pred, mask, fig_name=None):

    mask = mask[:, :, 28:29, 0]
    colors_true = []
    colors_pred = []
    for order in range(3):
        R_true = RIS_true[..., 28:29, order]
        R_pred = RIS_pred[..., 28:29, order]
        R_max = R_true[mask.bool()].max()
        R_min = R_true[mask.bool()].min()
        R_true = (R_true - R_min) / (R_max - R_min) * mask
        R_pred = (R_pred - R_min) / (R_max - R_min) * mask
        R_pred = R_pred.clamp(0, 1)

        colors_true.append(R_true.squeeze().cpu().numpy())
        colors_pred.append(R_pred.squeeze().cpu().numpy())

    fig, axarr = plt.subplots(2, 3)
    for i in range(3):
        M = colors_true[i].max()
        m = colors_true[i].min()
        axarr[0][i].imshow(colors_true[i], vmin=m, vmax=M)
        axarr[0][i].set_title('True RIS {}'.format(2 * i))
        axarr[1][i].imshow(colors_pred[i], vmin=m, vmax=M)
        axarr[1][i].set_title('Predicted RIS {}'.format(2 * i))

    if fig_name is not None:
        plt.savefig('./{}.png'.format(fig_name))
    plt.suptitle('RIS features')
    plt.show()
