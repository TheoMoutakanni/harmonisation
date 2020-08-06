from dipy.viz import window, actor, has_fury
from dipy.data import default_sphere
from dipy.reconst.shm import sph_harm_lookup, order_from_ncoef

import matplotlib.pyplot as plt

import numpy as np
import torch

from harmonisation.functions.metrics import get_metric_dict
from harmonisation.settings import SIGNAL_PARAMETERS


def print_peaks(sh_signal, mask=None):
    if has_fury:
        data_small = sh_signal[:, :, 50:51]
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
    metric = get_metric_dict()[metric_name](sh_true,
                                            sh_pred,
                                            mask)

    if normalize:
        m = metric[~torch.isnan(metric)].min()
        M = metric[~torch.isnan(metric)].max()
        metric = (metric - m) / (M - m)

    fig, axarr = plt.subplots(1, 3)
    im1 = axarr[0].imshow(np.squeeze(metric[:, :, 50:51]), cmap='hot_r',
                          interpolation='nearest')
    fig.colorbar(im1, ax=axarr[0])

    im2 = axarr[1].imshow(np.squeeze(metric[:, 80:81, :]), cmap='hot_r',
                          interpolation='nearest')
    fig.colorbar(im2, ax=axarr[1])

    im3 = axarr[2].imshow(np.squeeze(metric[80:81, :, :]), cmap='hot_r',
                          interpolation='nearest')
    fig.colorbar(im3, ax=axarr[2])

    if fig_name is not None:
        plt.savefig('./{}.png'.format(fig_name))
    plt.suptitle(metric_name)
    plt.show()


def print_data(sh_true, sh_pred, mask, fig_name=None):
    mask = mask[:, :, :, 0]

    R_true = sh_true * mask
    R_pred = sh_pred * mask
    M = R_true.max()
    m = R_true.min()

    fig, axarr = plt.subplots(2, 3)
    axarr[0][0].imshow(np.squeeze(R_true[:, :, 50:51]), vmin=m, vmax=M)
    axarr[0][0].set_title('True value')
    axarr[1][0].imshow(np.squeeze(R_pred[:, :, 50:51]), vmin=m, vmax=M)
    axarr[1][0].set_title('Predicted value')

    axarr[0][1].imshow(np.squeeze(R_true[:, 80:81, :]), vmin=m, vmax=M)
    axarr[0][1].set_title('True value')
    axarr[1][1].imshow(np.squeeze(R_pred[:, 80:81, :]), vmin=m, vmax=M)
    axarr[1][1].set_title('Predicted value')

    axarr[0][2].imshow(np.squeeze(R_true[80:81, :, :]), vmin=m, vmax=M)
    axarr[0][2].set_title('True value')
    axarr[1][2].imshow(np.squeeze(R_pred[80:81, :, :]), vmin=m, vmax=M)
    axarr[1][2].set_title('Predicted value')

    if fig_name is not None:
        plt.suptitle(fig_name)
        plt.savefig('./{}.png'.format(fig_name))
    plt.show()


def print_RIS(RIS_true, RIS_pred, mask, fig_name=None):

    mask = mask[:, :, 50:51, 0]
    colors_true = []
    colors_pred = []
    for order in range(3):
        R_true = RIS_true[..., 50:51, order]
        R_pred = RIS_pred[..., 50:51, order]
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
