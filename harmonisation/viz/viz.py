from dipy.viz import window, actor, has_fury
from dipy.data import default_sphere
from dipy.reconst.shm import CsaOdfModel
from dipy.core.gradients import GradientTable

import matplotlib.pyplot as plt

import numpy as np
import torch

from harmonisation.functions.shm import sh_to_dwi
from harmonisation.functions.metrics import torch_angular_corr_coeff


def print_peaks(dmri_signal, b0_signal, gtab, mask=None):
    gtab = GradientTable(np.concatenate(
        ([[0, 0, 0]], gtab.gradients[~gtab.b0s_mask]), axis=0),
        gtab.big_delta,
        gtab.small_delta,
        gtab.b0_threshold)
    dmri_signal = sh_to_dwi(dmri_signal, gtab)

    dmri_signal = np.concatenate((b0_signal, dmri_signal), axis=-1)

    csa_model = CsaOdfModel(gtab, 4)

    if has_fury:
        data_small = dmri_signal[:, :, 28:29]
        ren = window.Renderer()
        csa_odfs = csa_model.fit(data_small).odf(default_sphere)

        csa_odfs = np.clip(csa_odfs, 0, np.max(csa_odfs, -1)[..., None])
        csa_odfs_actor = actor.odf_slicer(csa_odfs, sphere=default_sphere,
                                          colormap='plasma', scale=0.4)
        csa_odfs_actor.display(z=0)

        ren.add(csa_odfs_actor)
        print('Saving illustration as csa_odfs.png')
        window.record(ren, n_frames=1,
                      out_path='csa_odfs.png', size=(600, 600))
        window.show(ren)


def print_acc(dmri_true, dmri_pred, normalize=True):
    acc = torch_angular_corr_coeff(dmri_true[:, :, 28:29],
                                   dmri_pred[:, :, 28:29])

    if normalize:
        m = acc[~torch.isnan(acc)].min()
        M = acc[~torch.isnan(acc)].max()
        acc = (acc - m) / (M - m)

    im = plt.imshow(np.squeeze(acc), cmap='hot_r', interpolation='nearest')
    plt.colorbar(im)
    plt.savefig('./acc.png')
    plt.show()


def print_RIS(RIS_true, RIS_pred, mask):

    mask = mask[:, :, 28:29, 0]
    colors_true = []
    colors_pred = []
    for order in range(3):
        R_true = RIS_true[..., 28:29, order]
        R_pred = RIS_pred[..., 28:29, order]
        R_max = torch.max(R_true[mask.bool()], R_pred[mask.bool()]).max()
        R_min = torch.min(R_true, R_pred).min()
        R_true = (R_true - R_min) / (R_max - R_min) * mask
        R_pred = (R_pred - R_min) / (R_max - R_min) * mask

        colors_true.append(R_true.squeeze().cpu().numpy())
        colors_pred.append(R_pred.squeeze().cpu().numpy())

    colors_true = np.transpose(colors_true, (1, 2, 0))
    colors_pred = np.transpose(colors_pred, (1, 2, 0))

    fig, axarr = plt.subplots(2, 1)
    axarr[0].imshow(colors_true)
    axarr[1].imshow(colors_pred)
    plt.show()
