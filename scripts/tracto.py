from harmonisation.functions.shm import sh_to_dwi
from harmonisation.functions import metrics

from dipy.io.image import save_nifti, load_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table

import matplotlib.pyplot as plt

import numpy as np
import torch

mask = np.load("./data/mask.npy")
sh_true, affine = load_nifti("./data/sh_true.nii.gz")
dwi_true, affine = load_nifti("./data/dwi_true.nii.gz")
sh_pred, affine = load_nifti("./data/sh_pred.nii.gz")
dwi_pred, affine = load_nifti("./data/dwi_pred.nii.gz")

gtab = gradient_table(*read_bvals_bvecs('./data/bval', './data/bvec'))

sh_true = torch.FloatTensor(sh_true)
dwi_true = torch.FloatTensor(dwi_true)
sh_pred = torch.FloatTensor(sh_pred)
dwi_pred = torch.FloatTensor(dwi_pred)

gfa_true = metrics.torch_gfa(sh_true)
gfa_pred = metrics.torch_gfa(sh_pred)

plt.subplot(1, 2, 1).set_axis_off()
plt.imshow((gfa_true[:, :, 90]).T, cmap='gray', origin='lower')

from dipy.reconst.csdeconv import auto_response
from dipy.reconst.shm import CsaOdfModel
from dipy.data import default_sphere
from dipy.direction import peaks_from_model

response, ratio = auto_response(gtab, dwi_true.numpy(), roi_radius=10, fa_thr=0.7)
csa_model = CsaOdfModel(gtab, sh_order=6)
csa_peaks = peaks_from_model(csa_model, dwi_true.numpy(), default_sphere,
                             relative_peak_threshold=.8,
                             min_separation_angle=45,
                             mask=mask.squeeze())

plt.subplot(1, 2, 2).set_axis_off()
plt.imshow(csa_peaks.gfa[:, :, 90].T, cmap='gray', origin='lower')
plt.show()
