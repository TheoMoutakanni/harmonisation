from harmonisation.functions import metrics

from dipy.io.image import load_nifti

import torch
import numpy as np

path = "./data/"
mean, std = np.load("./.saved_models/style_fa/" + 'mean_std.npy')

mask, affine = load_nifti(path + "mask.nii.gz")
sh_true, _ = load_nifti(path + "sh_true.nii.gz")
sh_pred, _ = load_nifti(path + "sh_pred.nii.gz")

sh_true = torch.FloatTensor((sh_true - mean) / std)
sh_pred = torch.FloatTensor((sh_pred - mean) / std)

acc = metrics.torch_angular_corr_coeff(sh_true, sh_pred)
acc = (acc.numpy() * mask.squeeze()).sum() / mask.sum()

print("acc: ", acc)

mse_sh = metrics.weighted_mse_loss(
    sh_true, sh_pred, torch.FloatTensor(mask)).sum() / mask.sum()

print("mse sh: ", mse_sh)

dwi_true, _ = load_nifti(path + "dwi_true.nii.gz")
dwi_pred, _ = load_nifti(path + "dwi_pred.nii.gz")

dwi_true = torch.FloatTensor(dwi_true)
dwi_pred = torch.FloatTensor(dwi_pred)

mse = metrics.weighted_mse_loss(
    dwi_true, dwi_pred, torch.FloatTensor(mask)).sum() / mask.sum()

print("mse: ", mse)
