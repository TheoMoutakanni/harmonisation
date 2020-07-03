from harmonisation.functions.shm import sh_to_dwi, normalize_data
from harmonisation.datasets import SHDataset
from harmonisation.datasets.utils import batch_to_xyz
from harmonisation.utils import get_paths_ADNI, get_paths_SIMON
from harmonisation.models import ENet

from harmonisation.settings import SIGNAL_PARAMETERS

from dipy.io.image import save_nifti, load_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table

import numpy as np


save_folder = "./.saved_models/style_test/"
dwi_file = 'HAM_DTI_2018'  # '003_S_4288_S142486'
net_file = '45_net.tar.gz'

SIGNAL_PARAMETERS['overlap_coeff'] = 2

paths, _ = get_paths_SIMON()  # get_paths_ADNI()

paths = [d for d in paths if d['name'] == dwi_file]

mean, std, b0_mean, b0_std = np.load(save_folder + 'mean_std.npy',
                                     allow_pickle=True)

# Create the dataset
dataset = SHDataset(paths,
                    patch_size=SIGNAL_PARAMETERS["patch_size"],
                    signal_parameters=SIGNAL_PARAMETERS,
                    transformations=None,
                    normalize_data=True,
                    mean=mean,
                    std=std,
                    b0_mean=b0_mean,
                    b0_std=b0_std,
                    n_jobs=8)

# Load the network
net, _ = ENet.load(save_folder + net_file)

net = net.to("cuda")
net.return_dict_layers = True
net.eval()

# Get the dmri name
dwi_name = dataset.names[0]
data = dataset.get_data_by_name(dwi_name)

sh_true = batch_to_xyz(
    data['sh'],
    data['real_size'],
    empty=data['empty'],
    overlap_coeff=SIGNAL_PARAMETERS['overlap_coeff'])
sh_true = sh_true * dataset.std + dataset.mean

net_pred = net.predict_dataset(dataset, batch_size=16)[dwi_name]
sh_pred = net_pred['sh_pred']
mean_b0_pred = net_pred['mean_b0_pred']
alpha = net_pred['alpha']
beta = net_pred['beta']

sh_pred = batch_to_xyz(
    sh_pred,
    data['real_size'],
    empty=data['empty'],
    overlap_coeff=SIGNAL_PARAMETERS['overlap_coeff'],
    remove_border=1)
sh_pred = sh_pred * dataset.std + dataset.mean

mean_b0_pred = batch_to_xyz(
    mean_b0_pred,
    data['real_size'],
    empty=data['empty'],
    overlap_coeff=SIGNAL_PARAMETERS['overlap_coeff'],
    remove_border=1)
mean_b0_pred = mean_b0_pred * dataset.b0_std + dataset.b0_mean

alpha = batch_to_xyz(
    alpha,
    data['real_size'],
    empty=data['empty'],
    overlap_coeff=SIGNAL_PARAMETERS['overlap_coeff'],
    remove_border=1)

beta = batch_to_xyz(
    beta,
    data['real_size'],
    empty=data['empty'],
    overlap_coeff=SIGNAL_PARAMETERS['overlap_coeff'],
    remove_border=1)

mask = batch_to_xyz(
    data['mask'],
    data['real_size'],
    empty=data['empty'],
    overlap_coeff=SIGNAL_PARAMETERS['overlap_coeff'])

dwi_pred = sh_to_dwi(sh_pred, data['gtab'], mask=None, add_b0=False)

dwi_true, affine = load_nifti(paths[0]['dwi'])
gtab = gradient_table(*read_bvals_bvecs(paths[0]["bval"], paths[0]["bvec"]))

b0 = dwi_true[..., gtab.b0s_mask]
mean_b0 = b0.mean(-1)

# dwi_true = normalize_data(dwi_true, gtab.b0s_mask)

patch_size = np.array(SIGNAL_PARAMETERS['patch_size'])
pad_needed = patch_size - dwi_true.shape[:3] % patch_size
pad_needed = [(x // 2, x // 2 + x % 2) for x in pad_needed]


def pad(x, pad_needed):
    return x[pad_needed[0][0]:-pad_needed[0][1],
             pad_needed[1][0]:-pad_needed[1][1],
             pad_needed[2][0]:-pad_needed[2][1]]


dwi_pred = pad(dwi_pred, pad_needed)
sh_pred = pad(sh_pred, pad_needed)
alpha = pad(alpha, pad_needed)
beta = pad(beta, pad_needed)
mean_b0_pred = pad(mean_b0_pred, pad_needed)
sh_true = pad(sh_true, pad_needed)
mask = pad(mask, pad_needed)

dwi_pred *= mean_b0_pred  # np.expand_dims(mean_b0, axis=-1)
temp = np.zeros_like(dwi_true)
temp[..., ~gtab.b0s_mask] = dwi_pred
temp[..., gtab.b0s_mask] = np.repeat(mean_b0_pred, b0.shape[-1], axis=-1)  # b0
dwi_pred = temp  # np.concatenate([b0, dwi_pred], axis=-1)

dwi_pred *= mask
alpha *= mask
beta *= mask

assert dwi_pred.shape == dwi_true.shape, (dwi_true.shape, dwi_pred.shape)

save_nifti("./data/mask.nii.gz", mask, affine)
save_nifti("./data/sh_true.nii.gz", sh_true, affine)
save_nifti("./data/dwi_true.nii.gz", dwi_true, affine)
save_nifti("./data/sh_pred.nii.gz", sh_pred, affine)
save_nifti("./data/dwi_pred.nii.gz", dwi_pred, affine)
save_nifti("./data/alpha.nii.gz", alpha, affine)
save_nifti("./data/beta.nii.gz", beta, affine)
