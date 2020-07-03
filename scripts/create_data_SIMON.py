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
import os
import tqdm


save_folder = "./.saved_models/style_test/"
net_file = '47_net.tar.gz'

SIGNAL_PARAMETERS['overlap_coeff'] = 2

paths, _ = get_paths_SIMON()  # get_paths_ADNI()

mean, std, b0_mean, b0_std = np.load(save_folder + 'mean_std.npy',
                                     allow_pickle=True)

# Load the network
net, _ = ENet.load(save_folder + net_file)
net.return_dict_layers = True
net = net.to("cuda")

FORCE = True

for dwi_name in tqdm.tqdm([path['name'] for path in paths]):
    dir_name = os.path.join('./data/CCNA/', dwi_name)

    if os.path.exists(dir_name) and not FORCE:
        continue

    dataset = None
    data = None
    sh_pred = None
    mask = None
    net_pred = None
    mean_b0_pred = None
    alpha = None
    beta = None
    dwi_true = None
    dwi_pred = None
    b0 = None
    mean_b0 = None
    temp = None

    dataset = SHDataset([path for path in paths if path['name'] == dwi_name],
                        patch_size=SIGNAL_PARAMETERS["patch_size"],
                        signal_parameters=SIGNAL_PARAMETERS,
                        transformations=None,
                        normalize_data=True,
                        mean=mean,
                        std=std,
                        b0_mean=b0_mean,
                        b0_std=b0_std,
                        n_jobs=8)

    data = dataset.get_data_by_name(dwi_name)

    # sh_true = batch_to_xyz(
    #     data['sh'],
    #     data['real_size'],
    #     empty=data['empty'],
    #     overlap_coeff=SIGNAL_PARAMETERS['overlap_coeff'])
    # sh_true = sh_true * dataset.std + dataset.mean

    net_pred = net.predict_dataset(dataset, batch_size=32)[dwi_name]
    sh_pred = net_pred['sh_pred']
    mean_b0_pred = net_pred['mean_b0_pred']
    alpha = net_pred['alpha']
    beta = net_pred['beta']

    sh_pred = batch_to_xyz(
        sh_pred,
        data['real_size'],
        empty=data['empty'],
        overlap_coeff=SIGNAL_PARAMETERS['overlap_coeff'],
        remove_border=2)
    sh_pred = sh_pred * dataset.std + dataset.mean

    mean_b0_pred = batch_to_xyz(
        mean_b0_pred,
        data['real_size'],
        empty=data['empty'],
        overlap_coeff=SIGNAL_PARAMETERS['overlap_coeff'],
        remove_border=2)
    mean_b0_pred = mean_b0_pred * dataset.b0_std + dataset.b0_mean

    alpha = batch_to_xyz(
        alpha,
        data['real_size'],
        empty=data['empty'],
        overlap_coeff=SIGNAL_PARAMETERS['overlap_coeff'],
        remove_border=2)

    beta = batch_to_xyz(
        beta,
        data['real_size'],
        empty=data['empty'],
        overlap_coeff=SIGNAL_PARAMETERS['overlap_coeff'],
        remove_border=2)

    mask = batch_to_xyz(
        data['mask'],
        data['real_size'],
        empty=data['empty'],
        overlap_coeff=SIGNAL_PARAMETERS['overlap_coeff'])

    dwi_pred = sh_to_dwi(sh_pred, data['gtab'], mask=None, add_b0=False)

    dwi_path = [x for x in paths if x['name'] == dwi_name][0]
    dwi_true, affine = load_nifti(dwi_path['dwi'])
    gtab = gradient_table(
        *read_bvals_bvecs(dwi_path["bval"], dwi_path["bvec"]))

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
    mask = pad(mask, pad_needed)

    dwi_pred *= mean_b0_pred
    temp = np.zeros_like(dwi_true)
    temp[..., ~gtab.b0s_mask] = dwi_pred
    temp[..., gtab.b0s_mask] = np.repeat(mean_b0_pred, b0.shape[-1], axis=-1)
    dwi_pred = temp

    dwi_pred *= mask
    alpha *= mask
    beta *= mask

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # save_nifti(os.path.join(dir_name, "sh_true.nii.gz"), sh_true, affine)
    save_nifti(os.path.join(dir_name, "sh_pred.nii.gz"), sh_pred, affine)
    save_nifti(os.path.join(dir_name, "dwi_pred.nii.gz"), dwi_pred, affine)
    save_nifti(os.path.join(dir_name, "alpha.nii.gz"), alpha, affine)
    save_nifti(os.path.join(dir_name, "beta.nii.gz"), beta, affine)
