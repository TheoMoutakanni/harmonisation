from harmonisation.functions.shm import sh_to_dwi
from harmonisation.datasets import SHDataset
from harmonisation.datasets.utils import batch_to_xyz
from harmonisation.utils import get_paths_ADNI, train_test_split
from harmonisation.models import ENet

from harmonisation.settings import SIGNAL_PARAMETERS

from dipy.io.image import save_nifti, load_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table

import numpy as np

paths = get_paths_ADNI()
path_train, _, _ = train_test_split(
    paths,
    test_proportion=0,
    validation_proportion=0,
    max_combined_size=None)

file = '003_S_4288_S142486'
path_train = [d for d in path_train if d['name'] == file]

# Create the dataset
dataset = SHDataset(path_train,
                    patch_size=SIGNAL_PARAMETERS["patch_size"],
                    signal_parameters=SIGNAL_PARAMETERS,
                    transformations=None,
                    normalize_data=True,
                    n_jobs=8)

# Load the network
net, _ = ENet.load('../.saved_models/128-4-4-4/9_net')

net = net.to("cuda")
net.eval()

# Get the dmri name
print_name = dataset.names[0]
print_data = dataset.get_data_by_name(print_name)

sh_true = batch_to_xyz(
    print_data['sh'],
    print_data['real_size'],
    SIGNAL_PARAMETERS['overlap_coeff'])

sh_true = sh_true * dataset.std + dataset.mean

sh_pred = net.predict_dataset(dataset, batch_size=128)[print_name]

sh_pred = batch_to_xyz(
    sh_pred,
    print_data['real_size'],
    SIGNAL_PARAMETERS['overlap_coeff'])
sh_pred = sh_pred * dataset.std + dataset.mean

mask = batch_to_xyz(
    print_data['mask'],
    print_data['real_size'],
    SIGNAL_PARAMETERS['overlap_coeff'])

dwi_pred = sh_to_dwi(sh_pred, print_data['gtab'], mask, add_b0=False)

dwi_true, affine = load_nifti(path_train[0]['dwi'])
gtab = gradient_table(*read_bvals_bvecs(
    path_train[0]["bval"], path_train[0]["bvec"]))

patch_size = np.array(SIGNAL_PARAMETERS['patch_size'])
pad_needed = patch_size - dwi_true.shape[:3] % patch_size
pad_needed = [(x // 2, x // 2 + x % 2) for x in pad_needed]

dwi_pred = dwi_pred[pad_needed[0][0]:-pad_needed[0][1],
                    pad_needed[1][0]:-pad_needed[1][1],
                    pad_needed[2][0]:-pad_needed[2][1]]
sh_pred = sh_pred[pad_needed[0][0]:-pad_needed[0][1],
                  pad_needed[1][0]:-pad_needed[1][1],
                  pad_needed[2][0]:-pad_needed[2][1]]
sh_true = sh_true[pad_needed[0][0]:-pad_needed[0][1],
                  pad_needed[1][0]:-pad_needed[1][1],
                  pad_needed[2][0]:-pad_needed[2][1]]
mask = mask[pad_needed[0][0]:-pad_needed[0][1],
            pad_needed[1][0]:-pad_needed[1][1],
            pad_needed[2][0]:-pad_needed[2][1]]

dwi_pred *= np.expand_dims(dwi_true[..., gtab.b0s_mask].mean(-1), axis=-1)
dwi_pred = np.concatenate([dwi_true[..., gtab.b0s_mask], dwi_pred], axis=-1)

assert dwi_pred.shape == dwi_true.shape, (dwi_true.shape, dwi_pred.shape)

np.save("mask.npy", mask)
save_nifti("sh_true.nii.gz", sh_true, affine)
save_nifti("dwi_true.nii.gz", dwi_true, affine)
save_nifti("sh_pred.nii.gz", sh_pred, affine)
save_nifti("dwi_pred.nii.gz", dwi_pred, affine)
