from dipy.segment.mask import median_otsu
from dipy.io.image import load_nifti

from harmonisation.functions.shm import dwi_to_sh, normalize_data
from harmonisation.datasets.utils import xyz_to_batch

import numpy as np
import torch


def process_data(path_dict, gtab, signal_parameters):
    processing_parameters = signal_parameters["processing_params"]
    data, _ = load_nifti(path_dict["dwi"])
    data = data[:].copy()
    # Crop the MRI
    b0_mask_crop, mask = median_otsu(
        data,
        vol_idx=gtab.b0s_mask,
        autocrop=True,
        **processing_parameters['median_otsu_params'])

    b0_mask_crop = normalize_data(b0_mask_crop, ~gtab.b0s_mask)

    shm_coeff = dwi_to_sh(b0_mask_crop, gtab, mask=mask,
                          **processing_parameters["sh_params"])

    # Pad the x,y,z axes so they can be divided by the respective patch size
    patch_size = np.array(signal_parameters["patch_size"])
    pad_needed = patch_size - shm_coeff.shape[:3] % patch_size

    shm_coeff = np.pad(shm_coeff,
                       pad_width=[(0, x) for x in pad_needed] + [(0, 0)],
                       mode="edge")

    mask = np.pad(mask,
                  pad_width=[(0, x) for x in pad_needed],
                  constant_values=0)

    # (x, y, z, sh)
    shm_coeff = torch.FloatTensor(shm_coeff)
    mask = torch.FloatTensor(mask.astype(int)).unsqueeze(-1)

    shm_coeff, number_of_patches = xyz_to_batch(shm_coeff, patch_size)
    mask, _ = xyz_to_batch(mask, patch_size)

    data = {'sh': shm_coeff,
            'mask': mask,
            'number_of_patches': number_of_patches,
            'gtab': gtab}

    return data
