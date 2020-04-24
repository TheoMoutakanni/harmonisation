from dipy.segment.mask import median_otsu
from dipy.io.image import load_nifti

from harmonisation.functions.shm import dwi_to_sh, normalize_data

import numpy as np


def process_data(path_dict, gtab, signal_parameters):
    processing_parameters = signal_parameters["processing_params"]
    data, affine = load_nifti(path_dict["dwi"])
    data = data[:].copy()
    # Crop the MRI

    try:
        mask, _ = load_nifti(path_dict["mask"])
    except FileNotFoundError:
        print('No mask found, generating one, may be erroneous')
        data, mask = median_otsu(
            data,
            vol_idx=gtab.b0s_mask,
            autocrop=True,
            **processing_parameters['median_otsu_params'])

    data = normalize_data(data, ~gtab.b0s_mask)

    mask = np.expand_dims(mask.astype(int), axis=-1)

    sh_coeff = dwi_to_sh(data, gtab, mask=mask,
                         **processing_parameters["sh_params"])

    # Pad the x,y,z axes so they can be divided by the respective patch size
    patch_size = np.array(signal_parameters["patch_size"])
    pad_needed = patch_size - sh_coeff.shape[:3] % patch_size
    pad_needed = [(x // 2, x // 2 + x % 2) for x in pad_needed] + [(0, 0)]

    sh_coeff = np.pad(sh_coeff, pad_width=pad_needed)
    mask = np.pad(mask, pad_width=pad_needed)

    real_size = sh_coeff.shape[:3]

    sh_coeff = sh_coeff.astype(np.float32)

    data = {'sh': sh_coeff,
            'mask': mask,
            'real_size': real_size,
            'gtab': gtab}

    return data
