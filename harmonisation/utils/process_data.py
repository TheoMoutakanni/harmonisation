from dipy.segment.mask import median_otsu
from dipy.io.image import load_nifti

from harmonisation.functions.shm import dwi_to_sh, normalize_data

import numpy as np


def process_data(path_dict, gtab, signal_parameters):
    processing_parameters = signal_parameters["processing_params"]
    try:
        data, affine = load_nifti(path_dict["dwi"])
    except Exception as e:
        print(path_dict['name'])
        raise e
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

    mask = np.expand_dims(mask.astype(int), axis=-1)

    mean_b0 = data[..., gtab.b0s_mask].mean(-1)
    mean_b0 = np.expand_dims(mean_b0, axis=-1)
    mean_b0 *= mask

    data = normalize_data(data, gtab.b0s_mask)

    sh_coeff = dwi_to_sh(data, gtab, mask=mask,
                         sh_order=signal_parameters['sh_order'],
                         ** processing_parameters["sh_params"])

    # Pad the x,y,z axes so they can be divided by the respective patch size
    patch_size = np.array(signal_parameters["patch_size"])
    pad_needed = patch_size - sh_coeff.shape[:3] % patch_size
    pad_needed = [(x // 2, x // 2 + x % 2) for x in pad_needed] + [(0, 0)]

    sh_coeff = np.pad(sh_coeff, pad_width=pad_needed)
    mask = np.pad(mask, pad_width=pad_needed)
    mean_b0 = np.pad(mean_b0, pad_width=pad_needed)

    real_size = sh_coeff.shape[:3]

    sh_coeff = sh_coeff.astype(np.float32)
    mean_b0 = mean_b0.astype(np.float32)

    data = {'sh': sh_coeff,
            'mask': mask,
            'mean_b0': mean_b0,
            'real_size': real_size,
            'gtab': gtab}

    if 'site' in path_dict.keys():
        data['site'] = path_dict['site']

    return data
