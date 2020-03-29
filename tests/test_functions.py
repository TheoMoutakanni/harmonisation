import pytest
import numpy as np
from os.path import join as pjoin

from dipy.io.image import load_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.segment.mask import median_otsu

from harmonisation.functions import shm


@pytest.fixture
def ADNI_names():
    return ["003_S_0908_S210038", "003_S_1074_S256382"]


@pytest.fixture
def path_dict(ADNI_names):
    path = "./tests/test_files"
    patient = np.random.choice(ADNI_names)
    path_dict = {'dwi': pjoin(path, 'raw', patient, 'dwi.nii.gz'),
                 't1': pjoin(path, 'raw', patient, 't1.nii.gz'),
                 'bval': pjoin(path, 'raw', patient, 'bval'),
                 'bvec': pjoin(path, 'raw', patient, 'bvec'), }

    return path_dict


@pytest.fixture
def data_dwi(path_dict):
    data, affine = load_nifti(path_dict['dwi'])
    return data


@pytest.fixture
def gtab(path_dict):
    bvals, bvecs = read_bvals_bvecs(
        path_dict["bval"], path_dict["bvec"])
    gtab = gradient_table(bvals, bvecs)
    return gtab


def test_dwi_sh_conversion(data_dwi, gtab):
    """ Check if the dwi->sh->dwi conversion is working
    """
    b0_mask_crop, mask_crop = median_otsu(
        data_dwi,
        vol_idx=gtab.b0s_mask,
        autocrop=False)

    data_dwi_norm = shm.normalize_data(data_dwi, gtab.b0s_mask)

    error = []

    for sh_order in [2, 4, 6, 8, 10]:

        data_sh = shm.dwi_to_sh(data_dwi_norm, gtab,
                                mask=mask_crop, sh_order=sh_order)
        data_dwi_revert = shm.sh_to_dwi(data_sh, gtab, mask=mask_crop)

        data_dwi_true = data_dwi_norm[..., ~gtab.b0s_mask]
        data_dwi_true *= mask_crop[..., None]

        assert data_dwi_revert.shape == data_dwi_true.shape

        error.append(
            np.sum(abs(data_dwi_revert - data_dwi_true)) / np.sum(mask_crop))

    # Check if increasing the order of the SH improve the error
    assert error == sorted(error, reverse=True)
