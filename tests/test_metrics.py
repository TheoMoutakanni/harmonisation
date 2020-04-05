import pytest
import numpy as np
import torch
from os.path import join as pjoin

from dipy.io.image import load_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.shm import CsaOdfModel

from harmonisation.functions import metrics
from harmonisation.functions import shm


@pytest.fixture
def ADNI_names():
    return ["003_S_0908_S210038", "003_S_1074_S256382"]


@pytest.fixture
def path_dicts(ADNI_names):
    path = "./tests/test_files"

    path_dicts = {patient: {'dwi': pjoin(path, 'raw', patient, 'dwi.nii.gz'),
                            't1': pjoin(path, 'raw', patient, 't1.nii.gz'),
                            'bval': pjoin(path, 'raw', patient, 'bval'),
                            'bvec': pjoin(path, 'raw', patient, 'bvec')}
                  for patient in ADNI_names}

    return path_dicts


@pytest.fixture
def gtabs(path_dicts):
    gtabs = []
    for patient, path_dict in path_dicts.items():
        bvals, bvecs = read_bvals_bvecs(
            path_dict["bval"], path_dict["bvec"])
        gtab = gradient_table(bvals, bvecs)
        gtabs.append(gtab)
    return gtabs


@pytest.fixture
def datas_dwi(path_dicts):
    datas = []
    for patient, path_dict in path_dicts.items():
        data, affine = load_nifti(path_dict['dwi'])
        datas.append(data)
    return datas


@pytest.fixture
def datas_sh(datas_dwi, gtabs):
    return [shm.dwi_to_sh(data, gtab, sh_order=4)
            for data, gtab in zip(datas_dwi, gtabs)]


def test_torch_angular_corr_coeff(datas_sh):
    rtol = 1e-05
    atol = 1e-08

    datas_sh = [torch.FloatTensor(data) for data in datas_sh]

    acc_11 = metrics.torch_angular_corr_coeff(datas_sh[0], datas_sh[0])
    acc_11[torch.isnan(acc_11)] = 1
    acc_22 = metrics.torch_angular_corr_coeff(datas_sh[1], datas_sh[1])
    acc_22[torch.isnan(acc_22)] = 1
    acc_12 = metrics.torch_angular_corr_coeff(datas_sh[0], datas_sh[1])
    acc_12[torch.isnan(acc_12)] = 1
    acc_21 = metrics.torch_angular_corr_coeff(datas_sh[1], datas_sh[0])
    acc_21[torch.isnan(acc_21)] = 1
    acc_1n1 = metrics.torch_angular_corr_coeff(datas_sh[0], -datas_sh[0])
    acc_1n1[torch.isnan(acc_1n1)] = -1

    ones = torch.ones(acc_11.shape)

    assert torch.max(acc_12) <= 1. + rtol
    assert torch.min(acc_12) >= -1. - rtol

    assert torch.isclose(acc_12, acc_21, rtol=rtol, atol=atol).all()
    assert torch.isclose(acc_11, ones, rtol=rtol, atol=atol).all()
    assert torch.isclose(acc_22, ones, rtol=rtol, atol=atol).all()
    assert torch.isclose(acc_1n1, -ones, rtol=rtol, atol=atol).all()


def test_torch_gfa(datas_dwi, gtabs):
    rtol = 1e-04
    atol = 1e-08

    idx = 0
    data_dwi = datas_dwi[idx][100:140, 100:140, 28:29]
    gtab = gtabs[idx]

    csamodel = CsaOdfModel(gtab, 4)
    csamodel = csamodel.fit(data_dwi)
    data_sh = csamodel.shm_coeff
    csagfa = csamodel.gfa

    gfa = metrics.torch_gfa(torch.FloatTensor(data_sh)).numpy()

    assert np.isclose(csagfa, gfa, rtol=rtol, atol=atol).all()
