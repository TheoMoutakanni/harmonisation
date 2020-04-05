import shutil
import pytest
import time
import os
from os.path import join as pjoin
import numpy as np

import torch

from harmonisation.datasets import SHDataset
from harmonisation.datasets.utils import xyz_to_batch, batch_to_xyz


@pytest.fixture
def ADNI_names():
    return ["003_S_0908_S210038", "003_S_1074_S256382"]


@pytest.fixture
def path_dicts(ADNI_names):
    path = "./tests/test_files"
    path_dict = [{'name': patient,
                  'dwi': pjoin(path, 'raw', patient, 'dwi.nii.gz'),
                  't1': pjoin(path, 'raw', patient, 't1.nii.gz'),
                  'bval': pjoin(path, 'raw', patient, 'bval'),
                  'bvec': pjoin(path, 'raw', patient, 'bvec'), }
                 for patient in ADNI_names]

    return path_dict


@pytest.fixture
def cache_directory():
    return "./tests/test_files/.cache"


@pytest.fixture
def signal_params():
    signal_parameters = {
        'patch_size': [12, 12, 12],
        'overlap_coeff': 2,
        'processing_params': {
            'median_otsu_params': {
                'median_radius': 3,
                'numpass': 1,
            },
            'sh_params': {
                'sh_order': 4,
                'smooth': 0.006,
            }
        }
    }
    return signal_parameters


def test_dataset(ADNI_names, path_dicts, signal_params):
    dataset = SHDataset(path_dicts,
                        patch_size=signal_params['patch_size'],
                        signal_parameters=signal_params,
                        transformations=None,
                        n_jobs=-1,
                        cache_dir=None)

    sh_order = signal_params['processing_params']['sh_params']['sh_order']
    ncoef = (sh_order + 2) * (sh_order + 1) / 2

    signal, mask = dataset[300]

    assert list(signal.shape) == signal_params['patch_size'] + [ncoef]

    assert len(dataset) == 945

    patient = np.random.choice(ADNI_names)
    dataset.get_data_by_name(patient)


def test_parallel_is_faster(path_dicts, signal_params, cache_directory):

    shutil.rmtree(cache_directory, ignore_errors=True)
    t1 = time.time()
    SHDataset(path_dicts,
              patch_size=signal_params['patch_size'],
              signal_parameters=signal_params,
              transformations=None,
              n_jobs=-1,
              cache_dir=None)
    t1 = time.time() - t1

    shutil.rmtree(cache_directory, ignore_errors=True)
    t2 = time.time()
    SHDataset(path_dicts,
              patch_size=signal_params['patch_size'],
              signal_parameters=signal_params,
              transformations=None,
              n_jobs=1,
              cache_dir=None)
    t2 = time.time() - t2

    assert t2 > t1


def test_cache_is_faster(path_dicts, signal_params, cache_directory):

    shutil.rmtree(cache_directory, ignore_errors=True)
    t1 = time.time()
    SHDataset(path_dicts,
              patch_size=signal_params['patch_size'],
              signal_parameters=signal_params,
              transformations=None,
              n_jobs=-1,
              cache_dir=cache_directory)
    t1 = time.time() - t1

    # We don't delete the cache so the new dataset can use it
    t2 = time.time()
    SHDataset(path_dicts,
              patch_size=signal_params['patch_size'],
              signal_parameters=signal_params,
              transformations=None,
              n_jobs=-1,
              cache_dir=cache_directory)
    t2 = time.time() - t2

    assert t2 < t1


def test_cache_no_cache(path_dicts, signal_params, cache_directory):

    shutil.rmtree(cache_directory, ignore_errors=True)
    SHDataset(path_dicts,
              patch_size=signal_params['patch_size'],
              signal_parameters=signal_params,
              transformations=None,
              n_jobs=-1,
              cache_dir=None)
    assert not os.path.isdir(cache_directory)

    SHDataset(path_dicts,
              patch_size=signal_params['patch_size'],
              signal_parameters=signal_params,
              transformations=None,
              n_jobs=-1,
              cache_dir=cache_directory)
    assert os.path.isdir(cache_directory)


def test_batch_xyz(ADNI_names, path_dicts, signal_params):
    dataset = SHDataset(path_dicts,
                        patch_size=signal_params['patch_size'],
                        signal_parameters=signal_params,
                        transformations=None,
                        cache_dir=None)

    patient = ADNI_names[0]  # np.random.choice(ADNI_names)
    data_patient = dataset.get_data_by_name(patient)

    data_batch = data_patient['sh']

    data_xyz = batch_to_xyz(data_batch, data_patient['real_size'])

    assert data_xyz.shape[:3] == data_patient['real_size']

    data_batch_2, number_of_patches = xyz_to_batch(
        data_xyz, signal_params['patch_size'],
        overlap_coeff=signal_params['overlap_coeff'])

    assert number_of_patches == data_patient['number_of_patches']
    assert torch.isclose(data_batch,
                         data_batch_2,
                         rtol=0.05, atol=1e-6).all()


def test_normalization(ADNI_names, path_dicts, signal_params):
    dataset = SHDataset(path_dicts,
                        patch_size=signal_params['patch_size'],
                        signal_parameters=signal_params,
                        transformations=None,
                        normalize_data=False,
                        cache_dir=None)

    patient = np.random.choice(ADNI_names)
    data_raw = dataset.get_data_by_name(patient)['sh']
    dataset.normalize_data()
    mean, std = dataset.mean, dataset.std
    data_normalized = dataset.get_data_by_name(patient)['sh']

    assert torch.isclose(data_normalized * std + mean,
                         data_raw,
                         rtol=0.05, atol=1e-6).all()
