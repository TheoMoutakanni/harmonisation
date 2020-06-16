import pytest
import numpy as np
import torch
from os.path import join as pjoin

from harmonisation.datasets import SHDataset
from harmonisation.utils import get_paths_ADNI, train_test_val_split
from harmonisation.trainers import BaseTrainer
from harmonisation.models import ENet


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
def signal_params():
    signal_parameters = {
        'patch_size': [12, 12, 12],
        'sh_order': 4,
        'overlap_coeff': 1,
        'processing_params': {
            'median_otsu_params': {
                'median_radius': 3,
                'numpass': 1,
            },
            'sh_params': {
                'smooth': 0.006,
            }
        }
    }
    return signal_parameters


@pytest.fixture
def network(signal_params):
    net = ENet(patch_size=signal_params["patch_size"],
               sh_order=signal_params['sh_order'],
               embed=128,
               encoder_relu=False,
               decoder_relu=True)
    net = net.to('cuda')
    return net


def test_training(path_dicts, signal_params, net):
    dataset = SHDataset(path_dicts,
                        patch_size=signal_params['patch_size'],
                        signal_parameters=signal_params,
                        transformations=None,
                        n_jobs=-1,
                        cache_dir=None)

    trainer = BaseTrainer(
        net,
        optimizer_parameters={
            "lr": 0.01,
            "weight_decay": 1e-8,
        },
        loss_specs={
            "type": "mse",
            "parameters": {}
        },
        metrics=["acc", "mse_gfa", "mse"],
        metric_to_maximize="mse",
        patience=100,
        save_folder=None,
    )

    trainer.train(dataset,
                  dataset,
                  num_epochs=5,
                  batch_size=128,
                  validation=True)
