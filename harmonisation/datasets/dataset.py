from harmonisation.utils.process_data import process_data
from harmonisation.datasets.utils import xyz_to_batch

from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table

from joblib import Memory, Parallel, delayed
import tqdm

import torch
import numpy as np


class SHDataset(torch.utils.data.Dataset):
    def __init__(self,
                 path_dicts,
                 patch_size,
                 signal_parameters,
                 transformations=None,
                 normalize_data=True,
                 mean=None,
                 std=None,
                 b0_mean=None,
                 b0_std=None,
                 n_jobs=1,
                 cache_dir=None):

        self.names = [p['name'] for p in path_dicts]
        self.transformations = transformations
        self.signal_parameters = signal_parameters
        self.mean = mean
        self.std = std
        self.b0_mean = b0_mean
        self.b0_std = b0_std
        self.patch_size = patch_size

        get_data = process_data
        if cache_dir is not None:
            memory = Memory(cache_dir + "/.cache/",
                            mmap_mode="r", verbose=0)
            get_data = memory.cache(get_data)

        gtabs = {path_dict['name']: gradient_table(*read_bvals_bvecs(
            path_dict["bval"], path_dict["bvec"]))
            for path_dict in path_dicts}

        self.data = Parallel(
            n_jobs=n_jobs,
            prefer="threads")(delayed(get_data)(
                path_dict=path_dict,
                gtab=gtabs[path_dict['name']],
                signal_parameters=signal_parameters,
            ) for path_dict in tqdm.tqdm(path_dicts))

        if normalize_data:
            self.normalize_data()

        for d in self.data:
            d['sh'], _ = xyz_to_batch(
                d['sh'],
                patch_size,
                overlap_coeff=signal_parameters['overlap_coeff'])
            d['mask'], _ = xyz_to_batch(
                d['mask'],
                patch_size,
                overlap_coeff=signal_parameters['overlap_coeff'])
            d['mean_b0'], _ = xyz_to_batch(
                d['mean_b0'],
                patch_size,
                overlap_coeff=signal_parameters['overlap_coeff'])

            # Remove patches with no data
            d['empty'] = d['mask'].squeeze().sum(-1).sum(-1).sum(-1) == 0
            d['sh'] = d['sh'][~d['empty']]
            d['mask'] = d['mask'][~d['empty']]
            d['mean_b0'] = d['mean_b0'][~d['empty']]

        self.name_to_idx = {name: idx for idx, name in enumerate(self.names)}
        self.dataset_indexes = [(i, j) for i, d in enumerate(self.data)
                                for j in range(d['sh'].shape[0])]

        print(len(self.dataset_indexes))

    def __len__(self):
        return len(self.dataset_indexes)

    def __getitem__(self, idx):
        patient_idx, patch_idx = self.dataset_indexes[idx]
        signal = self.data[patient_idx]['sh'][patch_idx]
        mask = self.data[patient_idx]['mask'][patch_idx]
        mean_b0 = self.data[patient_idx]['mean_b0'][patch_idx]
        site = self.data[patient_idx]['site']

        signal = torch.FloatTensor(signal)
        mask = torch.LongTensor(mask)
        mean_b0 = torch.FloatTensor(mean_b0)
        site = torch.LongTensor(site)

        if self.transformations is not None:
            signal = self.transformations(signal)

        return {'sh': signal, 'mask': mask,
                'mean_b0': mean_b0, 'site': site}

    def get_data_by_name(self, dmri_name):
        """Return a dict with params:
        -'sh': signal in Spherical Harmonics Basis and batch shape
        -'mask': mask of the brain in batch shape
        -'number_of_patches': the number of patches per axis in x,y,z coords"""
        patient_idx = self.name_to_idx[dmri_name]
        return self.data[patient_idx]

    def normalize_data(self):
        if self.mean is None or self.std is None:
            self.mean = np.mean(np.stack(
                [d['sh'].reshape(-1, d['sh'].shape[-1]).mean(0)
                 for d in self.data]), 0)
            self.std = np.mean(np.stack(
                [d['sh'].reshape(-1, d['sh'].shape[-1]).std(0)
                 for d in self.data]), 0)

        if self.b0_mean is None or self.b0_std is None:
            self.b0_mean = np.mean(np.stack(
                [d['mean_b0'].reshape(-1, d['mean_b0'].shape[-1]).mean(0)
                 for d in self.data]), 0)
            self.b0_std = np.mean(np.stack(
                [d['mean_b0'].reshape(-1, d['mean_b0'].shape[-1]).std(0)
                 for d in self.data]), 0)

        for d in self.data:
            d['sh'] = (d['sh'] - self.mean) / self.std
            d['mean_b0'] = (d['mean_b0'] - self.b0_mean) / self.b0_std
