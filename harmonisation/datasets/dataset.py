from harmonisation.utils.process_data import process_data

from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table

from joblib import Memory, Parallel, delayed
import tqdm

import torch


class SHDataset(torch.utils.data.Dataset):
    def __init__(self,
                 path_dicts,
                 patch_size,
                 signal_parameters,
                 transformations=None,
                 normalize_data=True,
                 mean=None,
                 std=None,
                 n_jobs=1,
                 cache_dir=None):

        self.names = [p['name'] for p in path_dicts]
        self.transformations = transformations
        self.mean = mean
        self.std = std
        self.patch_size = patch_size

        get_data = process_data
        if cache_dir is not None:
            memory = Memory(cache_dir + "/.cache/",
                            mmap_mode="r", verbose=0)
            get_data = memory.cache(get_data)

        self.gtabs = [gradient_table(*read_bvals_bvecs(
            path_dict["bval"], path_dict["bvec"]))
            for path_dict in path_dicts]

        def gen():
            for path_dict, gtab in tqdm.tqdm(zip(path_dicts, self.gtabs),
                                             total=len(path_dicts)):
                # print(path_dict['name'])
                yield path_dict, gtab

        self.data = Parallel(
            n_jobs=n_jobs,
            prefer="threads")(delayed(get_data)(
                path_dict=path_dict,
                gtab=gtab,
                signal_parameters=signal_parameters,
            ) for path_dict, gtab in gen())

        self.name_to_idx = {name: idx for idx, name in enumerate(self.names)}
        self.dataset_indexes = [(i, j) for i, d in enumerate(self.data)
                                for j in range(d['sh'].shape[0])]

        if normalize_data:
            self.normalize_data()

    def __len__(self):
        return len(self.dataset_indexes)

    def __getitem__(self, idx):
        patient_idx, patch_idx = self.dataset_indexes[idx]
        signal = self.data[patient_idx]['sh'][patch_idx]
        mask = self.data[patient_idx]['mask'][patch_idx]

        if self.transformations is not None:
            signal = self.transformations(signal)

        return signal, mask

    def get_data_by_name(self, dmri_name):
        """Return a dict with params:
        -'sh': signal in Spherical Harmonics Basis and batch shape
        -'mask': mask of the brain in batch shape
        -'number_of_patches': the number of patches per axis in x,y,z coords"""
        patient_idx = self.name_to_idx[dmri_name]
        return self.data[patient_idx]

    def normalize_data(self):
        if self.mean is None or self.std is None:
            self.mean = torch.mean(torch.stack(
                [d['sh'].flatten(end_dim=-2).mean(0)
                 for d in self.data]), 0)
            self.std = torch.mean(torch.stack(
                [d['sh'].flatten(end_dim=-2).std(0)
                 for d in self.data]), 0)

        for d in self.data:
            d['sh'] = (d['sh'] - self.mean) / self.std
