from os.path import join as pjoin
import os
import random

import pandas as pd

from harmonisation.settings import ADNI_PATH, PPMI_PATH


def get_paths_ADNI(patients=None):
    if patients is None:
        patients = os.listdir(pjoin(ADNI_PATH, 'raw'))

    path_dicts = [
        {'name': patient,
         'dwi': pjoin(ADNI_PATH, 'raw', patient, 'dwi.nii.gz'),
         't1': pjoin(ADNI_PATH, 'raw', patient, 't1.nii.gz'),
         'bval': pjoin(ADNI_PATH, 'raw', patient, 'bval'),
         'bvec': pjoin(ADNI_PATH, 'raw', patient, 'bvec'),
         'mask': pjoin(ADNI_PATH, 'raw', patient, 'brain_mask.nii.gz')}
        for patient in patients]

    return path_dicts


def get_paths_PPMI(patients=None):
    if patients is None:
        patients = [x for x in os.listdir(PPMI_PATH) if os.path.isdir(
            pjoin(PPMI_PATH, x))]

    sites = pd.read_csv(pjoin(PPMI_PATH, 'Center-Subject_List.csv'))
    sites = sites.astype({'PATNO': 'str'})

    sites = {p: s for p, s in zip(sites['PATNO'], sites['CNO']) if p in [
        p.split('_')[0] for p in patients]}
    sites_dict = {s: i for i, s in enumerate(sorted(set(sites.values())))}

    path_dicts = [
        {'name': patient,
         'dwi': pjoin(PPMI_PATH, patient, 'dwi.nii.gz'),
         't1': pjoin(PPMI_PATH, patient, 't1.nii.gz'),
         'bval': pjoin(PPMI_PATH, patient, 'bval'),
         'bvec': pjoin(PPMI_PATH, patient, 'bvec'),
         'mask': pjoin(PPMI_PATH, patient, 'brain_mask.nii.gz'),
         'site': sites_dict[sites[patient.split('_')[0]]],
         }
        for patient in patients]

    return path_dicts, sites_dict


def train_test_split(paths,
                     test_proportion,
                     validation_proportion,
                     seed=None,
                     max_combined_size=None,
                     blacklist=[]):
    random.seed(seed)
    if max_combined_size is None:
        max_combined_size = len(paths)
    paths = [x for x in paths[:max_combined_size] if x not in blacklist]

    index_test = int(len(paths) * test_proportion)
    random.shuffle(paths)
    test = paths[:index_test]
    paths_train = paths[index_test:]

    index_validation = int(len(paths_train) * validation_proportion)
    random.shuffle(paths_train)
    validation = paths_train[:index_validation]
    train = paths_train[index_validation:]

    return train, validation, test
