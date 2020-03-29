from os.path import join as pjoin
import os
import random

from harmonisation.settings import ADNI_PATH


def get_paths_ADNI(patients=None):
    if patients is None:
        patients = os.listdir(pjoin(ADNI_PATH, 'raw'))

    path_dicts = [
        {'name': patient,
         'dwi': pjoin(ADNI_PATH, 'raw', patient, 'dwi.nii.gz'),
         't1': pjoin(ADNI_PATH, 'raw', patient, 't1.nii.gz'),
         'bval': pjoin(ADNI_PATH, 'raw', patient, 'bval'),
         'bvec': pjoin(ADNI_PATH, 'raw', patient, 'bvec'), }
        for patient in patients]

    return path_dicts


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
