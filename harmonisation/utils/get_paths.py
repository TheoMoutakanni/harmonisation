from os.path import join as pjoin
import os
import random

import pandas as pd
from sklearn.model_selection import train_test_split

from harmonisation.settings import ADNI_PATH, PPMI_PATH, SIMON_PATH


def get_paths_SIMON(simon_path=None, patients=None):
    if simon_path is None:
        simon_path = SIMON_PATH

    if patients is None:
        patients = [x for x in os.listdir(simon_path) if os.path.isdir(
            pjoin(simon_path, x))]

    sites = pd.read_csv(pjoin(simon_path, 'CCNA_manufacturers.csv'))
    sites = {p: s for p, s in zip(sites['folder name'], sites['manufacturer'])}
    sites_dict = {s: i for i, s in enumerate(sorted(set(sites.values())))}

    path_dicts = [
        {'name': patient,
         'site': sites_dict[sites[patient]],
         'dwi': pjoin(simon_path, patient, 'dwi.nii.gz'),
         'bval': pjoin(simon_path, patient, 'bval'),
         'bvec': pjoin(simon_path, patient, 'bvec'),
         'mask': pjoin(simon_path, patient, 'brain_mask.nii.gz'),
         'wm_mask': pjoin(simon_path, patient, 'mask_wm_m.nii.gz'),
         'csf_mask': pjoin(simon_path, patient, 'mask_csf_m.nii.gz'),
         'fa': pjoin(simon_path, patient, 'fa.nii.gz'),
         'md': pjoin(simon_path, patient, 'md.nii.gz'),
         'ad': pjoin(simon_path, patient, 'ad.nii.gz'),
         'rd': pjoin(simon_path, patient, 'rd.nii.gz'),
         'fodf': pjoin(simon_path, patient, 'fodf.nii.gz'),
         'nufo': pjoin(simon_path, patient, 'nufo.nii.gz'),
         'afd_sum': pjoin(simon_path, patient, 'afd_sum.nii.gz'),
         'afd_total': pjoin(simon_path, patient, 'afd_total_sh0.nii.gz')}
        for patient in patients]

    sites_dict = {v: k for k, v in sites_dict.items()}

    return path_dicts, sites_dict


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

    return path_dicts, None


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

    sites_dict = {v: k for k, v in sites_dict.items()}

    return path_dicts, sites_dict


def train_test_val_split(paths,
                         test_proportion,
                         validation_proportion,
                         seed=None,
                         max_combined_size=None,
                         balanced_classes=None,
                         blacklist=[]):
    random.seed(seed)
    paths = [x for x in paths if x['name'] not in blacklist]
    if max_combined_size is None:
        max_combined_size = len(paths)
    if not balanced_classes:
        paths = paths[:max_combined_size]
    else:
        max_nb = max_combined_size // len(balanced_classes)
        nb_classes = {k: 0 for k in balanced_classes}
        new_paths = []
        for path in paths:
            if path['site'] in balanced_classes:
                if nb_classes[path['site']] < max_nb:
                    nb_classes[path['site']] += 1
                    new_paths.append(path)
        paths = new_paths

    train, test = train_test_split(paths,
                                   test_size=test_proportion,
                                   stratify=[x['site'] for x in paths],
                                   random_state=seed)

    validation_proportion = validation_proportion / (1 - test_proportion)

    train, validation = train_test_split(train,
                                         test_size=validation_proportion,
                                         stratify=[x['site'] for x in train],
                                         random_state=seed)

    # index_test = int(round(len(paths) * test_proportion))
    # random.shuffle(paths)
    # test = paths[:index_test]
    # paths_train = paths[index_test:]

    # index_validation = int(round(len(paths) * validation_proportion))
    # random.shuffle(paths_train)
    # validation = paths_train[:index_validation]
    # train = paths_train[index_validation:]

    return train, validation, test
