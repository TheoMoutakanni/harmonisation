import numpy as np
import os
from os.path import join as pjoin
import pandas as pd
import tqdm

from dipy.io.image import load_nifti

import seaborn as sns
import matplotlib.pyplot as plt


PPMI_PATH = '/media/theo/285EDDF95EDDC02C/Users/Public/Documents/PPMI'
PPMI_PATH_patients = '/home/theo/Documents/Data/PPMI/'

patients = [x for x in os.listdir(PPMI_PATH_patients) if os.path.isdir(
    pjoin(PPMI_PATH_patients, x))]

sites = pd.read_csv(pjoin(PPMI_PATH, 'Center-Subject_List.csv'))
sites = sites.astype({'PATNO': 'str'})

sites = {p: s for p, s in zip(sites['PATNO'], sites['CNO']) if p in [
    p.split('_')[0] for p in patients]}
sites_dict = {s: i for i, s in enumerate(sorted(set(sites.values())))}

path_dicts = [
    {'name': patient,
     'fa': pjoin(PPMI_PATH, 'metrics', patient, 'metrics', 'fa.nii.gz'),
     'md': pjoin(PPMI_PATH, 'metrics', patient, 'metrics', 'md.nii.gz'),
     'gfa': pjoin(PPMI_PATH, 'metrics', patient, 'metrics', 'gfa.nii.gz'),
     'site': sites_dict[sites[patient.split('_')[0]]],
     }
    for patient in patients]

sites_dict = {i: s for s, i in sites_dict.items()}

sns.set_style("white")
kwargs = dict(hist_kws={'alpha': .6}, kde_kws={'linewidth': 2})
plt.figure(figsize=(10, 7), dpi=80)

colors = ['r', 'g', 'b', 'k', 'y', 'purple', 'pink', 'cyan', 'orange']
colors = {s: colors[i] for i, s in sites_dict.items()}

lfa = {}
nb_lfa = {}
for path in tqdm.tqdm(path_dicts):
    try:
        fa, affine = load_nifti(path['fa'])
    except Exception as e:
        continue
    fa = fa.reshape(-1)
    fa = fa[fa != 0]
    site = sites_dict[path['site']]
    if site not in lfa.keys():
        lfa[site] = fa
        nb_lfa[site] = 1
    else:
        lfa[site] = np.concatenate((lfa[site], fa), axis=0)
        nb_lfa[site] += 1

    #sns.distplot(fa, color=colors[site], label=str(site), **kwargs)

print(nb_lfa)

import operator as op
sorted_keys, sorted_vals = zip(
    *sorted(lfa.items()))#, key=op.itemgetter(1)))

sns.boxplot(data=sorted_vals, width=.18)
#sns.swarmplot(data=sorted_vals, size=6, edgecolor="black", linewidth=.9)

# category labels
plt.xticks(plt.xticks()[0], sorted_keys)

plt.legend()
plt.show()
