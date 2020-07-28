import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

from dipy.io.image import load_nifti

import numpy as np
import pandas as pd
import scipy.stats as stats

from harmonisation.utils import get_paths_SIMON

save_folder = "./.saved_models/style_feat_2/"

paths, sites = get_paths_SIMON()
paths_fake, _ = get_paths_SIMON(
    simon_path='/home/theo/Documents/Harmonisation/data/CCNA_feat_2')

folds = {}
with open(save_folder + "train_val_test.txt", "r") as file:
    folds['train'] = file.readline().replace("'", '').replace('[', '').replace(']', '').replace('\n', '').replace(' ', '').split(',')
    folds['val'] = file.readline().replace("'", '').replace('[', '').replace(']', '').replace('\n', '').replace(' ', '').split(',')
    folds['test'] = file.readline().replace("'", '').replace('[', '').replace(']', '').replace('\n', '').replace(' ', '').split(',')

print(folds)

metrics = ['mask', 'wm_mask', 'csf_mask',
           'fa', 'md', 'ad', 'rd']  # , 'afd_total', 'nufo']

sns.set_style("white")
kwargs = dict(hist_kws={'alpha': .6}, kde_kws={'linewidth': 2})
plt.figure(figsize=(10, 7), dpi=80)

colors = ['r', 'g', 'b', 'k', 'y', 'purple', 'pink', 'cyan', 'orange']
df = pd.DataFrame(index=[path['name'] for path in paths] +
                  [path['name'] + '_fake' for path in paths_fake])
df['site'] = ['' for _ in range(len(paths) + len(paths_fake))]
df['fake'] = ['' for _ in range(len(paths) + len(paths_fake))]
df['fold'] = ['' for _ in range(len(paths) + len(paths_fake))]
df['real_name'] = ['' for _ in range(len(paths) + len(paths_fake))]
for metric in metrics:
    df[metric] = [None for _ in range(len(paths) + len(paths_fake))]

for path in tqdm.tqdm(paths):
    data = {}
    try:
        for metric in metrics:
            data[metric] = load_nifti(path[metric])[0]
    except Exception as e:
        print(e)
        raise e
    for metric in metrics:
        data[metric] = data[metric].reshape(-1)
        if metric in ['fa', 'md', 'ad', 'rd']:
            data[metric] = data[metric][data['wm_mask'] != 0]
        elif metric in []:
            data[metric] = data[metric][data['csf_mask'] != 0]
        elif 'mask' not in metric:
            data[metric] = data[metric][data['mask'] != 0]

        if metric in ['rd', 'ad', 'md']:
            data[metric] = data[metric] * 1000

        df.loc[path['name']][metric] = data[metric]

    df.loc[path['name']]['site'] = path['site']
    df.loc[path['name']]['real_name'] = path['name']
    df.loc[path['name']]['fake'] = 'False'

    if path['name'] in folds['train']:
        df.loc[path['name']]['fold'] = 'train'
    elif path['name'] in folds['val']:
        df.loc[path['name']]['fold'] = 'val'
    else:
        df.loc[path['name']]['fold'] = 'test'

for path in tqdm.tqdm(paths_fake):
    data = {}
    try:
        for metric in metrics:
            if '_mask' in metric:
                continue
            data[metric] = load_nifti(path[metric])[0]
    except Exception as e:
        print(e)
        raise e
    for metric in metrics:
        if '_mask' in metric:
            continue
        data[metric] = data[metric].reshape(-1)
        if metric in ['fa', 'md', 'ad', 'rd']:
            data[metric] = data[metric][df.loc[path['name']]['wm_mask'] != 0]
        elif metric in []:
            data[metric] = data[metric][df.loc[path['name']]['csf_mask'] != 0]
        elif 'mask' not in metric:
            data[metric] = data[metric][data['mask'] != 0]

        if metric in ['rd', 'ad', 'md']:
            data[metric] = data[metric] * 1000

        df.loc[path['name'] + '_fake'][metric] = data[metric]

    df.loc[path['name'] + '_fake']['site'] = path['site']
    df.loc[path['name'] + '_fake']['real_name'] = path['name']
    df.loc[path['name'] + '_fake']['fake'] = 'True'

    if path['name'] in folds['train']:
        df.loc[path['name'] + '_fake']['fold'] = 'train'
    elif path['name'] in folds['val']:
        df.loc[path['name'] + '_fake']['fold'] = 'val'
    else:
        df.loc[path['name'] + '_fake']['fold'] = 'test'

    # sns.distplot(fa, color=colors[site], label=str(site), **kwargs)

# df = df[df['site'] != 0]

for metric in ['fa', 'md', 'ad', 'rd']:
    data = df[(df['fake'] == 'False')].groupby('site')
    data = [[np.mean(x) for x in v[metric].values] for k, v in data]
    print('{} Real stats:'.format(metric), stats.f_oneway(*data))
    data = df[(df['fake'] == 'True')].groupby('site')
    data = [[np.mean(x) for x in v[metric].values] for k, v in data]
    print('{} Fake stats:'.format(metric), stats.f_oneway(*data))

manufacturers = dict(sorted([(x['name'], x['site'])
                             for x in paths], key=lambda x: x[1]))
c_manufacturers = {0: 'yellow', 1: 'orange', 2: 'red'}

# df = df.reset_index()
# df['position'] = [np.arange(length) for length in df['fa'].apply(len)]
# df = df.set_index(['index', 'site'])
# df = df.apply(pd.Series.explode)
df = df.reset_index()
df = df.sort_values(by=['site'])

# for metric in metrics:
#     if metric in ['wm_mask', 'csf_mask', 'mask']:
#         continue

#     plt.figure()
#     ax = sns.boxplot(data=df[metric].values, width=.18, showfliers=False)
#     for i in range(len(df)):
#         ax.artists[i].set_facecolor(c_manufacturers[df['site'].iloc[i]])

#     #plt.xticks(df['site'].values, rotation=90)
#     plt.title(metric)


def concat_by_class(array, classes, order=None):
    if order is None:
        order = sorted(set(classes))
    new_array = {}
    for i in range(len(array)):
        if classes[i] not in new_array.keys():
            new_array[classes[i]] = array[i]
        else:
            new_array[classes[i]] = np.concatenate([new_array[classes[i]],
                                                    array[i]])
    return [new_array[x] for x in order]

# plt.figure()

# data_concat = concat_by_class(df['fa'].values, df['site'].values)
# sns.boxplot(data=data_concat, width=.18)
# #sns.swarmplot(data=df.groupby('index')['fa'].apply(np.mean).groupby('site').apply(list).values, size=6, edgecolor="black", linewidth=.9)

# plt.xticks(plt.xticks()[0], rotation=90)


plt.figure()

for metric in metrics:
    if 'mask' not in metric:
        df[metric + '_mean'] = df[metric].apply(np.mean)

for metric in metrics:
    if 'mask' in metric:
        continue

    # plt.figure()
    # ax = sns.swarmplot(data=df, x='site', y=metric + '_mean', hue='fake',
    #                    linewidth=.9, edgecolor="black", size=6, dodge=True)
    g = sns.catplot(x='site', y=metric + '_mean', hue='fold', col='fake',
                    data=df, kind='swarm', linewidth=.9, dodge=False, sharey=False)
    d = sns.catplot(x='fake', y=metric + '_mean', hue='real_name',
                    data=df, kind='point')

    #plt.xticks([sites[x] for x in df['site'].values], rotation=90)
    plt.title(metric)


# for metric in metrics:
#     if 'mask' in metric:
#         continue
#     fig, axarr = plt.subplots(3, 1, sharex=True, sharey=True)
#     for i in range(len(df)):
#         site = df['site'].iloc[i]
#         sns.distplot(df[metric].iloc[i], color=c_manufacturers[site],
#                      # label=df['name'].iloc[i],
#                      hist=False, kde_kws={'linewidth': 2},
#                      ax=axarr[site])
#     for i in range(3):
#         axarr[i].set_title(sites[i])
#     fig.suptitle(metric)
#     plt.tight_layout()

plt.show()
