import torch


class AdversarialDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, net, batch_size=128):
        data_true = {name: dataset.get_data_by_name(name)
                     for name in dataset.names}
        data_fake = net.predict_dataset(dataset,
                                        batch_size=batch_size)

        names = list(data_true.keys())
        for name in names:
            data_fake[name]['site'] = data_true[name]['site']
            data_fake[name]['mask'] = data_true[name]['mask']

        self.data = list([data_true[name] for name in names])
        self.data += list([data_fake[name] for name in names])
        self.labels = torch.FloatTensor(
            [1] * len(data_true) + [0] * len(data_fake)).unsqueeze(1)
        self.names = [name + "_true" for name in data_true.keys()] + \
            [name + "_fake" for name in data_fake.keys()]
        self.name_to_idx = {name: idx for idx, name in enumerate(self.names)}
        self.dataset_indexes = [(i, j) for i, d in enumerate(self.data)
                                for j in range(
                                    d['sh'].shape[0] if (
                                        'sh' in d) else (
                                        d['sh_fake'].shape[0]
                                    ))]

    def __len__(self):
        return len(self.dataset_indexes)

    def __getitem__(self, idx):
        patient_idx, patch_idx = self.dataset_indexes[idx]
        signal = self.data[patient_idx][patch_idx]
        labels = self.labels[patient_idx]
        mask = self.data[patient_idx]['mask'][patch_idx]
        mean_b0 = self.data[patient_idx]['mean_b0'][patch_idx]
        site = self.data[patient_idx]['site']

        site = torch.LongTensor([site])
        signal = torch.FloatTensor(signal)
        mask = torch.LongTensor(mask)
        mean_b0 = torch.FloatTensor(mean_b0)

        if self.transformations is not None:
            signal = self.transformations(signal)

        return {'sh': signal, 'mask': mask,
                'mean_b0': mean_b0, 'site': site,
                'labels_gan': labels}

    def get_data_by_name(self, dmri_name):
        """Return a dict with params:
        -'sh': signal in Spherical Harmonics Basis and batch shape
        -'labels_gan': the label of the signal (1=true, 0=fake)
        """
        patient_idx = self.name_to_idx[dmri_name]
        d = {'labels_gan': self.labels[patient_idx]}
        for k in self.data[patient_idx]:
            if k not in ['sh_fake', 'mean_b0_fake']:
                d[k] = self.data[patient_idx][k]
            elif k == 'sh_fake':
                d['sh'] = self.data[patient_idx][k]
            elif k == 'mean_b0_fake':
                d['mean_b0'] = self.data[patient_idx][k]
        return d
