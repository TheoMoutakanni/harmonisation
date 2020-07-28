import torch


class AdversarialDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, net, batch_size=128):
        data_true = {name: dataset.get_data_by_name(name)
                     for name in dataset.names}
        data_pred = net.predict_dataset(dataset,
                                        batch_size=batch_size)

        self.data = list(data_true.values()) + list(data_pred.values())
        self.labels = torch.FloatTensor(
            [1] * len(data_true) + [0] * len(data_pred)).unsqueeze(1)
        self.names = [name + "_true" for name in data_true.keys()] + \
            [name + "_pred" for name in data_pred.keys()]
        self.name_to_idx = {name: idx for idx, name in enumerate(self.names)}
        self.dataset_indexes = [(i, j) for i, d in enumerate(self.data)
                                for j in range(
                                    d['sh'].shape[0] if (
                                        'sh' in d) else (
                                        d['sh_pred'].shape[0]
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
                'label': labels}

    def get_data_by_name(self, dmri_name):
        """Return a dict with params:
        -'sh': signal in Spherical Harmonics Basis and batch shape
        -'label': the label of the signal (1=true signal, 0=predicted signal)
        """
        patient_idx = self.name_to_idx[dmri_name]
        d = {'label': self.labels[patient_idx]}
        d.update(self.data[patient_idx])
        return d
