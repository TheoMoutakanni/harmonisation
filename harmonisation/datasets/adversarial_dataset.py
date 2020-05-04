import torch


class AdversarialDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, net):
        data_true = {name: dataset.get_data_by_name(name)['sh']
                     for name in dataset.names}
        data_pred = net.predict_dataset(dataset,
                                        batch_size=128)

        self.data = list(data_true.values()) + list(data_pred.values())
        self.sites = [dataset.get_data_by_name(name)['site']
                      for name in dataset.names] * 2
        self.labels = torch.FloatTensor(
            [1] * len(data_true) + [0] * len(data_pred)).unsqueeze(1)
        self.names = [name + "_true" for name in data_true.keys()] + \
            [name + "_pred" for name in data_pred.keys()]
        self.name_to_idx = {name: idx for idx, name in enumerate(self.names)}
        self.dataset_indexes = [(i, j) for i, d in enumerate(self.data)
                                for j in range(d.shape[0])]

    def __len__(self):
        return len(self.dataset_indexes)

    def __getitem__(self, idx):
        patient_idx, patch_idx = self.dataset_indexes[idx]
        signal = self.data[patient_idx][patch_idx]
        labels = self.labels[patient_idx]

        return signal, labels

    def get_data_by_name(self, dmri_name):
        """Return a dict with params:
        -'sh': signal in Spherical Harmonics Basis and batch shape
        -'label': the label of the signal (1=true signal, 0=predicted signal)
        """
        patient_idx = self.name_to_idx[dmri_name]
        return {'sh': self.data[patient_idx],
                'label': self.labels[patient_idx],
                'site': self.sites[patient_idx]}
