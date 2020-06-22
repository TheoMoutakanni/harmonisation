import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import copy
from tqdm import tqdm

from harmonisation.functions import metrics, shm
from harmonisation.functions.losses import get_loss_fun
from harmonisation.viz import print_peaks, print_diff, print_RIS, print_data
from harmonisation.datasets.utils import batch_to_xyz

import matplotlib.pyplot as plt


class BaseTrainer():
    def __init__(self,
                 net,
                 optimizer_parameters={
                     "lr": 0.001,
                     "weight_decay": 1e-8,
                 },
                 loss_specs=[{
                     "type": "acc",
                     "parameters": {},
                     "coeff": 1,
                 }],
                 metrics=["acc", "mse"],
                 metric_to_maximize="acc",
                 patience=10,
                 save_folder=None,
                 ):
        self.net = net
        self.metric_to_maximize = metric_to_maximize
        self.metrics = metrics
        self.losses = get_loss_fun(loss_specs)
        self.optimizer = optim.Adam(
            self.net.parameters(), **optimizer_parameters)

        self.patience = patience
        self.save_folder = save_folder

    def validate(self, validation_dataset, batch_size=128):
        """
        Compute metrics on validation_dataset
        """
        data_true = {name: validation_dataset.get_data_by_name(name)
                     for name in validation_dataset.names}
        data_pred = self.net.predict_dataset(validation_dataset,
                                             batch_size=batch_size)
        metrics_fun = metrics.get_metrics_fun()

        metrics_batches = []
        for name in list(set(data_pred.keys()) & set(data_true.keys())):
            dic = dict()
            for metric in self.metrics:
                metric_calc = []
                nb_batches = int(
                    np.ceil(len(data_pred[name]['sh_pred']) / batch_size))
                for batch in range(nb_batches):
                    idx = range(batch * batch_size,
                                min((batch + 1) * batch_size,
                                    len(data_pred[name]['sh_pred'])))
                    metric_calc.append(metrics_fun[metric](
                        torch.FloatTensor(data_true[name]['sh'][idx]),
                        torch.FloatTensor(data_pred[name]['sh_pred'][idx]),
                        torch.LongTensor(data_true[name]['mask'][idx])
                    ))
                dic[metric] = metrics.nanmean(torch.cat(metric_calc)).numpy()
            metrics_batches.append(dic)

        metrics_epoch = {
            metric: np.nanmean(
                [m[metric] for m in metrics_batches])
            for metric in self.metrics
        }

        return metrics_epoch

    def print_metrics(self, validation_dataset, batch_size=128):
        # Print fODF, RIS and acc for a slice of a validation dwi
        print('print')
        name = validation_dataset.names[0]
        data = validation_dataset.get_data_by_name(name)
        sh_true = data['sh']  # torch.FloatTensor(data['sh'])
        data_mask = data['mask']  # torch.LongTensor(data['mask'])
        sh_pred = self.net.predict_dataset(validation_dataset,
                                           batch_size=batch_size,
                                           names=[name])[name]

        print('predicted')

        overlap_coeff = validation_dataset.signal_parameters['overlap_coeff']

        sh_pred = batch_to_xyz(
            sh_pred,
            data['real_size'],
            empty=data['empty'],
            overlap_coeff=overlap_coeff)
        sh_true = batch_to_xyz(
            sh_true,
            data['real_size'],
            empty=data['empty'],
            overlap_coeff=overlap_coeff)
        mask = batch_to_xyz(
            data_mask,
            data['real_size'],
            empty=data['empty'],
            overlap_coeff=overlap_coeff)

        print('batch')

        sh_pred = torch.FloatTensor(sh_pred)
        sh_true = torch.FloatTensor(sh_true)
        mask = torch.LongTensor(mask)

        print('go')

        # print_peaks(sh_true)
        # print_peaks(sh_pred)

        if validation_dataset.mean is not None:
            sh_true = sh_true * validation_dataset.std + validation_dataset.mean
            sh_pred = sh_pred * validation_dataset.std + validation_dataset.mean

        print(metrics.torch_RIS(sh_true[50:51, 50:51, 28:29]))
        print(metrics.torch_RIS(sh_pred[50:51, 50:51, 28:29]))
        print_data(metrics.torch_gfa(sh_true),
                   metrics.torch_gfa(sh_pred),
                   mask)
        print_data(metrics.torch_anisotropic_power(sh_true),
                   metrics.torch_anisotropic_power(sh_pred),
                   mask)

        # dwi_true = shm.sh_to_dwi(sh_true, data['gtab'])
        # dwi_pred = shm.sh_to_dwi(sh_pred, data['gtab'])
        # evals_true = metrics.ols_fit_tensor(dwi_true, data['gtab'])
        # evals_pred = metrics.ols_fit_tensor(dwi_pred, data['gtab'])
        # print_data(metrics.torch_fa(evals_true),
        #           metrics.torch_fa(evals_pred),
        #           mask)

        print_RIS(metrics.torch_RIS(sh_true), metrics.torch_RIS(sh_pred), mask)
        print_diff(sh_true, sh_pred, mask, 'mse', normalize=False)

    def compute_modules(self, inputs_needed, inputs):
        if 'dwi' in inputs_needed and 'dwi' not in inputs.keys():
            inputs['dwi'] = self.modules['dwi'](inputs['sh'],
                                                inputs['mean_b0'])
        if 'dwi_pred' in inputs_needed and 'dwi_pred' not in inputs.keys():
            inputs['dwi_pred'] = self.modules['dwi'](inputs['sh_pred'],
                                                     inputs['mean_b0_pred'])
        if 'fa' in inputs_needed and 'fa' not in inputs.keys():
            inputs['fa'] = self.modules['fa'](inputs['dwi'], inputs['mask'])
        if 'fa_pred' in inputs_needed and 'fa_pred' not in inputs.keys():
            inputs['fa_pred'] = self.modules['fa'](inputs['dwi_pred'],
                                                   inputs['mask'])

        return inputs

    def get_batch_loss(self, inputs):
        """ Single forward and backward pass """

        for input_name in inputs.keys():
            inputs[input_name] = inputs[input_name].to(self.net.device)

        net_pred = self.net(inputs['sh'], inputs['mean_b0'])
        inputs.update(net_pred)

        inputs_needed = [inp for loss in self.losses for inp in loss['inputs']]
        inputs = self.compute_modules(inputs_needed, inputs)

        batch_loss = []
        for loss_d in self.losses:
            loss = loss_d['fun'](*[inputs[name] for name in loss_d['inputs']])
            loss = loss_d['coeff'] * loss
            batch_loss.append(loss)

        batch_loss = torch.stack(batch_loss, dim=0).sum()

        return inputs, batch_loss

    def train(self,
              train_dataset,
              validation_dataset,
              num_epochs,
              batch_size=128,
              validation=True,
              metrics_final=None,
              freq_print=50):

        train_dataset.set_return_site(False)
        validation_dataset.set_return_site(False)

        dataloader_parameters = {
            "num_workers": 0,
            "shuffle": True,
            "pin_memory": True,
            "batch_size": batch_size,
        }
        dataloader_train = DataLoader(train_dataset, **dataloader_parameters)

        if metrics_final is None:
            metrics_final = {
                metric: -np.inf
                for metric in self.metrics
            }
        metrics_epoch = {
            metric: -np.inf
            for metric in self.metrics
        }

        logger_metrics = {metric: [] for metric in metrics_epoch.keys()}

        best_value = metrics_final[self.metric_to_maximize]
        best_net = copy.deepcopy(self.net)
        counter_patience = 0
        last_update = None
        epoch_loss_train = 0
        t = tqdm(range(num_epochs,))
        for epoch, _ in enumerate(t):
            if epoch != 0:
                t.set_postfix(
                    best_metric=best_value,
                    metric_epoch=metrics_epoch,
                    last_update=last_update,
                    loss=epoch_loss_train,
                )

            # Training

            epoch_loss_train = 0.0

            for i, data in tqdm(enumerate(dataloader_train, 0),
                                total=len(train_dataset) // batch_size +
                                int(len(train_dataset) % batch_size != 0),
                                leave=False):
                self.optimizer.zero_grad()

                # Set network to train mode
                self.net.train()

                _, batch_loss = self.get_batch_loss(data)

                epoch_loss_train += batch_loss.item()

                loss = batch_loss
                loss.backward()

                # self.net.plot_grad_flow()

                # gradient descent
                self.optimizer.step()

            epoch_loss_train /= (i + 1)

            # Validation and print
            with torch.no_grad():

                metrics_epoch = self.validate(validation_dataset, batch_size)

                for metric in metrics_epoch.keys():
                    logger_metrics[metric].append(metrics_epoch[metric])

                if epoch % freq_print == 0 and epoch != 0:
                    self.print_metrics(validation_dataset, batch_size)

            if self.save_folder:
                self.net.save(self.save_folder + str(epoch) + "_net.tar.gz")

            if metrics_epoch[self.metric_to_maximize] > best_value:
                best_value = metrics_epoch[self.metric_to_maximize]
                last_update = epoch
                best_net = copy.deepcopy(self.net)
                metrics_final = {
                    metric: metrics_epoch[metric]
                    for metric in self.metrics
                }
                counter_patience = 0
            else:
                counter_patience += 1

            if counter_patience > self.patience:
                break

        for metric in logger_metrics.keys():
            plt.figure()
            plt.title(metric)
            plt.plot(logger_metrics[metric])
        plt.show()

        return best_net, metrics_final
