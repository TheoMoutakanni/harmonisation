import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import copy
from tqdm import tqdm

from harmonisation.functions.metrics import compute_metrics_dataset
from harmonisation.functions.losses import get_loss_fun
from harmonisation.viz import print_peaks, print_acc
from harmonisation.datasets.utils import batch_to_xyz


class BaseTrainer():
    def __init__(self,
                 net,
                 optimizer_parameters={
                     "lr": 0.001,
                     "weight_decay": 1e-8,
                 },
                 loss_specs={
                     "type": "acc",
                     "parameters": {}
                 },
                 metrics=["acc", "mse"],
                 metric_to_maximize="acc",
                 patience=10,
                 save_folder=None,
                 ):
        self.net = net
        self.metric_to_maximize = metric_to_maximize
        self.metrics = metrics
        self.loss = get_loss_fun(loss_specs)
        self.optimizer = optim.Adam(
            self.net.parameters(), **optimizer_parameters)

        self.patience = patience
        self.save_folder = save_folder

    def validate(self, epoch, validation_dataset):
        """
        Compute metrics on validation_dataset and print some metrics
        """

        data_true = {name: validation_dataset.get_data_by_name(name)
                     for name in validation_dataset.names}
        data_pred = self.net.predict_dataset(validation_dataset,
                                             batch_size=128)

        metrics_batches = compute_metrics_dataset(
            data_true, data_pred, self.metrics)

        metrics_epoch = {
            metric: np.nanmean(
                [m[metric] for m in metrics_batches])
            for metric in self.metrics
        }

        # Print fODF and acc for a slice of a validation dwi
        if epoch % 1000 == 0 and epoch != 0:
            print_name = validation_dataset.names[0]
            print_data = validation_dataset.get_data_by_name(print_name)

            sh_true = batch_to_xyz(
                print_data['sh'],
                print_data['number_of_patches']).cpu()
            sh_pred = batch_to_xyz(
                data_pred[print_name],
                print_data['number_of_patches']).cpu()
            b0_true = batch_to_xyz(
                print_data['mask'],
                print_data['number_of_patches']).cpu()

            print(sh_true[50:51, 50:51, 28:29])
            print(sh_pred[50:51, 50:51, 28:29])
            print_acc(sh_true, sh_pred)
            print_peaks(sh_true, b0_true, print_data['gtab'])
            print_peaks(sh_pred, b0_true, print_data['gtab'])

        return metrics_epoch

    def get_batch_loss(self, X, mask, Z=None):
        """ Single forward and backward pass """

        if Z is None:
            # If Z is not already given
            X = X.to(self.net.device)
            Z = self.net(X)

        X = X.to(self.net.device)
        mask = mask.to(self.net.device)
        Z = self.net.forward(X)

        batch_loss = self.loss(X, Z, mask)

        return batch_loss

    def train(self,
              train_dataset,
              validation_dataset,
              num_epochs,
              batch_size=128,
              validation=True,
              metrics_final=None):

        dataloader_parameters = {
            "num_workers": 1,
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
        metrics_train = {
            metric: -np.inf
            for metric in self.metrics
        }

        best_value = metrics_final[self.metric_to_maximize]
        best_net = copy.deepcopy(self.net)
        counter_patience = 0
        last_update = None
        t = tqdm(range(num_epochs,))
        for epoch, _ in enumerate(t):
            if epoch != 0:
                t.set_postfix(
                    best_metric=best_value,
                    metric_epoch=metrics_epoch,
                    last_update=last_update,
                    metrics_train=metrics_train[self.metric_to_maximize]
                )

            epoch_loss_train = 0.0

            with torch.autograd.detect_anomaly():
                for i, data in enumerate(dataloader_train, 0):

                    self.optimizer.zero_grad()

                    # Set network to train mode
                    self.net.train()

                    X, mask = data
                    batch_loss = self.get_batch_loss(X, mask)

                    epoch_loss_train += batch_loss.item()

                    loss = batch_loss
                    loss.backward()

                    # self.net.plot_grad_flow()

                    # gradient descent
                    self.optimizer.step()

            epoch_loss_train /= (i + 1)

            with torch.no_grad():

                metrics_epoch = self.validate(epoch, validation_dataset)
                metrics_train = self.validate(epoch, train_dataset)

            if self.save_folder:
                self.net.save(self.save_folder + str(epoch) + "_net")

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

        return best_net, metrics_final
