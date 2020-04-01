import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import copy
from tqdm import tqdm

from harmonisation.functions.metrics import compute_metrics_dataset
from harmonisation.functions.losses import get_loss_fun
from harmonisation.datasets import AdversarialDataset

from .trainer import BaseTrainer


class AdversarialTrainer(BaseTrainer):
    def __init__(self,
                 net,
                 adv_net,
                 optimizer_parameters={
                     "autoencoder": {
                         "lr": 0.01,
                         "weight_decay": 1e-8,
                     },
                     "adversarial": {
                         "lr": 0.01,
                         "weight_decay": 1e-8,
                     }
                 },
                 loss_specs={
                     "autoencoder": {
                         "type": "mse",
                         "parameters": {}
                     },
                     "adversarial": {
                         "type": "bce",
                         "parameters": {}
                     }
                 },
                 metrics={
                     "autoencoder": ["acc", "mse"],
                     "adversarial": ["accuracy"]
                 },
                 metric_to_maximize={
                     "autoencoder": "acc",
                     "adversarial": "accuracy"
                 },
                 patience=10,
                 save_folder=None,
                 ):
        self.net = net
        self.adv_net = adv_net

        self.metric_to_maximize = metric_to_maximize["autoencoder"]
        self.metrics = metrics["autoencoder"]
        self.loss = get_loss_fun(loss_specs["autoencoder"])
        self.optimizer = optim.Adam(
            self.net.parameters(), **optimizer_parameters["autoencoder"])

        self.adv_metric_to_maximize = metric_to_maximize["adversarial"]
        self.adv_metrics = metrics["adversarial"]
        self.adv_loss = get_loss_fun(loss_specs["adversarial"])
        self.adv_optimizer = optim.Adam(
            self.adv_net.parameters(), **optimizer_parameters["adversarial"])

        self.patience = patience
        self.save_folder = save_folder

    def validate_adv(self, epoch, dataset):
        """
        Compute metrics on validation_dataset and print some metrics
        """

        adv_dataset = AdversarialDataset(dataset, self.net)

        data_true = {name: {'sh': adv_dataset.get_data_by_name(name)['label'],
                            'mask': None}
                     for name in adv_dataset.names}
        data_pred = self.adv_net.predict_dataset(adv_dataset,
                                                 batch_size=128)

        metrics_batches = compute_metrics_dataset(
            data_true, data_pred, self.adv_metrics)

        metrics_epoch = {
            metric: np.nanmean(
                [m[metric] for m in metrics_batches])
            for metric in self.adv_metrics
        }

        return metrics_epoch

    def get_batch_adv_loss(self, X):
        """ Get the loss for the adversarial network
        Single forward and backward pass """
        X = X.to(self.net.device)
        Z = self.net(X)
        real_labels = torch.ones(Z.shape[0]).unsqueeze(1).to(self.net.device)
        fake_labels = torch.zeros(Z.shape[0]).unsqueeze(1).to(self.net.device)

        real_proba = self.adv_net.forward(X)
        fake_proba = self.adv_net.forward(Z.detach())

        batch_loss_real = self.adv_loss(real_proba, real_labels)
        batch_loss_fake = self.adv_loss(fake_proba, fake_labels)

        batch_loss = batch_loss_real + batch_loss_fake

        return Z, batch_loss

    def get_batch_loss(self, X, mask, Z=None):
        """ Get the loss + adversarial loss for the autoencoder
        Single forward and backward pass """
        X = X.to(self.net.device)
        mask = mask.to(self.net.device)

        if Z is None:
            # If Z is not already given
            Z = self.net(X)

        batch_loss_reconst = self.loss(X, Z, mask)
        labels = torch.zeros(Z.shape[0]).unsqueeze(1).to(self.adv_net.device)

        proba = self.adv_net.forward(Z)
        batch_loss_adv = self.adv_loss(proba, labels)

        batch_loss = 0.1 * batch_loss_reconst + batch_loss_adv

        return batch_loss

    def train_adv_net(self,
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
                for metric in self.adv_metrics
            }
        metrics_epoch = {
            metric: -np.inf
            for metric in self.adv_metrics
        }
        metrics_train = {
            metric: -np.inf
            for metric in self.adv_metrics
        }

        best_value = metrics_final[self.adv_metric_to_maximize]
        best_net = copy.deepcopy(self.adv_net)
        counter_patience = 0
        last_update = None
        epoch_loss_train = 0
        t = tqdm(range(num_epochs,))
        for epoch, _ in enumerate(t):
            if epoch != 0:
                t.set_postfix(
                    best_metric=best_value,
                    loss=epoch_loss_train,
                    last_update=last_update,
                    metrics_val=metrics_epoch,
                    metrics_train=metrics_train
                )

            epoch_loss_train = 0.0

            with torch.autograd.detect_anomaly():
                for i, data in enumerate(dataloader_train, 0):

                    self.adv_optimizer.zero_grad()

                    # Set network to train mode
                    self.adv_net.train()

                    X, mask = data
                    _, batch_loss = self.get_batch_adv_loss(X)

                    epoch_loss_train += batch_loss.item()

                    loss = batch_loss
                    loss.backward()

                    # self.net.plot_grad_flow()

                    # gradient descent
                    self.adv_optimizer.step()

            epoch_loss_train /= (i + 1)

            with torch.no_grad():
                metrics_epoch = self.validate_adv(epoch, validation_dataset)
                metrics_train = self.validate_adv(epoch, train_dataset)

            if self.save_folder:
                self.adv_net.save(self.save_folder + str(epoch) + "_advnet")

            if metrics_epoch[self.adv_metric_to_maximize] > best_value:
                best_value = metrics_epoch[self.adv_metric_to_maximize]
                last_update = epoch
                best_net = copy.deepcopy(self.adv_net)
                metrics_final = {
                    metric: metrics_epoch[metric]
                    for metric in self.adv_metrics
                }
                counter_patience = 0
            else:
                counter_patience += 1

            if counter_patience > self.patience:
                break

        return best_net, metrics_final

    def train_both(self,
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
        metrics_epoch_adv = {
            metric: -np.inf
            for metric in self.metrics
        }

        best_value = metrics_final[self.metric_to_maximize]
        best_net = copy.deepcopy(self.net)
        best_adv_net = copy.deepcopy(self.adv_net)
        counter_patience = 0
        last_update = None
        epoch_loss_train = 0
        t = tqdm(range(num_epochs,))
        for epoch, _ in enumerate(t):
            if epoch != 0:
                t.set_postfix(
                    best_metric=best_value,
                    loss=epoch_loss_train,
                    last_update=last_update,
                    metrics_val=metrics_epoch,
                    metrics_val_adv=metrics_epoch_adv
                )

            epoch_loss_train_adv = 0.0
            epoch_loss_train = 0.0

            with torch.autograd.detect_anomaly():
                for i, data in enumerate(dataloader_train, 0):
                    # TRain adversarial network

                    self.adv_optimizer.zero_grad()

                    self.adv_net.train()

                    X, mask = data
                    Z, batch_loss_classif = self.get_batch_adv_loss(X)

                    epoch_loss_train_adv += batch_loss_classif.item()

                    batch_loss_classif.backward()
                    self.adv_optimizer.step()

                    # Train autoencoder

                    self.optimizer.zero_grad()

                    self.net.train()

                    batch_loss_autoencoder = self.get_batch_loss(
                        X, mask, Z=Z)

                    epoch_loss_train += batch_loss_autoencoder.item()

                    batch_loss_autoencoder.backward()
                    self.optimizer.step()

                    # self.net.plot_grad_flow()

            epoch_loss_train_adv /= (i + 1)
            epoch_loss_train /= (i + 1)

            with torch.no_grad():
                metrics_epoch = self.validate(epoch, validation_dataset)
                metrics_epoch_adv = self.validate_adv(
                    epoch, validation_dataset)

            if self.save_folder:
                self.adv_net.save(self.save_folder + str(epoch) + "_advnet")

            if metrics_epoch[self.metric_to_maximize] > best_value:
                best_value = metrics_epoch[self.metric_to_maximize]
                last_update = epoch
                best_net = copy.deepcopy(self.net)
                best_adv_net = copy.deepcopy(self.adv_net)
                metrics_final = {
                    metric: metrics_epoch[metric]
                    for metric in self.metrics
                }
                counter_patience = 0
            else:
                counter_patience += 1

            if counter_patience > self.patience:
                break

        return best_net, best_adv_net, metrics_final
