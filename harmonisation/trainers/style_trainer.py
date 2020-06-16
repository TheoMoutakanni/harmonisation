import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import copy
from tqdm import tqdm

import matplotlib.pyplot as plt

from harmonisation.functions import metrics
from harmonisation.functions.losses import get_loss_fun
from harmonisation.datasets import AdversarialDataset

from .trainer import BaseTrainer


class StyleTrainer(BaseTrainer):
    def __init__(self,
                 net,
                 feat_net,
                 optimizer_parameters={
                     "autoencoder": {
                         "lr": 0.01,
                         "weight_decay": 1e-8,
                     },
                     "features": {
                         "lr": 0.01,
                         "weight_decay": 1e-8,
                     }
                 },
                 loss_specs={
                     "autoencoder": [
                         {
                             "coeff": 1.,
                             "type": "mse",
                             "parameters": {}
                         },
                     ],
                     "style": [],
                     "features": [
                         {
                             "coeff": 1.,
                             "type": "bce",
                             "parameters": {}
                         }
                     ]
                 },
                 metrics={
                     "autoencoder": ["acc", "mse"],
                     "features": ["accuracy"]
                 },
                 metric_to_maximize={
                     "autoencoder": "acc",
                     "features": "accuracy"
                 },
                 patience=10,
                 save_folder=None,
                 ):
        self.net = net
        self.feat_net = feat_net

        self.metric_to_maximize = metric_to_maximize["autoencoder"]
        self.metrics = metrics["autoencoder"]
        self.loss = get_loss_fun(loss_specs["autoencoder"])
        self.style_loss = get_loss_fun(loss_specs["style"])
        self.optimizer = optim.Adam(
            self.net.parameters(), **optimizer_parameters["autoencoder"])

        self.feat_metric_to_maximize = metric_to_maximize["features"]
        self.feat_metrics = metrics["features"]
        self.feat_loss = get_loss_fun(loss_specs["features"])
        self.feat_optimizer = optim.Adam(
            self.feat_net.parameters(), **optimizer_parameters["features"])

        self.patience = patience
        self.save_folder = save_folder

    def set_style_loss(self, loss_specs):
        self.style_loss = get_loss_fun(loss_specs)

    def validate_feat(self, dataset, batch_size=128, adversarial=True):
        """
        Compute metrics on validation_dataset and print some metrics
        """
        if adversarial:
            feat_dataset = AdversarialDataset(dataset, self.net)
        else:
            feat_dataset = dataset

        data_true = {name: feat_dataset.get_data_by_name(name)
                     for name in feat_dataset.names}
        data_pred = self.feat_net.predict_dataset(feat_dataset,
                                                  batch_size=128)

        metrics_fun = metrics.get_metrics_fun()

        metrics_batches = []
        for name in list(set(data_pred.keys()) & set(data_true.keys())):
            dic = dict()
            for metric in self.feat_metrics:
                dic[metric] = np.nanmean(
                    metrics_fun[metric](
                        torch.FloatTensor([data_true[name]['site']]),
                        torch.FloatTensor(data_pred[name]['out'])
                    ))

            metrics_batches.append(dic)

        metrics_epoch = {
            metric: np.nanmean(
                [m[metric] for m in metrics_batches])
            for metric in self.feat_metrics
        }

        return metrics_epoch

    def get_batch_feat_loss(self, data, Z=None, coeff_fake=0.):
        """ Get the loss for the adversarial network
        Single forward and backward pass """
        X, mask, mean_b0, y = data
        X = X.to(self.net.device)
        y = y.to(self.net.device)
        mean_b0 = mean_b0.to(self.net.device)

        batch_loss = 0

        if coeff_fake < 1:
            real_y = self.feat_net.forward(X, mean_b0)['out']
            batch_loss_real = self.feat_loss(real_y, y.squeeze())

            batch_loss += (1 - coeff_fake) * batch_loss_real

        if coeff_fake > 0:
            if Z is None:
                Z = self.net(X)
            fake_y = self.feat_net.forward(Z.detach(), mean_b0)['out']
            batch_loss_fake = self.feat_loss(fake_y, y.squeeze())

            batch_loss += coeff_fake * batch_loss_fake

        return Z, batch_loss

    def get_batch_loss(self, data):
        """ Get the loss + adversarial loss for the autoencoder
        Single forward and backward pass """

        X, mask, mean_b0 = data
        X = X.to(self.net.device)
        mask = mask.to(self.net.device)
        mean_b0 = mean_b0.to(self.net.device)

        Z = self.net(X)

        batch_loss_reconst = self.loss(X, Z, mask)

        pred_y = self.feat_net.forward(Z, mean_b0)

        batch_loss_style = self.style_loss(pred_y)
        print(batch_loss_reconst, batch_loss_style)
        batch_loss = batch_loss_reconst + batch_loss_style

        return batch_loss

    def train_feat_net(self,
                       train_dataset,
                       validation_dataset,
                       num_epochs,
                       coeff_fake=0.,
                       batch_size=128,
                       validation=True,
                       metrics_final=None):

        train_dataset.set_return_site(True)
        validation_dataset.set_return_site(True)

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
                for metric in self.feat_metrics
            }
        metrics_epoch = {
            metric: -np.inf
            for metric in self.feat_metrics
        }

        metrics_train_epoch = {
            metric: -np.inf
            for metric in self.feat_metrics
        }

        best_value = metrics_final[self.feat_metric_to_maximize]
        best_net = copy.deepcopy(self.feat_net)
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
                    metrics_train=metrics_train_epoch,
                )

            epoch_loss_train = 0.0

            for i, data in enumerate(dataloader_train, 0):

                self.feat_optimizer.zero_grad()

                # Set network to train mode
                self.feat_net.train()

                _, batch_loss = self.get_batch_feat_loss(
                    data, coeff_fake=coeff_fake)

                epoch_loss_train += batch_loss.item()

                loss = batch_loss
                loss.backward()

                # self.net.plot_grad_flow()

                # gradient descent
                self.feat_optimizer.step()

            epoch_loss_train /= (i + 1)

            with torch.no_grad():
                metrics_epoch = self.validate_feat(validation_dataset,
                                                   batch_size,
                                                   adversarial=False)
                metrics_train_epoch = self.validate_feat(train_dataset,
                                                         batch_size,
                                                         adversarial=False)

            if self.save_folder:
                file_name = self.save_folder + str(epoch) + "_featnet.tar.gz"
                self.feat_net.save(file_name)

            if metrics_epoch[self.feat_metric_to_maximize] > best_value:
                best_value = metrics_epoch[self.feat_metric_to_maximize]
                last_update = epoch
                best_net = copy.deepcopy(self.feat_net)
                metrics_final = {
                    metric: metrics_epoch[metric]
                    for metric in self.feat_metrics
                }
                counter_patience = 0
            else:
                counter_patience += 1

            if counter_patience > self.patience:
                break

        return best_net, metrics_final
