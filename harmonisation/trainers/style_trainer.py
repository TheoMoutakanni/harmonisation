import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

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
                 modules={},
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
        self.losses = get_loss_fun(loss_specs["autoencoder"], self.net.device)
        self.style_losses = get_loss_fun(loss_specs["style"], self.net.device)
        self.optimizer = optim.Adam(
            self.net.parameters(), **optimizer_parameters["autoencoder"])

        self.feat_metric_to_maximize = metric_to_maximize["features"]
        self.feat_metrics = metrics["features"]
        self.feat_losses = get_loss_fun(loss_specs["features"],
                                        self.net.device)
        self.feat_optimizer = optim.Adam(
            self.feat_net.parameters(), **optimizer_parameters["features"])

        self.modules = modules

        self.patience = patience
        self.save_folder = save_folder
        self.writer_path = self.save_folder + '/data/'
        self.writer = SummaryWriter(logdir=self.writer_path)

    def set_style_loss(self, loss_specs):
        self.style_losses = get_loss_fun(loss_specs, self.net.device)

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
                        torch.FloatTensor(data_pred[name]['y_proba'])
                    ))

            metrics_batches.append(dic)

        metrics_epoch = {
            metric: np.nanmean(
                [m[metric] for m in metrics_batches])
            for metric in self.feat_metrics
        }

        return metrics_epoch

    def get_batch_feat_loss(self, inputs, coeff_fake=0.):
        """ Get the loss for the adversarial network
        Single forward and backward pass """
        for input_name in inputs.keys():
            inputs[input_name] = inputs[input_name].to(self.net.device)

        batch_loss = 0

        if coeff_fake < 1:
            feat_net_pred = self.feat_net.forward(
                inputs['sh'],
                inputs['mean_b0'],
                inputs['mask'])
            batch_loss_real = self.feat_losses[0]['fun'](
                feat_net_pred['y_proba'],
                inputs['site'].squeeze(-1))

            batch_loss += (1 - coeff_fake) * batch_loss_real

        if coeff_fake > 0:
            if 'sh_pred' not in inputs.keys():
                inputs.update(self.net(inputs['sh'], inputs['mean_b0']))
            feat_net_pred = self.feat_net.forward(
                inputs['sh_pred'].detach(),
                inputs['mean_b0_pred'].detach(),
                inputs['mask'])
            batch_loss_fake = self.feat_losses[0]['fun'](
                feat_net_pred['y_proba'],
                inputs['site'].squeeze(-1))

            batch_loss += coeff_fake * batch_loss_fake

        return inputs, {'batch_loss': batch_loss}

    def get_batch_loss(self, inputs):
        """ Get the loss + adversarial loss for the autoencoder
        Single forward and backward pass """

        loss_dict = {}

        for input_name in inputs.keys():
            inputs[input_name] = inputs[input_name].to(self.net.device)

        net_pred = self.net(inputs['sh'], inputs['mean_b0'])
        inputs.update(net_pred)

        feat_net_pred = self.feat_net.forward(inputs['sh_pred'],
                                              inputs['mean_b0_pred'],
                                              inputs['mask'])
        inputs.update(feat_net_pred)

        inputs_needed = [inp for loss in self.losses + self.style_losses
                         for inp in loss['inputs']]
        inputs = self.compute_modules(inputs_needed, inputs)

        batch_loss_reconst = []
        for loss_d in self.losses:
            loss = loss_d['fun'](*[inputs[name] for name in loss_d['inputs']])
            loss = loss_d['coeff'] * loss
            batch_loss_reconst.append(loss)
            loss_dict[loss_d['type'] + '_' + loss_d['inputs'][0]] = loss
        batch_loss_reconst = torch.stack(batch_loss_reconst, dim=0).sum()

        batch_loss_style = []
        for loss_d in self.style_losses:
            loss = loss_d['fun'](*[inputs[name] for name in loss_d['inputs']])
            loss = loss_d['coeff'] * loss
            batch_loss_style.append(loss)
            loss_dict[loss_d['type'] + '_' + loss_d['inputs'][0]] = loss
        batch_loss_style = torch.stack(batch_loss_style, dim=0).sum()

        batch_loss = batch_loss_reconst + batch_loss_style

        loss_dict.update({'batch_loss': batch_loss,
                          'reconst_loss': batch_loss_reconst,
                          'style_loss': batch_loss_style})

        return inputs, loss_dict

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

        logger_metrics = {metric: [] for metric in metrics_epoch.keys()}

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
                    loss=epoch_loss_train['batch_loss'],
                    last_update=last_update,
                    metrics_val=metrics_epoch,
                    metrics_train=metrics_train_epoch,
                )

            epoch_loss_train = {}

            for i, data in enumerate(dataloader_train, 0):

                self.feat_optimizer.zero_grad()

                # Set network to train mode
                self.feat_net.train()

                _, batch_losses = self.get_batch_feat_loss(
                    data, coeff_fake=coeff_fake)

                loss = batch_losses['batch_loss']
                loss.backward()

                for name, loss in batch_losses.items():
                    loss = epoch_loss_train.setdefault(name, 0) + loss.item()
                    epoch_loss_train[name] = loss

                # self.net.plot_grad_flow()

                # gradient descent
                self.feat_optimizer.step()

            for name, loss in epoch_loss_train.items():
                self.writer.add_scalar('feat_loss/' + name,
                                       loss / (i + 1), epoch)
                epoch_loss_train[name] = loss / (i + 1)

            with torch.no_grad():
                metrics_epoch = self.validate_feat(validation_dataset,
                                                   batch_size,
                                                   adversarial=False)
                for metric in metrics_epoch.keys():
                    logger_metrics[metric].append(metrics_epoch[metric])

                metrics_train_epoch = self.validate_feat(train_dataset,
                                                         batch_size,
                                                         adversarial=False)

            for name, metric in metrics_epoch.items():
                self.writer.add_scalar('feat_metric/' + name + '_val',
                                       metric, epoch)
            for name, metric in metrics_train_epoch.items():
                self.writer.add_scalar('feat_metric/' + name + '_train',
                                       metric, epoch)

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

        # for metric in logger_metrics.keys():
        #     plt.figure()
        #     plt.title(metric)
        #     plt.plot(logger_metrics[metric])
        # plt.show()

        return best_net, metrics_final
