import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import numpy as np
import copy
from tqdm import tqdm

import matplotlib.pyplot as plt

from harmonisation.functions.metrics import get_metric_fun
from harmonisation.functions.losses import get_loss_fun
from harmonisation.datasets import AdversarialDataset

from .trainer import BaseTrainer


class StyleTrainer(BaseTrainer):
    def __init__(self,
                 net,
                 dict_adversarial_net,
                 modules={},
                 optimizer_parameters={
                     "autoencoder": {
                         "lr": 0.01,
                         "weight_decay": 1e-8,
                     },
                     "aversarial": {
                         "lr": 0.01,
                         "weight_decay": 1e-8,
                     }
                 },
                 scheduler_parameters={
                     "autoencoder": {
                         "base_lr": 1e-3,
                         "max_lr": 1e-2,
                         "step_size_up": 2000,
                     },
                     "aversarial": {
                         "base_lr": 1e-3,
                         "max_lr": 1e-2,
                         "step_size_up": 2000,
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
                     "adversarial": [
                         {
                             "coeff": 1.,
                             "type": "bce",
                             "parameters": {}
                         }
                     ]
                 },
                 metrics_specs={
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
        self.adversarial_net = dict_adversarial_net

        self.metric_to_maximize = metric_to_maximize["autoencoder"]
        self.metrics = metrics_specs["autoencoder"]
        self.losses = get_loss_fun(loss_specs["autoencoder"], self.net.device)
        self.style_losses = get_loss_fun(loss_specs["style"], self.net.device)
        self.optimizer = optim.Adam(
            self.net.parameters(), **optimizer_parameters["autoencoder"])
        # self.scheduler = optim.lr_scheduler.CyclicLR(
        #     self.optimizer, **scheduler_parameters["autoencoder"])

        self.adv_metric_to_maximize = metric_to_maximize["adversarial"]
        self.adversarial_metrics = {
            name: get_metric_fun(metric_d)
            for name, metric_d in metrics_specs["adversarial"].items()}
        self.adversarial_losses = {
            name: get_loss_fun(loss_d, self.net.device)
            for name, loss_d in loss_specs["adversarial"].items()}
        self.adversarial_optimizer = {
            name: optim.SGD(self.adversarial_net[name].parameters(), **params)
            for name, params in optimizer_parameters["adversarial"].items()}
        self.adversarial_scheduler = {
            name: optim.lr_scheduler.CyclicLR(
                self.adversarial_optimizer[name], **params)
            for name, params in scheduler_parameters["adversarial"].items()
        }

        self.modules = modules

        self.patience = patience
        self.save_folder = save_folder
        self.writer_path = self.save_folder + '/data/'
        self.writer = SummaryWriter(logdir=self.writer_path)

    def set_style_loss(self, loss_specs):
        self.style_losses = get_loss_fun(loss_specs, self.net.device)

    def set_adversarial_loss(self, loss_specs_by_net):
        for net_name, loss_specs in loss_specs_by_net.items():
            self.adversarial_losses[net_name] = get_loss_fun(
                loss_specs, self.net.device)

    def set_loss(self, loss_specs):
        self.losses = get_loss_fun(loss_specs, self.net.device)

    def set_adversarial_metric(self, metric_specs_by_net):
        for net_name, metric_specs in metric_specs_by_net.items():
            self.adversarial_metrics[net_name] = get_metric_fun(
                metric_specs)

    def validate_adversarial(self, net_name, dataset,
                             batch_size=128, force_use_fake=False):
        """
        Compute metrics on validation_dataset and print some metrics
        """
        use_fake = any(self.adversarial_metrics[net_name][name]["use_fake"]
                       for name in self.adversarial_metrics[net_name])
        use_fake = use_fake or force_use_fake
        if use_fake:
            feat_dataset = AdversarialDataset(dataset, self.net,
                                              batch_size=batch_size)
        else:
            feat_dataset = dataset

        data = {name: feat_dataset.get_data_by_name(name)
                for name in feat_dataset.names}

        net_pred = self.adversarial_net[net_name].predict_dataset(
            feat_dataset,
            batch_size=batch_size)

        metrics_batches = []
        for name in list(set(net_pred.keys()) & set(data.keys())):
            dic = dict()
            inputs = {}
            inputs.update(net_pred[name])
            inputs.update(data[name])
            for metric, metric_d in self.adversarial_metrics[net_name].items():
                dic[metric] = np.nanmean(
                    metric_d["fun"](
                        *[torch.FloatTensor(inputs[name])
                          for name in metric_d['inputs']]
                    )
                )

            metrics_batches.append(dic)

        metrics_epoch = {
            metric: np.nanmean(
                [m[metric] for m in metrics_batches])
            for metric in self.adversarial_metrics[net_name]
        }

        return metrics_epoch

    def get_batch_adversarial_loss(self, net, inputs, losses):
        """ Get the loss for the adversarial network
        Single forward and backward pass """
        #inputs = inputs.copy()

        if 'site' in inputs.keys():
            inputs['site'] = inputs['site'].squeeze(-1)

        for input_name in inputs.keys():
            inputs[input_name] = inputs[input_name].to(self.net.device)

        inputs_needed = [inp for loss in losses
                         for inp in loss['inputs']]

        if 'y_logits' in inputs_needed:
            # If we need y_logits, we compute it
            feat_net_pred_real = net.forward(
                inputs['sh'],
                inputs['mean_b0'],
                inputs['mask'])

            inputs.update(feat_net_pred_real)

        if 'y_logits_fake' in inputs_needed:
            # If we need y_logits_pred, we compute it
            if 'sh_fake' not in inputs.keys():
                inputs.update(self.net(inputs['sh'], inputs['mean_b0']))
            feat_net_pred_fake = net.forward(
                inputs['sh_fake'].detach(),
                inputs['mean_b0_fake'].detach(),
                inputs['mask'])

            inputs.update({k + '_fake': v
                           for k, v in feat_net_pred_fake.items()})

        loss_dict = {}

        batch_loss = []
        for loss_d in losses:
            loss = loss_d['fun'](*[inputs[name] for name in loss_d['inputs']])
            loss = loss_d['coeff'] * loss
            batch_loss.append(loss)
            loss_dict[loss_d['type'] + '_' + loss_d['inputs'][0]] = loss
        batch_loss = torch.stack(batch_loss, dim=0).sum()

        return inputs, {'batch_loss': batch_loss}

    def get_batch_loss(self, inputs):
        """ Get the loss + adversarial loss for the autoencoder
        Single forward and backward pass """
        #inputs = inputs.copy()

        if 'site' in inputs.keys():
            inputs['site'] = inputs['site'].squeeze(-1)

        for input_name in inputs.keys():
            inputs[input_name] = inputs[input_name].to(self.net.device)

        net_pred = self.net(inputs['sh'], inputs['mean_b0'])
        inputs.update(net_pred)

        inputs_needed = [inp for loss in self.losses + self.style_losses
                         for inp in loss['inputs']]
        inputs = self.compute_modules(inputs_needed, inputs)

        for net_name, adv_net in self.adversarial_net.items():
            if not any(net_name in s for s in inputs_needed):
                # If we do not need the output of that network
                continue

            feat_net_pred = adv_net.forward(inputs['sh_fake'],
                                            inputs['mean_b0_fake'],
                                            inputs['mask'])

            # Add the name of the network to the keys
            inputs.update({k + '_fake_' + net_name: v
                           for k, v in feat_net_pred.items()})

        loss_dict = {}

        batch_loss_reconst = []
        for loss_d in self.losses:
            loss = loss_d['fun'](*[inputs[name] for name in loss_d['inputs']])
            loss = loss_d['coeff'] * loss
            batch_loss_reconst.append(loss)
            loss_dict[loss_d['type'] + '_' + loss_d['inputs'][0]] = loss
        batch_loss_reconst = torch.stack(batch_loss_reconst, dim=0).sum()

        loss_dict['reconst_loss'] = batch_loss_reconst

        batch_loss_style = []
        for loss_d in self.style_losses:
            loss = loss_d['fun'](*[inputs[name] for name in loss_d['inputs']])
            loss = loss_d['coeff'] * loss
            batch_loss_style.append(loss)
            loss_dict[loss_d['type'] + '_' + loss_d['inputs'][0]] = loss
        if len(self.style_losses) != 0:
            batch_loss_style = torch.stack(batch_loss_style, dim=0).sum()
            loss_dict['style_loss'] = batch_loss_style
        else:
            batch_loss_style = 0

        batch_loss = batch_loss_reconst + batch_loss_style
        loss_dict['batch_loss'] = batch_loss

        return inputs, loss_dict

    def train_adversarial_net(self,
                              nets_list,
                              train_dataset,
                              validation_dataset,
                              num_epochs,
                              train_net_X_time=0,
                              batch_size=128,
                              validation=True,
                              metrics_final=None,
                              keep_best_net=True):

        dataloader_parameters = {
            "num_workers": 1,
            "shuffle": True,
            "pin_memory": True,
            "batch_size": batch_size,
        }
        dataloader_train = DataLoader(train_dataset, **dataloader_parameters)

        if metrics_final is None:
            metrics_final = {
                net_name: {metric: -np.inf
                           for metric in self.adversarial_metrics[net_name]}
                for net_name in nets_list}
            if train_net_X_time > 0:
                metrics_final['autoencoder'] = {metric: -np.inf
                                                for metric in self.metrics}
        metrics_epoch = {net_name:
                         {metric: -np.inf
                          for metric in self.adversarial_metrics[net_name]}
                         for net_name in nets_list}

        metrics_train_epoch = {
            net_name: {metric: -np.inf
                       for metric in self.adversarial_metrics[net_name]}
            for net_name in nets_list}

        best_value = {
            net_name:
            metrics_final[net_name][self.adv_metric_to_maximize[net_name]]
            for net_name in nets_list
        }
        best_net = {net_name: copy.deepcopy(self.adversarial_net[net_name])
                    for net_name in nets_list}

        if train_net_X_time > 0:
            metrics_epoch['autoencoder'] = {metric: -np.inf
                                            for metric in self.metrics}
            # metrics_train_epoch['autoencoder'] = {metric: -np.inf
            #                                       for metric in self.metrics}
            best_value['autoencoder'] = metrics_final['autoencoder'][
                self.metric_to_maximize]
            best_net['autoencoder'] = copy.deepcopy(self.net)

        counter_patience = 0
        last_update = None
        t = tqdm(range(num_epochs,))
        for epoch, _ in enumerate(t):
            if epoch != 0:
                t.set_postfix(
                    best_metric=best_value,
                    last_update=last_update,
                    metrics_val=metrics_epoch,
                    # metrics_train=metrics_train_epoch,
                )

            epoch_loss_train = {net_name: {} for net_name in nets_list}

            if train_net_X_time > 0:
                epoch_loss_train['autoencoder'] = {}

            for i, data in enumerate(dataloader_train, 0):

                inputs = data

                # Train autoencoder if train_net_X_time > 0
                for _ in range(train_net_X_time):
                    self.optimizer.zero_grad()
                    self.net.train()
                    _, batch_losses = self.get_batch_loss(data)
                    loss = batch_losses['batch_loss']
                    loss.backward()
                    self.optimizer.step()
                    # self.scheduler.step()
                    for name, loss in batch_losses.items():
                        loss = loss.item()
                        loss /= float(train_net_X_time)
                        loss += epoch_loss_train['autoencoder'].setdefault(
                            name, 0)
                        epoch_loss_train['autoencoder'][name] = loss
                    del batch_losses

                # Train adversarial networks
                for net_name in nets_list:
                    self.adversarial_optimizer[net_name].zero_grad()

                    self.adversarial_net[net_name].train()

                    inputs, batch_losses = self.get_batch_adversarial_loss(
                        self.adversarial_net[net_name],
                        inputs,
                        self.adversarial_losses[net_name])

                    loss = batch_losses['batch_loss']
                    loss.backward()
                    self.adversarial_optimizer[net_name].step()
                    self.adversarial_scheduler[net_name].step()

                    for name, loss in batch_losses.items():
                        loss = loss.item()
                        loss += epoch_loss_train[net_name].setdefault(
                            name, 0)
                        epoch_loss_train[net_name][name] = loss
                    del batch_losses

                # self.net.plot_grad_flow()

            for net_name in epoch_loss_train.keys():
                for name, loss in epoch_loss_train[net_name].items():
                    self.writer.add_scalar('loss/' + net_name + '/' + name,
                                           loss / (i + 1.), epoch)
                    epoch_loss_train[net_name][name] = loss / (i + 1.)

            with torch.no_grad():
                if train_net_X_time > 0:
                    metrics_epoch['autoencoder'] = self.validate(
                        validation_dataset, batch_size)
                    for name, metric in metrics_epoch['autoencoder'].items():
                        self.writer.add_scalar('metric/autoencoder/' + name + '_val',
                                               metric, epoch)

                for net_name in nets_list:
                    metrics_epoch[net_name] = self.validate_adversarial(
                        net_name,
                        validation_dataset,
                        batch_size=batch_size,
                        force_use_fake=False)

                    # metrics_train_epoch[net_name] = self.validate_adversarial(
                    #     net_name,
                    #     train_dataset,
                    #     batch_size=batch_size,
                    #     force_use_fake=False)

                    for name, metric in metrics_epoch[net_name].items():
                        self.writer.add_scalar('metric/' + net_name + '/' + name + '_val',
                                               metric, epoch)
                    # for name, metric in metrics_train_epoch[net_name].items():
                    #     self.writer.add_scalar('metric/' + net_name + '/' + name + '_train',
                    #                            metric, epoch)

            if self.save_folder:
                if train_net_X_time > 0:
                    file_name = self.save_folder + 'autoencoder'
                    file_name = file_name + "_" + str(epoch) + ".tar.gz"
                    self.net.save(file_name)
                for net_name in nets_list:
                    file_name = self.save_folder + net_name
                    file_name = file_name + "_" + str(epoch) + ".tar.gz"
                    self.adversarial_net[net_name].save(file_name)

            if train_net_X_time > 0:
                """
                If we train the autoencoder, each adversarial network
                is saved at the same time.
                """
                metric_name = self.metric_to_maximize
                actual_value = metrics_epoch['autoencoder'][metric_name]
                if actual_value > best_value['autoencoder']:
                    best_value['autoencoder'] = actual_value
                    last_update = epoch
                    metrics_final = metrics_epoch
                    best_net['autoencoder'] = copy.deepcopy(self.net)
                    for net_name in nets_list:
                        best_net[net_name] = copy.deepcopy(
                            self.adversarial_net[net_name])

            else:
                """
                Else, each adversarial  network is saved when its best metric
                is reached.
                """
                for net_name in nets_list:
                    metric_name = self.adv_metric_to_maximize[net_name]
                    actual_value = metrics_epoch[net_name][metric_name]
                    if actual_value > best_value[net_name]:
                        best_value[net_name] = actual_value
                        last_update = epoch
                        best_net[net_name] = copy.deepcopy(
                            self.adversarial_net[net_name])
                        metrics_final[net_name] = metrics_epoch[net_name]

            if last_update == epoch:
                counter_patience = 0
            else:
                counter_patience += 1

            if counter_patience > self.patience:
                break

        if keep_best_net:
            if train_net_X_time > 0:
                self.net = best_net['autoencoder']

            for net_name in nets_list:
                self.adversarial_net[net_name] = best_net[net_name]

        return best_net, metrics_final
