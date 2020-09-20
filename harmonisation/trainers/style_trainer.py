import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import numpy as np
import copy
from tqdm import tqdm

from harmonisation.functions.metrics import get_metric_fun
from harmonisation.functions.losses import get_loss_fun
from harmonisation.datasets import AdversarialDataset

from harmonisation.utils import compute_modules

from .trainer import BaseTrainer, Iterator

import torch.autograd.profiler as profiler


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
        self.metrics = get_metric_fun(metrics_specs["autoencoder"], "cpu")
        self.losses = get_loss_fun(loss_specs["autoencoder"], self.net.device)
        self.style_losses = get_loss_fun(loss_specs["style"], self.net.device)
        self.optimizer = optim.Adam(
            self.net.parameters(), **optimizer_parameters["autoencoder"])
        self.scheduler = optim.lr_scheduler.CyclicLR(
            self.optimizer, **scheduler_parameters["autoencoder"])
        self.scaler = torch.cuda.amp.GradScaler()

        self.adv_metric_to_maximize = metric_to_maximize["adversarial"]
        self.adversarial_metrics = {
            name: get_metric_fun(metric_d, self.net.device)
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
        self.adversarial_scaler = {
            name: torch.cuda.amp.GradScaler()
            for name in self.adversarial_net.keys()
        }
        self.nb_step = {name: 0 for name in self.adversarial_net}
        self.nb_step['autoencoder'] = 0

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
                metric_specs, "cpu")

    def validate_adversarial(self, net, metrics, dataset,
                             batch_size=128):
        """
        Compute metrics on validation_dataset and print some metrics
        """

        data = {name: dataset.get_data_by_name(name)
                for name in dataset.names}

        inputs_needed = [
            x for metric_d in metrics.values()
            for x in metric_d['inputs']]

        net_pred = net.predict_dataset(
            dataset,
            inputs_needed,
            batch_size=batch_size,
            modules=self.modules,
            networks={'autoencoder': self.net, **self.adversarial_net})

        metrics_batches = []
        for name in net_pred.keys():
            dic = dict()
            inputs = {}
            inputs.update(net_pred[name])
            for metric, metric_d in metrics.items():
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
            for metric in metrics
        }

        return metrics_epoch

    def get_batch_adversarial_loss(self, net, inputs, losses):
        """ Get the loss for the adversarial network
        Single forward and backward pass """

        if 'site' in inputs.keys():
            inputs['site'] = inputs['site'].squeeze(-1)

        for input_name in inputs.keys():
            inputs[input_name] = inputs[input_name].to(self.net.device)

        inputs_needed = [inp for loss in losses
                         for inp in loss['inputs']]

        for input_needed in inputs_needed:
            inputs = compute_modules(
                input_needed, inputs,
                {'autoencoder': self.net, **self.adversarial_net},
                self.modules, detach_input=True)

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
        inputs = inputs.copy()

        if 'site' in inputs.keys():
            inputs['site'] = inputs['site'].squeeze(-1)

        for input_name in inputs.keys():
            inputs[input_name] = inputs[input_name].to(self.net.device)

        net_pred = self.net(inputs['sh'], inputs['mean_b0'])
        inputs.update(net_pred)

        inputs_needed = [inp for loss in self.losses + self.style_losses
                         for inp in loss['inputs']]
        for input_needed in inputs_needed:
            inputs = compute_modules(
                input_needed, inputs,
                {'autoencoder': self.net, **self.adversarial_net},
                self.modules
            )

        for net_name, adv_net in self.adversarial_net.items():
            if not any(net_name in s for s in inputs_needed):
                # If we do not need the output of that network
                continue

            net_inputs = [name + '_fake' if name not in ['mask'] else name
                          for name in adv_net.inputs]
            feat_net_pred = adv_net.forward(
                *(inputs[name] for name in net_inputs))

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
        if len(self.losses) != 0:
            batch_loss_reconst = torch.stack(batch_loss_reconst, dim=0).sum()
            loss_dict['reconst_loss'] = batch_loss_reconst
        else:
            batch_loss_reconst = 0

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
            self.aggregate_metrics(
                metrics_final[net_name],
                self.adv_metric_to_maximize[net_name]["inputs"],
                self.adv_metric_to_maximize[net_name]["agg_fun"])
            for net_name in nets_list
        }
        best_net = {net_name: copy.deepcopy(self.adversarial_net[net_name])
                    for net_name in nets_list}

        if train_net_X_time > 0:
            metrics_epoch['autoencoder'] = {metric: -np.inf
                                            for metric in self.metrics}
            # metrics_train_epoch['autoencoder'] = {metric: -np.inf
            #                                       for metric in self.metrics}
            best_value['autoencoder'] = self.aggregate_metrics(
                metrics_final['autoencoder'],
                self.metric_to_maximize["inputs"],
                self.metric_to_maximize["agg_fun"])
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

            for i, data in tqdm(enumerate(dataloader_train, 0), leave=False):
                # with profiler.profile(record_shapes=True) as prof:
                inputs = data

                # Train autoencoder if train_net_X_time > 0
                for _ in range(train_net_X_time):
                    self.optimizer.zero_grad()
                    self.net.train()
                    # with profiler.record_function("autoencoder_backprop"):
                    # with torch.cuda.amp.autocast(enabled=False):
                    _, batch_losses = self.get_batch_loss(data)
                    loss = batch_losses['batch_loss']
                    # self.scaler.scale(loss).backward()
                    loss.backward()
                    # self.scaler.step(self.optimizer)
                    self.optimizer.step()
                    self.scheduler.step()
                    # self.scaler.update()
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

                    # with profiler.record_function(net_name + "_backprop"):
                    # with torch.cuda.amp.autocast(enabled=False):
                    inputs, batch_losses = self.get_batch_adversarial_loss(
                        self.adversarial_net[net_name],
                        inputs,
                        self.adversarial_losses[net_name])

                    loss = batch_losses['batch_loss']
                    # self.adversarial_scaler[net_name].scale(loss).backward()
                    loss.backward()
                    # self.adversarial_scaler[net_name].step(
                    #     self.adversarial_optimizer[net_name])
                    self.adversarial_optimizer[net_name].step()
                    self.adversarial_scheduler[net_name].step()
                    # self.adversarial_scaler[net_name].update()

                    for name, loss in batch_losses.items():
                        loss = loss.item()
                        loss += epoch_loss_train[net_name].setdefault(
                            name, 0)
                        epoch_loss_train[net_name][name] = loss
                    del batch_losses
                del inputs

            # print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total"))
            # prof.export_chrome_trace("trace.json")
            # self.net.plot_grad_flow()

            for net_name in epoch_loss_train.keys():
                self.nb_step[net_name] += 1
                for name, loss in epoch_loss_train[net_name].items():
                    self.writer.add_scalar(net_name + '/loss/' + name,
                                           loss / (i + 1.),
                                           self.nb_step[net_name])
                    epoch_loss_train[net_name][name] = loss / (i + 1.)

            with torch.no_grad():
                if train_net_X_time > 0:
                    # metrics_epoch['autoencoder'] = self.validate(
                    #     validation_dataset, batch_size)
                    metrics_epoch['autoencoder'] = self.validate_adversarial(
                        self.net,
                        self.metrics,
                        validation_dataset,
                        batch_size=batch_size)

                    for name, metric in metrics_epoch['autoencoder'].items():
                        self.writer.add_scalar('autoencoder/metric/' + name + '_val',
                                               metric,
                                               self.nb_step['autoencoder'])

                for net_name in nets_list:
                    metrics_epoch[net_name] = self.validate_adversarial(
                        self.adversarial_net[net_name],
                        self.adversarial_metrics[net_name],
                        validation_dataset,
                        batch_size=batch_size)

                    # metrics_train_epoch[net_name] = self.validate_adversarial(
                    #     self.adversarial_net[net_name],
                    #     self.adversarial_metrics[net_name],
                    #     train_dataset,
                    #     batch_size=batch_size)

                    for name, metric in metrics_epoch[net_name].items():
                        self.writer.add_scalar(net_name + '/metric/' + name + '_val',
                                               metric,
                                               self.nb_step[net_name])
                    # for name, metric in metrics_train_epoch[net_name].items():
                    #     self.writer.add_scalar(net_name + '/metric/' + name + '_train',
                    #                            metric,
                    #                            self.nb_step[net_name])

                feat_dataset = None

            if self.save_folder:
                if train_net_X_time > 0:
                    file_name = self.save_folder + 'autoencoder'
                    file_name = "{}_{}.tar.gz".format(
                        file_name, str(self.nb_step['autoencoder']))
                    self.net.save(file_name)
                for net_name in nets_list:
                    file_name = self.save_folder + net_name
                    file_name = "{}_{}.tar.gz".format(
                        file_name, str(self.nb_step[net_name]))
                    self.adversarial_net[net_name].save(file_name)

            if train_net_X_time > 0:
                """
                If we train the autoencoder, each adversarial network
                is saved at the same time.
                """
                actual_value = self.aggregate_metrics(
                    metrics_epoch["autoencoder"],
                    self.metric_to_maximize["inputs"],
                    self.metric_to_maximize["agg_fun"])
                if actual_value > best_value['autoencoder']:
                    best_value['autoencoder'] = actual_value
                    for net_name in nets_list:
                        best_value[net_name] = self.aggregate_metrics(
                            metrics_epoch[net_name],
                            self.adv_metric_to_maximize[net_name]["inputs"],
                            self.adv_metric_to_maximize[net_name]["agg_fun"])
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
                    actual_value = self.aggregate_metrics(
                        metrics_epoch[net_name],
                        self.adv_metric_to_maximize[net_name]["inputs"],
                        self.adv_metric_to_maximize[net_name]["agg_fun"])
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

    def aggregate_metrics(self, metrics, metrics_name, agg_fun='mean'):
        if type(agg_fun) is str:
            if agg_fun == 'mean':
                agg_fun = np.mean
            elif agg_fun == 'sum':
                agg_fun = np.sum
        result = agg_fun([metrics[name] for name in metrics_name])
        return result
