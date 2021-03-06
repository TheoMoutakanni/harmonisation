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

import torch.autograd.profiler as profiler


class Iterator():
    def __init__(self):
        self.iterator = 0

    def get_iter(self):
        return self.iterator

    def step(self):
        self.iterator += 1


class BaseTrainer():
    def __init__(self,
                 dict_net,
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
                 device="cuda",
                 ):
        self.device = device
        self.dict_net = dict_net
        self.metric_to_maximize = metric_to_maximize
        self.metrics = {
            name: get_metric_fun(metric_d, "cpu")
            for name, metric_d in metrics_specs.items()}
        self.losses = {
            name: get_loss_fun(loss_d, self.device)
            for name, loss_d in loss_specs.items()}
        self.optimizer = {
            name: optim.SGD(self.dict_net[name].parameters(), **params)
            for name, params in optimizer_parameters.items()}
        self.scheduler = {
            name: optim.lr_scheduler.CyclicLR(
                self.optimizer[name], **params)
            for name, params in scheduler_parameters.items()
        }

        self.nb_step = {name: 0 for name in self.dict_net}

        self.modules = modules

        self.patience = patience
        self.save_folder = save_folder
        self.writer_path = self.save_folder + '/data/'
        self.writer = SummaryWriter(logdir=self.writer_path)

    def set_loss(self, loss_specs_by_net, add=False):
        """If add=True, does not replace the old ones"""
        for net_name, loss_specs in loss_specs_by_net.items():
            new_losses = get_loss_fun(loss_specs, self.device)
            if add:
                self.losses[net_name].extend(new_losses)
            else:
                self.losses[net_name] = new_losses

    def set_metric(self, metric_specs_by_net, add=False):
        """If add=True, does not replace the old ones"""
        for net_name, metric_specs in metric_specs_by_net.items():
            new_metrics = get_metric_fun(metric_specs, "cpu")
            if add:
                self.metrics[net_name].update(new_metrics)
            else:
                self.metrics[net_name] = new_metrics

    def validate(self, net, metrics, dataset, batch_size=128):
        """
        Compute metrics on validation_dataset and print some metrics
        """

        inputs_needed = [x for metric_d in metrics.values()
                         for x in metric_d['inputs']]

        net_pred = net.predict_dataset(
            dataset,
            inputs_needed,
            batch_size=batch_size,
            modules=self.modules,
            networks=self.dict_net)

        metrics_batches = []
        for name in net_pred.keys():
            dic = dict()
            inputs = net_pred[name]
            for metric, metric_d in metrics.items():
                dic[metric] = np.nanmean(
                    metric_d["fun"](
                        *[torch.FloatTensor(
                            inputs[input_params["net"]][input_params["name"]])
                          for input_params in metric_d['inputs']]
                    )
                )

            metrics_batches.append(dic)

        metrics_epoch = {
            metric: np.nanmean(
                [m[metric] for m in metrics_batches])
            for metric in metrics
        }

        return metrics_epoch

    def get_batch_loss(self, net, inputs, losses):
        """ Get the loss for the adversarial network
        Single forward and backward pass """

        if 'site' in inputs["dataset"].keys():
            inputs["dataset"]['site'] = inputs["dataset"]['site'].squeeze(-1)

        # for input_name in inputs.keys():
        #     inputs[input_name] = inputs[input_name].to(self.device)

        inputs_needed = [(inp, loss['detach_input']) for loss in losses
                         for inp in loss['inputs']]

        for input_needed, detach_input in inputs_needed:
            inputs = compute_modules(
                input_needed, inputs,
                self.dict_net,
                self.modules,
                self.device,
                detach_input=detach_input)

        loss_dict = {}

        batch_loss = []
        for loss_d in losses:
            loss = loss_d['fun'](*[inputs[params["net"]][params["name"]]
                                   for params in loss_d['inputs']])
            loss = loss_d['coeff'] * loss
            batch_loss.append(loss)

            loss_dict[loss_d['name']] = loss
        batch_loss = torch.stack(batch_loss, dim=0).sum()
        loss_dict['batch_loss'] = batch_loss

        return inputs, loss_dict

    def train(self,
              nets_list,
              train_dataset,
              validation_dataset,
              num_epochs,
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
                           for metric in self.metrics[net_name]}
                for net_name in nets_list}

        metrics_epoch = {net_name:
                         {metric: -np.inf
                          for metric in self.metrics[net_name]}
                         for net_name in nets_list}

        # metrics_train_epoch = {
        #     net_name: {metric: -np.inf
        #                for metric in self.metrics[net_name]}
        #     for net_name in nets_list}

        best_value = {
            net_name:
            self.aggregate_metrics(
                metrics_final[net_name],
                self.metric_to_maximize[net_name]["inputs"],
                self.metric_to_maximize[net_name]["agg_fun"])
            for net_name in nets_list
        }
        best_net = {net_name: copy.deepcopy(self.dict_net[net_name])
                    for net_name in nets_list}

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

            for i, data in tqdm(enumerate(dataloader_train, 0), leave=False):
                inputs = {net_name: {} for net_name in self.dict_net}
                inputs["dataset"] = data

                # with torch.autograd.set_detect_anomaly(True):

                # Train networks
                for net_name in nets_list:
                    self.optimizer[net_name].zero_grad()

                    self.dict_net[net_name].train()

                    # with profiler.record_function(net_name + "_backprop"):
                    inputs, batch_losses = self.get_batch_loss(
                        self.dict_net[net_name],
                        inputs,
                        self.losses[net_name])

                    loss = batch_losses['batch_loss']
                    loss.backward(retain_graph=True)
                    self.optimizer[net_name].step()
                    self.scheduler[net_name].step()

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
                for net_name in nets_list:
                    metrics_epoch[net_name] = self.validate(
                        self.dict_net[net_name],
                        self.metrics[net_name],
                        validation_dataset,
                        batch_size=batch_size)

                    # metrics_train_epoch[net_name] = self.validate(
                    #     self.dict_net[net_name],
                    #     self.metrics[net_name],
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

            if self.save_folder:
                for net_name in nets_list:
                    file_name = self.save_folder + net_name
                    file_name = "{}_{}.tar.gz".format(
                        file_name, str(self.nb_step[net_name]))
                    self.dict_net[net_name].save(file_name)

            for net_name in nets_list:
                actual_value = self.aggregate_metrics(
                    metrics_epoch[net_name],
                    self.metric_to_maximize[net_name]["inputs"],
                    self.metric_to_maximize[net_name]["agg_fun"])
                if actual_value > best_value[net_name]:
                    best_value[net_name] = actual_value
                    last_update = epoch
                    best_net[net_name] = copy.deepcopy(
                        self.dict_net[net_name])
                    metrics_final[net_name] = metrics_epoch[net_name]

            if last_update == epoch:
                counter_patience = 0
            else:
                counter_patience += 1

            if counter_patience > self.patience:
                break

        if keep_best_net:
            for net_name in nets_list:
                self.dict_net[net_name] = best_net[net_name]

        return best_net, metrics_final

    def aggregate_metrics(self, metrics, metrics_name, agg_fun='mean'):
        if type(agg_fun) is str:
            if agg_fun == 'mean':
                agg_fun = np.mean
            elif agg_fun == 'sum':
                agg_fun = np.sum
        result = agg_fun([metrics[name] for name in metrics_name])
        return result
