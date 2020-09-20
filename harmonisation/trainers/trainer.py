import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import numpy as np
import copy
from tqdm import tqdm

from harmonisation.functions import metrics, shm
from harmonisation.functions.losses import get_loss_fun
from harmonisation.viz import print_peaks, print_diff, print_RIS, print_data
from harmonisation.datasets.utils import batch_to_xyz

import matplotlib.pyplot as plt

from harmonisation.utils import compute_modules


class Iterator():
    def __init__(self):
        self.iterator = 0

    def get_iter(self):
        return self.iterator

    def step(self):
        self.iterator += 1


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
        self.writer_path = self.save_folder + '/data/'
        self.writer = SummaryWriter(logdir=self.writer_path)

    def validate(self, validation_dataset, batch_size=128):
        """
        Compute metrics on validation_dataset
        """
        data_true = {name: validation_dataset.get_data_by_name(name)
                     for name in validation_dataset.names}

        inputs_needed = [
            x for metric_d in self.metrics.values()
            for x in metric_d['inputs']]

        data_fake = self.net.predict_dataset(validation_dataset,
                                             inputs_needed,
                                             batch_size=batch_size,
                                             modules=self.modules)
        metrics_fun = metrics.get_metric_dict()

        metrics_batches = []
        for name in list(set(data_fake.keys()) & set(data_true.keys())):
            dic = dict()
            for metric in self.metrics:
                metric_calc = []
                nb_batches = int(
                    np.ceil(len(data_fake[name]['sh_fake']) / batch_size))
                for batch in range(nb_batches):
                    idx = range(batch * batch_size,
                                min((batch + 1) * batch_size,
                                    len(data_fake[name]['sh_fake'])))
                    metric_calc.append(metrics_fun[metric](
                        torch.FloatTensor(data_true[name]['sh'][idx]),
                        torch.FloatTensor(data_fake[name]['sh_fake'][idx]),
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

    def compute_modules(self, input_needed, inputs, nets_dict, modules):
        # if 'dwi' in inputs_needed and 'dwi' not in inputs.keys():
        #     inputs['dwi'] = self.modules['dwi'](inputs['sh'],
        #                                         inputs['mean_b0'])
        # if 'dwi_fake' in inputs_needed and 'dwi_fake' not in inputs.keys():
        #     inputs['dwi_fake'] = self.modules['dwi'](inputs['sh_fake'],
        #                                              inputs['mean_b0_fake'])
        # if 'fa' in inputs_needed and 'fa' not in inputs.keys():
        #     inputs['fa'] = self.modules['fa'](inputs['dwi'], inputs['mask'])
        # if 'fa_fake' in inputs_needed and 'fa_fake' not in inputs.keys():
        #     inputs['fa_fake'] = self.modules['fa'](inputs['dwi_fake'],
        #                                            inputs['mask'])

        # return inputs

        if input_needed not in inputs.keys():
            if any([input_needed in x for x in modules.keys()]):
                module_name = input_needed.split('_')[0]
                net_inputs = modules[module_name].inputs

                if 'fake' in input_needed:
                    net_inputs = [
                        name + '_fake' if name not in ['mask'] else name
                        for name in net_inputs]
                    base_name = '{}_fake'
                else:
                    base_name = '{}'

                for name in input_needed:
                    inputs = compute_modules(
                        name, inputs, nets_dict, modules)

                output_name = base_name.format(input_needed)
                inputs[output_name] = modules[module_name](
                    **{name: inputs[name] for name in net_inputs})
                return inputs

            elif input_needed in nets_dict['autoencoder'].outputs:
                net = nets_dict['autoencoder']
                for name in net.inputs:
                    inputs = compute_modules(
                        name, inputs, nets_dict, modules)
                return inputs.update(net(**{name: inputs[name]
                                            for name in net.inputs}))
            else:
                for net_name in nets_dict.keys():
                    if net_name in input_needed:
                        net = nets_dict[net_name]
                        break
                net_inputs = net.inputs

                if 'fake' in input_needed:
                    net_inputs = [
                        name + '_fake' if name not in ['mask'] else name
                        for name in net_inputs]
                    base_name = '{}_fake_{}'
                else:
                    base_name = '{}_{}'

                for name in net_inputs:
                    inputs = compute_modules(
                        name, inputs, nets_dict, modules)
                net_pred = net(**{name: inputs[name] for name in net.inputs})
                return inputs.update(
                    {base_name.format(name, net_name): net_pred[name]
                     for name in net_pred.keys()})
        else:
            return inputs

    def get_batch_loss(self, inputs):
        """ Single forward and backward pass """

        for input_name in inputs.keys():
            inputs[input_name] = inputs[input_name].to(self.net.device)

        net_pred = self.net(inputs['sh'], inputs['mean_b0'])
        inputs.update(net_pred)

        inputs_needed = [inp for loss in self.losses for inp in loss['inputs']]
        inputs = compute_modules(
            inputs_needed, inputs, {'autoencoder': self.net}, self.modules)

        batch_loss = []
        for loss_d in self.losses:
            loss = loss_d['fun'](*[inputs[name] for name in loss_d['inputs']])
            loss = loss_d['coeff'] * loss
            batch_loss.append(loss)

        batch_loss = torch.stack(batch_loss, dim=0).sum()

        return inputs, {'batch_loss': batch_loss}

    def train(self,
              train_dataset,
              validation_dataset,
              num_epochs,
              batch_size=128,
              validation=True,
              metrics_final=None,
              freq_print=None):

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
                    loss=epoch_loss_train['batch_loss'],
                )

            # Training

            epoch_loss_train = {}

            for i, data in tqdm(enumerate(dataloader_train, 0),
                                total=len(train_dataset) // batch_size +
                                int(len(train_dataset) % batch_size != 0),
                                leave=False):
                self.optimizer.zero_grad()

                # Set network to train mode
                self.net.train()

                _, batch_losses = self.get_batch_loss(data)

                loss = batch_losses['batch_loss']
                loss.backward()

                for name, loss in batch_losses.items():
                    loss = epoch_loss_train.setdefault(name, 0) + loss.item()
                    epoch_loss_train[name] = loss

                # self.net.plot_grad_flow()

                # gradient descent
                self.optimizer.step()

            for name, loss in epoch_loss_train.items():
                self.writer.add_scalar('loss/' + name,
                                       loss / (i + 1), epoch)
                epoch_loss_train[name] = loss / (i + 1)

            # Validation and print
            with torch.no_grad():

                metrics_epoch = self.validate(validation_dataset, batch_size)

                # if freq_print is not None and (
                #         epoch % freq_print == 0 and epoch != 0):
                #     self.print_metrics(validation_dataset, batch_size)

            for name, metric in metrics_epoch.items():
                self.writer.add_scalar('metric/' + name + '_val',
                                       metric, epoch)

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

        return best_net, metrics_final
