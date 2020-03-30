import tarfile
import tempfile
import json
import shutil

import torch
import torch.nn as nn

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


matplotlib.use('TkAgg')


class BaseNet(nn.Module, object):

    def __init__(self):
        super(BaseNet, self).__init__()

    @property
    def device(self):
        try:
            out = next(self.parameters()).device
            return (out if isinstance(out, torch.device)
                    else torch.device('cpu'))
        except Exception:
            return torch.device('cpu')

    def save(self, filename, net_parameters):
        with tarfile.open(filename, "w") as tar:
            temporary_directory = tempfile.mkdtemp()
            name = "{}/net_params.json".format(temporary_directory)
            json.dump(net_parameters, open(name, "w"))
            tar.add(name, arcname="net_params.json")
            name = "{}/state.torch".format(temporary_directory)
            torch.save(self.state_dict(), name)
            tar.add(name, arcname="state.torch")
            shutil.rmtree(temporary_directory)
        return filename

    @classmethod
    def load(cls, filename, use_device=torch.device('cpu')):
        with tarfile.open(filename, "r") as tar:
            net_parameters = json.loads(
                tar.extractfile("net_params.json").read().decode("utf-8"))
            path = tempfile.mkdtemp()
            tar.extract("state.torch", path=path)
            net = cls(**net_parameters)
            net.load_state_dict(
                torch.load(
                    path + "/state.torch",
                    map_location=use_device,
                )
            )
        return net, net_parameters

    def forward(self, X):
        raise NotImplementedError("Please implement this method")

    def predict_dataset(self, dataset, batch_size=128):
        """Predicts signals in dictionnary inference_dataset = {name: data}.
        """
        with torch.no_grad():
            self.eval()

            results = dict()

            names = dataset.names

            for dmri_name in names:
                data = dataset.get_data_by_name(dmri_name)
                signal = data['sh']

                results[dmri_name] = []
                number_of_batches = int(np.ceil(signal.shape[0] / batch_size))

                for batch in range(number_of_batches):
                    signal_batch = signal[batch *
                                          batch_size:(batch + 1) * batch_size]
                    signal_batch = signal_batch.to(self.device)
                    signal_pred = self.forward(signal_batch)
                    results[dmri_name].append(signal_pred.to('cpu'))

                results[dmri_name] = torch.cat(results[dmri_name])

        return results

    @property
    def nelement(self):
        cpt = 0
        for p in self.parameters():
            cpt += p.nelement()
        return cpt

    def plot_grad_flow(self):
        """Plots the gradients flowing through different layers in the net
        during training.
        Can be used for checking for possible gradient vanishing / exploding
        problems.
        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow()" to visualize the gradient flow"""
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in self.named_parameters():
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())

        print(ave_grads)
        plt.bar(np.arange(len(max_grads)), max_grads,
                alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads,
                alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        # zoom in on the lower gradient regions
        plt.ylim(bottom=-0.001, top=0.02)
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)],
                   ['max-gradient', 'mean-gradient', 'zero-gradient'])
        plt.savefig("test.png")
        plt.show()