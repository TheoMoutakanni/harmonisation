import tarfile
import tempfile
import json
import shutil
import tqdm

import torch
import torch.nn as nn

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


matplotlib.use('TkAgg')


class BaseNet(nn.Module, object):

    def __init__(self, **kwargs):
        super(BaseNet, self).__init__()

        # Keep all the __init__ parameters for saving/loading
        self.net_parameters = kwargs

    @property
    def device(self):
        try:
            out = next(self.parameters()).device
            return (out if isinstance(out, torch.device)
                    else torch.device('cpu'))
        except Exception:
            return torch.device('cpu')

    def save(self, filename):
        with tarfile.open(filename, "w") as tar:
            temporary_directory = tempfile.mkdtemp()
            name = "{}/net_params.json".format(temporary_directory)
            json.dump(self.net_parameters, open(name, "w"))
            tar.add(name, arcname="net_params.json")
            name = "{}/state.torch".format(temporary_directory)
            torch.save(self.state_dict(), name)
            tar.add(name, arcname="state.torch")
            shutil.rmtree(temporary_directory)
        return filename

    @classmethod
    def load(cls, filename, use_device=torch.device('cpu'), *args, **kwargs):
        with tarfile.open(filename, "r") as tar:
            net_parameters = json.loads(
                tar.extractfile("net_params.json").read().decode("utf-8"))
            kwargs.update(net_parameters)
            path = tempfile.mkdtemp()
            tar.extract("state.torch", path=path)
            net = cls(*args, **kwargs)
            net.load_state_dict(
                torch.load(
                    path + "/state.torch",
                    map_location=use_device,
                )
            )
        return net, net_parameters

    def forward(self, X):
        raise NotImplementedError("Please implement this method")

    def move_to(self, obj, device, numpy=False):
        if torch.is_tensor(obj):
            obj = obj.to(device)
            if numpy:
                return obj.numpy()
            else:
                return obj
        elif isinstance(obj, dict):
            res = {}
            for k, v in obj.items():
                res[k] = self.move_to(v, device, numpy=numpy)
            return res
        elif isinstance(obj, list):
            res = []
            for v in obj:
                res.append(self.move_to(v, device, numpy=numpy))
            return res
        else:
            raise TypeError("Invalid type for move_to")

    def concatenate(self, obj):
        if isinstance(obj[0], dict):
            return {k: self.concatenate([d[k] for d in obj])
                    for k in obj[0].keys()}
        elif isinstance(obj[0], np.ndarray):
            return np.concatenate(obj)
        else:
            return torch.cat(obj)

    def predict_dataset(self, dataset, batch_size=128, names=None, numpy=True):
        """Predicts signals in dictionnary inference_dataset = {name: data}.
        """
        with torch.no_grad():
            self.eval()

            results = dict()

            if names is None:
                names = dataset.names

            for dmri_name in names:
                data = dataset.get_data_by_name(dmri_name)
                signals = [data[signal_name]
                           for signal_name in self.inputs]

                results[dmri_name] = []
                nb_input = signals[0].shape[0]
                number_of_batches = int(np.ceil(nb_input / batch_size))

                for batch in tqdm.tqdm(range(number_of_batches), leave=False):
                    signal_batch = [torch.FloatTensor(
                        signal[batch * batch_size:(batch + 1) * batch_size])
                        for signal in signals]
                    signal_batch = [signal.to(self.device)
                                    for signal in signal_batch]
                    signal_pred = self.forward(*signal_batch)
                    results[dmri_name].append(self.move_to(signal_pred, 'cpu',
                                                           numpy=numpy))
                results[dmri_name] = self.concatenate(results[dmri_name])

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
