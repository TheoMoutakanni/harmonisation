import torch
import torch.nn as nn
import numpy as np

from dipy.reconst.shm import order_from_ncoef

from collections import OrderedDict

from .base import BaseNet


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, X):
        return X


class AutoEncoderBase(BaseNet):

    def __init__(self,
                 patch_size,
                 mean_data,
                 std_data,
                 sh_order,
                 params_layers=[2000, 1000, 1000, 500, 500],
                 pdrop=0.1):
        super(AutoEncoderBase, self).__init__()

        self.mean_data = mean_data.to('cuda')
        self.std_data = std_data.to('cuda')

        self.pdrop = pdrop
        self.params_layers = params_layers

        self.ncoef = int((sh_order + 2) * (sh_order + 1) / 2)
        self.patch_size = patch_size

        self.input_size = int(self.ncoef * np.prod(self.patch_size))
        self.params_layers = [self.input_size] + self.params_layers

        self.encode_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    OrderedDict([
                        ("linear_encode_{}".format(k),
                         nn.Linear(in_features=self.params_layers[k],
                                   out_features=self.params_layers[k + 1],
                                   bias=True)),
                        ("batchnorm_{}".format(k), nn.BatchNorm1d(
                            self.params_layers[k + 1])),
                        ("relu_{}".format(k), nn.ReLU()),
                        ("dropout_{}".format(k), nn.Dropout(self.pdrop)),
                    ])
                ) for k in range(len(self.params_layers) - 1)
            ]
        )

        self.decode_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    OrderedDict([
                        ("linear_decode_{}".format(k - 1),
                         nn.Linear(in_features=self.params_layers[-k],
                                   out_features=self.params_layers[-k - 1],
                                   bias=True)),
                        ("batchnorm_{}".format(k), nn.BatchNorm1d(
                            self.params_layers[-k - 1])),
                        ("relu_{}".format(k), nn.ReLU() if k < len(
                            self.params_layers) - 1 else Identity()),
                        ("dropout_{}".format(k), nn.Dropout(self.pdrop)),
                    ])
                ) for k in range(1, len(self.params_layers))
            ]
        )

    def embbed(self, X):
        batch = X.shape[0]

        X = X.reshape(batch, -1)
        for block in self.encode_blocks:
            X = block(X)
        return X

    def decode(self, Y):
        for block in self.decode_blocks:
            Y = block(Y)
        batch = Y.shape[0]
        Y = Y.reshape(batch, *self.patch_size, self.ncoef + 1)
        return Y

    def forward(self, X_sh, X_b0):
        Y = self.embbed(X_sh, X_b0)
        Z_sh, Z_b0 = self.decode(Y)
        return Z_sh, Z_b0
