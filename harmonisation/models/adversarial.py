import torch.nn as nn
import numpy as np
from collections import OrderedDict

from .base import BaseNet


def conv_block_3d(in_dim, out_dim, activation, normalization,
                  spectral_norm=False):
    fun = nn.utils.spectral_norm if spectral_norm else lambda x: x

    return nn.Sequential(
        fun(nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)),
        normalization(out_dim),
        activation,)


def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


def conv_block_2_3d(in_dim, out_dim, activation, normalization,
                    spectral_norm=False):
    fun = nn.utils.spectral_norm if spectral_norm else lambda x: x

    return nn.Sequential(
        conv_block_3d(in_dim, out_dim,
                      activation, normalization, spectral_norm),
        fun(nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)),
        normalization(out_dim),)


class AdversarialNet(BaseNet):
    def __init__(self, in_dim, out_dim, num_filters,
                 nb_layers, embed_size, patch_size, modules,
                 spectral_norm=False):
        super(AdversarialNet, self).__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            num_filters=num_filters,
            nb_layers=nb_layers,
            embed_size=embed_size,
            patch_size=patch_size,
            # modules=modules),
            spectral_norm=spectral_norm,
        )

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        self.nb_layers = nb_layers
        self.embed_size = embed_size
        self.patch_size = np.array(patch_size)
        self.spectral_norm = spectral_norm

        self.inputs = ['sh', 'mean_b0', 'mask']

        self.modules = modules

        activation = nn.LeakyReLU(0.2, inplace=True)
        normalization = nn.BatchNorm3d

        classifier_feat = OrderedDict({})
        for i in range(self.nb_layers):
            classifier_feat['conv_feat_{}'.format(i)] = conv_block_2_3d(
                self.num_filters * 2**(i - 1) if i > 0 else self.in_dim,
                self.num_filters * 2**i,
                activation,
                normalization,
                self.spectral_norm)
            classifier_feat['maxpool_{}'.format(i)] = max_pooling_3d()
            classifier_feat['dropout_{}'.format(i)] = nn.Dropout3d(p=0.1)

        self.classifier_feat = nn.ModuleDict(classifier_feat)

        fun = nn.utils.spectral_norm if self.spectral_norm else lambda x: x

        self.classifier_dense = nn.ModuleDict(OrderedDict({
            'dense_feat_1': nn.Sequential(
                nn.Flatten(),
                fun(nn.Linear(
                    self.num_filters * 2**(self.nb_layers - 1) * int(
                        np.prod(self.patch_size / (2**self.nb_layers))),
                    self.embed_size)),
                activation,
                # nn.BatchNorm1d(self.embed_size),
                # nn.Dropout3d(p=0.2),
            )
        }))

        self.classifier_out = fun(nn.Linear(self.embed_size, self.out_dim))

    def forward(self, x, mean_b0, mask):
        out_dict = {}

        dwi = self.modules['dwi'](x, mean_b0)
        fa = self.modules['fa'](dwi, mask)

        # out_feat['dwi'] = dwi
        # out_feat['fa'] = fa

        x = fa.permute((0, 4, 1, 2, 3))

        for name, layer in self.classifier_feat.items():
            x = layer(x)
            out_dict[name] = x

        for name, layer in self.classifier_dense.items():
            x = layer(x)
            out_dict[name] = x

        out = self.classifier_out(x)

        out_dict['y_logits'] = out

        return out_dict
