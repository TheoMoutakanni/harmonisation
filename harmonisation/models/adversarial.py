import torch.nn as nn
import numpy as np
from collections import OrderedDict

from .base import BaseNet
from .enet import InitialBlock, DownsamplingBottleneck, RegularBottleneck
from .metric_module import FAModule, DWIModule


def conv_block_3d(in_dim, out_dim, activation, normalization):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        normalization(out_dim),
        activation,)


def conv_trans_block_3d(in_dim, out_dim, activation, normalization):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3,
                           stride=2, padding=1, output_padding=1),
        normalization(out_dim),
        activation,)


def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


def conv_block_2_3d(in_dim, out_dim, activation, normalization):
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation, normalization),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        normalization(out_dim),)


class AdversarialNet(BaseNet):
    def __init__(self, in_dim, out_dim, num_filters,
                 nb_layers, embed_size, patch_size, modules,
                 return_dict_layers=False):
        super(AdversarialNet, self).__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            num_filters=num_filters,
            nb_layers=nb_layers,
            embed_size=embed_size,
            patch_size=patch_size,
            # modules=modules,
            return_dict_layers=return_dict_layers)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        self.nb_layers = nb_layers
        self.embed_size = embed_size
        self.patch_size = np.array(patch_size)
        self.return_dict_layers = return_dict_layers

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
                normalization)
            classifier_feat['maxpool_{}'.format(i)] = max_pooling_3d()
            classifier_feat['dropout_{}'.format(i)] = nn.Dropout3d(p=0.05)

        self.classifier_feat = nn.ModuleDict(classifier_feat)

        self.classifier_dense = nn.ModuleDict(OrderedDict({
            'dense_feat_1': nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    self.num_filters * 2**(self.nb_layers - 1) * int(
                        np.prod(self.patch_size / (2**self.nb_layers))),
                    self.embed_size),
                activation,
                # nn.BatchNorm1d(self.embed_size),
                # nn.Dropout3d(p=0.2),
            )}))

        self.classifier_out = nn.Linear(self.embed_size, self.out_dim)

    def forward(self, x, mean_b0, mask):
        out_feat = {}

        dwi = self.modules['dwi'](x, mean_b0)
        fa = self.modules['fa'](dwi, mask)

        # out_feat['dwi'] = dwi
        out_feat['fa'] = fa

        x = fa.permute((0, 4, 1, 2, 3))

        for name, layer in self.classifier_feat.items():
            x = layer(x)
            out_feat[name] = x

        for name, layer in self.classifier_dense.items():
            x = layer(x)
            out_feat[name] = x

        out = self.classifier_out(x)

        dict_layer = out_feat

        dict_layer['y_proba'] = out

        if self.return_dict_layers:
            return dict_layer
        else:
            return out


class OldAdversarialNet(BaseNet):
    """Generate the ENet model.
    Keyword arguments:
    - num_classes (int): the number of classes to segment.
    - encoder_relu (bool, optional): When ``True`` ReLU is used as the
    activation function in the encoder blocks/layers; otherwise, PReLU
    is used. Default: False.
    - decoder_relu (bool, optional): When ``True`` ReLU is used as the
    activation function in the decoder blocks/layers; otherwise, PReLU
    is used. Default: True.
    """

    def __init__(self,
                 patch_size,
                 sh_order,
                 number_of_classes,
                 embed=[16, 32, 64],
                 encoder_relu=False):
        super().__init__(patch_size=patch_size,
                         sh_order=sh_order,
                         number_of_classes=number_of_classes,
                         embed=embed,
                         encoder_relu=encoder_relu)

        self.ncoef = int((sh_order + 2) * (sh_order + 1) / 2)
        self.patch_size = patch_size

        self.initial_block = InitialBlock(self.ncoef,
                                          embed[0], relu=encoder_relu)

        # Stage 1 - Encoder
        self.downsample1_0 = DownsamplingBottleneck(
            embed[0],
            embed[1],
            return_indices=True,
            dropout_prob=0.01,
            relu=encoder_relu)
        self.regular1_1 = RegularBottleneck(
            embed[1], padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_2 = RegularBottleneck(
            embed[1], padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_3 = RegularBottleneck(
            embed[1], padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_4 = RegularBottleneck(
            embed[1], padding=1, dropout_prob=0.01, relu=encoder_relu)

        # Stage 2 - Encoder
        self.downsample2_0 = DownsamplingBottleneck(
            embed[1],
            embed[2],
            return_indices=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.regular2_1 = RegularBottleneck(
            embed[2], padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_2 = RegularBottleneck(
            embed[2], dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_3 = RegularBottleneck(
            embed[2],
            kernel_size=5,
            padding=2,
            asymmetric=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.dilated2_4 = RegularBottleneck(
            embed[2], dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        # self.regular2_5 = RegularBottleneck(
        #     embed[2], padding=1, dropout_prob=0.1, relu=encoder_relu)
        # self.dilated2_6 = RegularBottleneck(
        #     embed[2], dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        # self.asymmetric2_7 = RegularBottleneck(
        #     embed[2],
        #     kernel_size=5,
        #     asymmetric=True,
        #     padding=2,
        #     dropout_prob=0.1,
        #     relu=encoder_relu)
        # self.dilated2_8 = RegularBottleneck(
        #     embed[2], dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)

        self.adaptive_avgpool3_1 = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classif3_2 = nn.Sequential(
            nn.Linear(in_features=embed[2], out_features=embed[2], bias=True),
            nn.BatchNorm1d(embed[2]),
            nn.ReLU(),
            nn.Linear(in_features=embed[2], out_features=number_of_classes))

    def forward(self, x):
        # Initial block
        x = x.permute((0, 4, 1, 2, 3))
        x = self.initial_block(x)

        # Stage 1 - Encoder
        x, max_indices1_0 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)

        # Stage 2 - Encoder
        x, max_indices2_0 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        # x = self.regular2_5(x)
        # x = self.dilated2_6(x)
        # x = self.asymmetric2_7(x)
        # x = self.dilated2_8(x)

        x = self.adaptive_avgpool3_1(x)
        x = x.squeeze()
        x = self.classif3_2(x)

        return x
