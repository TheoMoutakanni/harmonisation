import torch
import torch.nn as nn

from .base import BaseNet


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


def add_affine(x, affine):
    # cond = [batch_size, x_channels * 2]
    p = affine.size(1) // 2
    mean, std = affine[:, :p], affine[:, p:]
    mean = mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    std = std.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    out = x * std + mean
    return out


class DisNet(BaseNet):
    def __init__(self, in_dim, out_dim, num_filters, nb_layers, embed_size):
        super(DisNet, self).__init__(in_dim=in_dim,
                                     out_dim=out_dim,
                                     num_filters=num_filters,
                                     nb_layers=nb_layers,
                                     embed_size=embed_size)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        self.nb_layers = nb_layers
        self.embed_size = embed_size
        activation = nn.LeakyReLU(0.2, inplace=True)
        normalization = nn.InstanceNorm3d

        # Site Embedding
        self.embed = nn.Sequential(
            *[nn.Sequential(
                conv_block_2_3d(
                    self.num_filters * 2**(i - 1) if i > 0 else self.in_dim,
                    self.num_filters * 2**i,
                    activation,
                    nn.Identity),
                max_pooling_3d(),
            )
                for i in range(self.nb_layers)
            ],
            nn.Flatten(),
            nn.Linear(
                self.num_filters * 2**(self.nb_layers - 1) * int(
                    (32 / (2**self.nb_layers))**3),
                self.embed_size),
            activation,
            nn.Linear(self.embed_size, self.embed_size)
        )

        # Down sampling
        self.down_conv = nn.ModuleList([
            conv_block_2_3d(
                self.num_filters * 2**(i - 1) if i > 0 else self.in_dim,
                self.num_filters * 2**i,
                activation,
                normalization)
            for i in range(self.nb_layers)])

        self.down_pool = nn.ModuleList([max_pooling_3d()
                                        for i in range(self.nb_layers)])

        # Bridge
        self.bridge = conv_block_2_3d(
            self.num_filters * 2**(self.nb_layers - 1),
            self.num_filters * 2**self.nb_layers,
            activation, normalization)

        # Up sampling
        self.up_trans = nn.ModuleList([
            conv_trans_block_3d(
                self.num_filters * 2**(i + 1),
                self.num_filters * 2**(i + 1),
                activation, nn.Identity)
            for i in range(self.nb_layers)[::-1]])

        self.up_conv = nn.ModuleList([
            conv_block_2_3d(
                self.num_filters * (2**(i + 1) + 2**i),
                self.num_filters * 2**i,
                activation, nn.Identity)
            for i in range(self.nb_layers)[::-1]])

        self.up_norm_1 = nn.ModuleList([
            nn.InstanceNorm3d(self.num_filters * 2**(i + 1), affine=False)
            for i in range(self.nb_layers)[::-1]])
        self.up_affine_1 = nn.ModuleList([
            nn.Linear(self.embed_size, 2 * self.num_filters * 2**(i + 1))
            for i in range(self.nb_layers)[::-1]])

        self.up_norm_2 = nn.ModuleList([
            nn.InstanceNorm3d(self.num_filters * 2**i, affine=False)
            for i in range(self.nb_layers)[::-1]])
        self.up_affine_2 = nn.ModuleList([
            nn.Linear(self.embed_size, 2 * self.num_filters * 2**i)
            for i in range(self.nb_layers)[::-1]])

        # Output
        self.out = nn.Sequential(
            nn.Conv3d(self.num_filters, out_dim,
                      kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        x = x.permute((0, 4, 1, 2, 3))
        # Down sampling

        site_embedding = self.embed(x)

        down_x = []
        temp = x
        for i in range(self.nb_layers):
            temp = self.down_conv[i](temp)
            down_x.append(temp)
            temp = self.down_pool[i](temp)

        # Bridge
        bridge = self.bridge(temp)

        # Up sampling

        temp = bridge
        for i in range(self.nb_layers):
            temp = self.up_trans[i](temp)

            temp = self.up_norm_1[i](temp)
            affine = self.up_affine_1[i](site_embedding)
            temp = add_affine(temp, affine)

            temp = torch.cat([temp, down_x[-1 - i]], dim=1)
            temp = self.up_conv[i](temp)

            temp = self.up_norm_2[i](temp)
            affine = self.up_affine_2[i](site_embedding)
            temp = add_affine(temp, affine)

        # Output
        out = self.out(temp)

        out = out.permute((0, 2, 3, 4, 1))
        return out
