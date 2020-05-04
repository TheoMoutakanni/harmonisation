import torch.nn as nn

from .base import BaseNet
from .enet import InitialBlock, DownsamplingBottleneck, RegularBottleneck


class AdversarialNet(BaseNet):
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
        super().__init__()

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
