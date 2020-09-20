import torch.nn as nn
import torch

from .base import BaseNet


def pixel_shuffle(input, upscale_factor):
    # From https://github.com/pytorch/pytorch/pull/6340/files
    """Rearranges elements in a Tensor of shape :
    math:`(N, C, d_{1}, d_{2}, ..., d_{n})`
    to a tensor of shape :
    math:`(N, C/(r^n), d_{1}*r, d_{2}*r, ..., d_{n}*r)`.
    Where :math:`n` is the dimensionality of the data.
    See :class:`~torch.nn.PixelShuffle` for details.
    Args:
        input (Variable): Input
        upscale_factor (int): factor to increase spatial resolution by
    Examples::
        # 1D example
        >>> input = torch.Tensor(1, 4, 8)
        >>> output = F.pixel_shuffle(input, 2)
        >>> print(output.size())
        torch.Size([1, 2, 16])
        # 2D example
        >>> input = torch.Tensor(1, 9, 8, 8)
        >>> output = F.pixel_shuffle(input, 3)
        >>> print(output.size())
        torch.Size([1, 1, 24, 24])
        # 3D example
        >>> input = torch.Tensor(1, 8, 16, 16, 16)
        >>> output = F.pixel_shuffle(input, 2)
        >>> print(output.size())
        torch.Size([1, 1, 32, 32, 32])
    """
    input_size = list(input.size())
    dimensionality = len(input_size) - 2

    input_size[1] //= (upscale_factor ** dimensionality)
    output_size = [dim * upscale_factor for dim in input_size[2:]]

    input_view = input.contiguous().view(
        input_size[0], input_size[1],
        *(([upscale_factor] * dimensionality) + input_size[2:])
    )

    indicies = list(range(2, 2 + 2 * dimensionality))
    indicies = indicies[1::2] + indicies[0::2]

    shuffle_out = input_view.permute(0, 1, *(indicies[::-1])).contiguous()
    return shuffle_out.view(input_size[0], input_size[1], *output_size)


class PixelShuffle(nn.Module):
    """Real-Time Single Image and Video Super-Resolution Using an Efficient
       Sub-Pixel Convolutional Neural Network:
       https://arxiv.org/abs/1609.05158
    """

    def __init__(self, upscale_factor: int):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return pixel_shuffle(input, self.upscale_factor)


class InitialBlock(nn.Module):
    """The initial block is composed of two branches:
    1. a main branch which performs a regular convolution with stride 2;
    2. an extension branch which performs max-pooling.
    Doing both operations in parallel and concatenating their results
    allows for efficient downsampling and expansion. The main branch
    outputs 13 feature maps while the extension branch outputs 3, for a
    total of 16 feature maps after concatenation.
    Keyword arguments:
    - in_channels (int): the number of input channels.
    - out_channels (int): the number output channels.
    - kernel_size (int, optional): the kernel size of the filters used in
    the convolution layer. Default: 3.
    - padding (int, optional): zero-padding added to both sides of the
    input. Default: 0.
    - bias (bool, optional): Adds a learnable bias to the output if
    ``True``. Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False,
                 relu=True):
        super(InitialBlock, self).__init__()

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # Main branch - As stated above the number of output channels for this
        # branch is the total minus in_channels, since the remaining channels
        # come from the extension branch
        self.main_branch = nn.Conv3d(
            in_channels,
            out_channels - in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=bias)

        # Extension branch
        self.ext_branch = nn.MaxPool3d(3, stride=2, padding=1)

        # Initialize batch normalization to be used after concatenation
        self.batch_norm = nn.BatchNorm3d(
            affine=True, num_features=out_channels)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()

    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)

        # Concatenate branches
        out = torch.cat((main, ext), 1)

        # Apply batch normalization
        out = self.batch_norm(out)

        return self.out_activation(out)


class RegularBottleneck(nn.Module):
    """Regular bottlenecks are the main building block of ENet.
    Main branch:
    1. Shortcut connection.
    Extension branch:
    1. 1x1 convolution which decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. regular, dilated or asymmetric convolution;
    3. 1x1 convolution which increases the number of channels back to
    ``channels``, also called an expansion;
    4. dropout as a regularizer.
    Keyword arguments:
    - channels (int): the number of input and output channels.
    - internal_ratio (int, optional): a scale factor applied to
    ``channels`` used to compute the number of
    channels after the projection. eg. given ``channels`` equal to 128 and
    internal_ratio equal to 2 the number of channels after the projection
    is embed[1]. Default: 4.
    - kernel_size (int, optional): the kernel size of the filters used in
    the convolution layer described above in item 2 of the extension
    branch. Default: 3.
    - padding (int, optional): zero-padding added to both sides of the
    input. Default: 0.
    - dilation (int, optional): spacing between kernel elements for the
    convolution described in item 2 of the extension branch. Default: 1.
    asymmetric (bool, optional): flags if the convolution described in
    item 2 of the extension branch is asymmetric or not. Default: False.
    - dropout_prob (float, optional): probability of an element to be
    zeroed. Default: 0 (no dropout).
    - bias (bool, optional): Adds a learnable bias to the output if
    ``True``. Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.
    """

    def __init__(self,
                 channels,
                 internal_ratio=4,
                 kernel_size=3,
                 padding=0,
                 dilation=1,
                 asymmetric=False,
                 dropout_prob=0,
                 bias=False,
                 relu=True,
                 out_activation=True):
        super().__init__()

        # Check in the internal_scale parameter is within the expected range
        # [1, channels]
        if internal_ratio <= 1 or internal_ratio > channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}."
                               .format(channels, internal_ratio))

        internal_channels = channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # Main branch - shortcut connection

        # Extension branch - 1x1 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution, and,
        # finally, a regularizer (spatial dropout). Number of channels is
        # constant.

        # 1x1 projection convolution
        self.ext_conv1 = nn.Sequential(
            nn.Conv3d(
                channels,
                internal_channels,
                kernel_size=1,
                stride=1,
                bias=bias),
            nn.BatchNorm3d(internal_channels), activation())

        # If the convolution is asymmetric we split the main convolution in
        # two. Eg. for a 5x5 asymmetric convolution we have two convolution:
        # the first is 5x1 and the second is 1x5.
        if asymmetric:
            self.ext_conv2 = nn.Sequential(
                nn.Conv3d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(kernel_size, 1, 1),
                    stride=1,
                    padding=(padding, 0, 0),
                    dilation=dilation,
                    bias=bias),
                nn.BatchNorm3d(internal_channels),
                activation(),
                nn.Conv3d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(1, kernel_size, 1),
                    stride=1,
                    padding=(0, padding, 0),
                    dilation=dilation,
                    bias=bias),
                nn.BatchNorm3d(internal_channels),
                activation(),
                nn.Conv3d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(1, 1, kernel_size),
                    stride=1,
                    padding=(0, 0, padding),
                    dilation=dilation,
                    bias=bias),
                nn.BatchNorm3d(internal_channels),
                activation())
        else:
            self.ext_conv2 = nn.Sequential(
                nn.Conv3d(
                    internal_channels,
                    internal_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=bias),
                nn.BatchNorm3d(internal_channels),
                activation())

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv3d(
                internal_channels,
                channels,
                kernel_size=1,
                stride=1,
                bias=bias),
            nn.BatchNorm3d(channels), activation())

        self.ext_regul = nn.Dropout3d(p=dropout_prob)

        # PReLU layer to apply after adding the branches
        if out_activation:
            self.out_activation = activation()
        else:
            self.out_activation = None

    def forward(self, x):
        # Main branch shortcut
        main = x

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext

        if self.out_activation is not None:
            out = self.out_activation(out)
        return out


class DownsamplingBottleneck(nn.Module):
    """Downsampling bottlenecks further downsample the feature map size.
    Main branch:
    1. max pooling with stride 2; indices are saved to be used for
    unpooling later.
    Extension branch:
    1. 2x2 convolution with stride 2 that decreases the number of channels
    by ``internal_ratio``, also called a projection;
    2. regular convolution (by default, 3x3);
    3. 1x1 convolution which increases the number of channels to
    ``out_channels``, also called an expansion;
    4. dropout as a regularizer.
    Keyword arguments:
    - in_channels (int): the number of input channels.
    - out_channels (int): the number of output channels.
    - internal_ratio (int, optional): a scale factor applied to ``channels``
    used to compute the number of channels after the projection. eg. given
    ``channels`` equal to 128 and internal_ratio equal to 2 the number of
    channels after the projection is embed[1]. Default: 4.
    - return_indices (bool, optional):  if ``True``, will return the max
    indices along with the outputs. Useful when unpooling later.
    - dropout_prob (float, optional): probability of an element to be
    zeroed. Default: 0 (no dropout).
    - bias (bool, optional): Adds a learnable bias to the output if
    ``True``. Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio=4,
                 return_indices=False,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        # Store parameters that are needed later
        self.return_indices = return_indices

        # Check in the internal_scale parameter is within the expected range
        # [1, channels]
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}. "
                               .format(in_channels, internal_ratio))

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_max1 = nn.MaxPool3d(
            2,
            stride=2,
            return_indices=return_indices)

        # Extension branch - 2x2 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution. Number
        # of channels is doubled.

        # 2x2 projection convolution with stride 2
        self.ext_conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels,
                internal_channels,
                kernel_size=2,
                stride=2,
                bias=bias), nn.BatchNorm3d(internal_channels), activation())

        # Convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv3d(
                internal_channels,
                internal_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias), nn.BatchNorm3d(internal_channels), activation())

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv3d(
                internal_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm3d(out_channels), activation())

        self.ext_regul = nn.Dropout3d(p=dropout_prob)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()

    def forward(self, x):
        # Main branch shortcut
        if self.return_indices:
            main, max_indices = self.main_max1(x)
        else:
            main = self.main_max1(x)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Main branch channel padding
        n, ch_ext, h, w, z = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n, ch_ext - ch_main, h, w, z)

        # Before concatenating, check if main is on the CPU or GPU and
        # convert padding accordingly
        if main.is_cuda:
            padding = padding.cuda()

        # Concatenate
        main = torch.cat((main, padding), 1)

        # Add main and extension branches
        out = main + ext

        return self.out_activation(out), max_indices


class UpsamplingBottleneck(nn.Module):
    """The upsampling bottlenecks upsample the feature map resolution using max
    pooling indices stored from the corresponding downsampling bottleneck.
    Main branch:
    1. 1x1 convolution with stride 1 that decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. max unpool layer using the max pool indices from the corresponding
    downsampling max pool layer.
    Extension branch:
    1. 1x1 convolution with stride 1 that decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. transposed convolution (by default, 3x3);
    3. 1x1 convolution which increases the number of channels to
    ``out_channels``, also called an expansion;
    4. dropout as a regularizer.
    Keyword arguments:
    - in_channels (int): the number of input channels.
    - out_channels (int): the number of output channels.
    - internal_ratio (int, optional): a scale factor applied to ``in_channels``
     used to compute the number of channels after the projection. eg. given
     ``in_channels`` equal to 128 and ``internal_ratio`` equal to 2 the number
     of channels after the projection is embed[1]. Default: 4.
    - dropout_prob (float, optional): probability of an element to be zeroed.
    Default: 0 (no dropout).
    - bias (bool, optional): Adds a learnable bias to the output if ``True``.
    Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio=4,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        # Check in the internal_scale parameter is within the expected range
        # [1, channels]
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}. "
                               .format(in_channels, internal_ratio))

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm3d(out_channels))

        # Remember that the stride is the same as the kernel_size, just like
        # the max pooling layers
        self.main_unpool1 = nn.MaxUnpool3d(kernel_size=2)

        # Extension branch - 1x1 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution. Number
        # of channels is doubled.

        # 1x1 projection convolution with stride 1
        self.ext_conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels, internal_channels, kernel_size=1, bias=bias),
            nn.BatchNorm3d(internal_channels), activation())

        # Transposed convolution
        # self.ext_tconv1 = nn.ConvTranspose3d(
        #     internal_channels,
        #     internal_channels,
        #     kernel_size=2,
        #     stride=2,
        #     bias=bias)
        self.ext_tconv1 = nn.Sequential(
            nn.Conv3d(
                internal_channels,
                internal_channels * 2**3,
                kernel_size=3,
                padding=1,
                bias=bias),
            PixelShuffle(2))

        self.ext_tconv1_bnorm = nn.BatchNorm3d(
            affine=True, num_features=internal_channels)
        self.ext_tconv1_activation = activation()

        # 1x1 expansion convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv3d(
                internal_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm3d(out_channels), activation())

        self.ext_regul = nn.Dropout3d(p=dropout_prob)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()

    def forward(self, x, max_indices, output_size):
        # Main branch shortcut
        main = self.main_conv1(x)
        main = self.main_unpool1(
            main, max_indices, output_size=output_size)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_tconv1(ext)  # , output_size=output_size)
        ext = self.ext_tconv1_bnorm(ext)
        ext = self.ext_tconv1_activation(ext)
        ext = self.ext_conv2(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext

        return self.out_activation(out)


class ENet(BaseNet):
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
                 sh_order,
                 embed=[32, 64, 128, 256],
                 encoder_relu=False,
                 decoder_relu=True,
                 return_dict_layers=False):

        super().__init__(sh_order=sh_order,
                         embed=embed,
                         encoder_relu=encoder_relu,
                         decoder_relu=decoder_relu)

        self.inputs = ['sh', 'mean_b0']
        self.outputs = ['sh_fake', 'mean_b0_fake', 'alpha', 'beta']

        self.ncoef = int((sh_order + 2) * (sh_order + 1) / 2)
        self.ncoef += 1

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
        # self.regular1_3 = RegularBottleneck(
        #     embed[1], padding=1, dropout_prob=0.01, relu=encoder_relu)
        # self.regular1_4 = RegularBottleneck(
        #     embed[1], padding=1, dropout_prob=0.01, relu=encoder_relu)

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

        # Stage 3 - Encoder
        self.downsample3_0 = DownsamplingBottleneck(
            embed[2],
            embed[3],
            return_indices=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.regular3_1 = RegularBottleneck(
            embed[3], padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_2 = RegularBottleneck(
            embed[3], dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_3 = RegularBottleneck(
            embed[3],
            kernel_size=5,
            padding=2,
            asymmetric=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.dilated3_4 = RegularBottleneck(
            embed[3], dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)

        self.upsample4_0 = UpsamplingBottleneck(
            embed[3], embed[2], dropout_prob=0.1, relu=decoder_relu)
        self.regular4_1 = RegularBottleneck(
            embed[2], padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_2 = RegularBottleneck(
            embed[2], padding=1, dropout_prob=0.1, relu=decoder_relu)

        # Stage 4 - Decoder
        self.upsample5_0 = UpsamplingBottleneck(
            embed[2], embed[1], dropout_prob=0.1, relu=decoder_relu)
        self.regular5_1 = RegularBottleneck(
            embed[1], padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular5_2 = RegularBottleneck(
            embed[1], padding=1, dropout_prob=0.1, relu=decoder_relu)

        # Stage 5 - Decoder
        self.upsample6_0 = UpsamplingBottleneck(
            embed[1], embed[0], dropout_prob=0.1, relu=decoder_relu)
        self.regular6_1 = RegularBottleneck(
            embed[0], padding=1, dropout_prob=0.1, relu=decoder_relu)

        # self.transposed_conv = nn.ConvTranspose3d(
        #     embed[0],
        #     self.ncoef,
        #     kernel_size=3,
        #     stride=2,
        #     padding=1,
        #     bias=False)
        self.transposed_conv = nn.Sequential(
            nn.Conv3d(
                embed[0],
                self.ncoef * 2**3,
                kernel_size=5,
                padding=2,
                bias=True),
            PixelShuffle(2))

        self.conv6_2 = nn.Sequential(
            nn.Conv3d(
                self.ncoef,
                self.ncoef + 45,
                kernel_size=5,
                padding=2,
                bias=True),
            nn.BatchNorm3d(self.ncoef + 45),
            nn.ReLU(),
            nn.Conv3d(
                self.ncoef + 45,
                self.ncoef + 45,
                kernel_size=1,
                bias=True))

        self.convout_1 = nn.Sequential(
            nn.Conv3d(
                embed[0],
                self.ncoef + 45,
                kernel_size=1,
                bias=True),
            nn.Tanh(),
            nn.Upsample(scale_factor=2, mode='trilinear'))

        self.convout_2 = nn.Sequential(
            nn.Conv3d(
                embed[1],
                self.ncoef + 45,
                kernel_size=1,
                bias=True),
            nn.Tanh(),
            nn.Upsample(scale_factor=4, mode='trilinear'))

        self.convout_3 = nn.Sequential(
            nn.Conv3d(
                embed[2],
                self.ncoef + 45,
                kernel_size=1,
                bias=True),
            nn.Tanh(),
            nn.Upsample(scale_factor=8, mode='trilinear'))

        self.convout_4 = nn.Sequential(
            nn.Conv3d(
                embed[3],
                self.ncoef + 45,
                kernel_size=1,
                bias=True),
            nn.Tanh(),
            nn.Upsample(scale_factor=16, mode='trilinear'))

        self.activation_6 = nn.Tanh()

    def forward(self, sh, mean_b0):
        # Initial block
        inputs = torch.cat([mean_b0, sh], dim=-1)

        x = inputs.permute((0, 4, 1, 2, 3))

        out_dict = dict()

        input_size = x.size()

        x = self.initial_block(x)

        # Stage 1 - Encoder
        x_beforedown_1 = x  # self.norm_beforedown_1(x)
        stage1_input_size = x.size()
        x, max_indices1_0 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        # x = self.regular1_3(x)
        # x = self.regular1_4(x)

        # Stage 2 - Encoder
        x_beforedown_2 = x  # self.norm_beforedown_2(x)
        stage2_input_size = x.size()
        x, max_indices2_0 = self.downsample2_0(x)

        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)

        # Stage 3 - Encoder
        x_beforedown_3 = x  # self.norm_beforedown_3(x)
        stage3_input_size = x.size()
        x, max_indices3_0 = self.downsample3_0(x)

        x = self.regular3_1(x)
        x = self.dilated3_2(x)
        x = self.asymmetric3_3(x)
        x = self.dilated3_4(x)

        x_out_4 = self.convout_4(x)

        # Stage 5 - Decoder
        x = self.upsample4_0(x, max_indices3_0, output_size=stage3_input_size)
        x = x + x_beforedown_3

        x = self.regular4_1(x)
        x = self.regular4_2(x)

        x_out_3 = self.convout_3(x)

        # Stage 5 - Decoder
        x = self.upsample5_0(x, max_indices2_0, output_size=stage2_input_size)
        x = x + x_beforedown_2

        x = self.regular5_1(x)
        x = self.regular5_2(x)

        x_out_2 = self.convout_2(x)

        # Stage 6 - Decoder
        x = self.upsample6_0(x, max_indices1_0, output_size=stage1_input_size)
        x = x + x_beforedown_1
        x = self.regular6_1(x)

        x_out_1 = self.convout_1(x)

        x = self.transposed_conv(x)
        # x = self.transposed_conv(x, output_size=input_size)
        x = self.conv6_2(x)

        x = self.activation_6(x)

        x_final = (x + x_out_1 + x_out_2 + x_out_3 + x_out_4) / 5.
        x_final = x_final.permute((0, 2, 3, 4, 1))

        fodf = x_final[..., self.ncoef:]
        beta = x_final[..., :self.ncoef] * 0.3

        out_dict['beta'] = beta
        #out_dict['alpha'] = alpha

        #x = inputs * (1 + alpha) + beta
        x = inputs + beta

        sh_pred = x[..., 1:]
        mean_b0_pred = x[..., :1]

        out_dict['sh_fake'] = sh_pred
        out_dict['fodf_sh_fake'] = fodf
        out_dict['mean_b0_fake'] = mean_b0_pred

        return out_dict
