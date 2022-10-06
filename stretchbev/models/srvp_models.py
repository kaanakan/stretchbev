import torch

import torch.nn as nn


def activation_factory(name):
    """
    Returns the activation layer corresponding to the input activation name.
    Parameters
    ----------
    name : str
        'relu', 'leaky_relu', 'elu', 'sigmoid', or 'tanh'. Adds the corresponding activation function after the
        convolution.
    Returns
    -------
    torch.nn.Module
        Element-wise activation layer.
    """
    if name == 'relu':
        return nn.ReLU(inplace=True)
    if name == 'leaky_relu':
        return nn.LeakyReLU(0.2, inplace=True)
    if name == 'elu':
        return nn.ELU(inplace=True)
    if name == 'sigmoid':
        return nn.Sigmoid()
    if name == 'tanh':
        return nn.Tanh()
    raise ValueError(f'Activation function \'{name}\' not yet implemented')


def make_conv_block(conv, activation, bn=True):
    """
    Supplements a convolutional block with activation functions and batch normalization.
    Parameters
    ----------
    conv : torch.nn.Module
        Convolutional block.
    activation : str
        'relu', 'leaky_relu', 'elu', 'sigmoid', 'tanh', or 'none'. Adds the corresponding activation function, or no
        activation if 'none' is chosen, after the convolution.
    bn : bool
        Whether to add batch normalization after the activation.
    Returns
    -------
    torch.nn.Sequential
        Sequence of the input convolutional block, the potentially chosen activation function, and the potential batch
        normalization.
    """
    out_channels = conv.out_channels
    modules = [conv]
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if activation != 'none':
        modules.append(activation_factory(activation))
    return nn.Sequential(*modules)


class VGG64Encoder(nn.Module):
    # a note: we are downsampling to 1/8 size
    # hard coded for now
    """
    Module implementing the VGG encoder.
    """

    def __init__(self, nc, nh, nf):
        """
        Parameters
        ----------
        nc : int
            Number of channels in the input data.
        nh : int
            Number of dimensions of the output flat vector.
        nf : int
            Number of filters per channel of the first convolution.
        """
        super(VGG64Encoder, self).__init__()
        self.conv = nn.ModuleList([
            nn.Sequential(
                make_conv_block(nn.Conv2d(nc, nf, 3, 1, 1, bias=False), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(nf, nf, 3, 1, 1, bias=False), activation='leaky_relu'),
            ),
            nn.Sequential(
                # nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                make_conv_block(nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(nf * 2, nf * 2, 3, 1, 1, bias=False), activation='leaky_relu'),
            ),
            nn.Sequential(
                # nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                make_conv_block(nn.Conv2d(nf * 2, nf * 2, 3, 1, 1, bias=False), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(nf * 2, nf * 2, 3, 1, 1, bias=False), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(nf * 2, nf * 2, 3, 1, 1, bias=False), activation='leaky_relu'),
            ),
            nn.Sequential(
                # nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                make_conv_block(nn.Conv2d(nf * 2, nf * 2, 3, 1, 1, bias=False), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(nf * 2, nf * 2, 3, 1, 1, bias=False), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False), activation='leaky_relu'),
            )
        ])
        self.last_conv = nn.Sequential(
            make_conv_block(nn.Conv2d(nf * 4, nh, 3, 1, 1, bias=False), activation='tanh')
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x, return_skip=False):
        """
        Parameters
        ----------
        x : torch.*.Tensor
            Encoder input.
        return_skip : bool
            Whether to extract and return, besides the network output, skip connections.
        Returns
        -------
        torch.*.Tensor
            Encoder output as a tensor of shape (batch, size).
        list
            Only if return_skip is True. List of skip connections represented as torch.*.Tensor corresponding to each
            convolutional block in reverse order (from the deepest to the shallowest convolutional block).
        """
        skips = []
        h = x
        for i, layer in enumerate(self.conv):

            if i in [1, 2]:
                h = self.maxpool(h)
            h_res = layer(h)
            print(i, h.shape, h_res.shape)
            h = h + h_res
            skips.append(h)
        h = self.last_conv(h)
        if return_skip:
            return h, skips[::-1]
        return h


class VGG64Decoder(nn.Module):
    # a note: we are upsampling to 1/8 size
    # hard coded for now
    """
    Module implementing the VGG decoder.
    """

    def __init__(self, nc, ny, nf, skip):
        """
        Parameters
        ----------
        nc : int
            Number of channels in the output shape.
        ny : int
            Number of dimensions of the input flat vector.
        nf : int
            Number of filters per channel of the first convolution of the mirror encoder architecture.
        skip : list
            List of torch.*.Tensor representing skip connections in the same order as the decoder convolutional
            blocks. Must be None when skip connections are not allowed.
        """
        super(VGG64Decoder, self).__init__()
        # decoder
        coef = 2 if skip else 1
        self.skip = skip
        self.first_upconv = nn.Sequential(
            make_conv_block(nn.ConvTranspose2d(ny, nf * 4, 3, 1, 1, bias=False), activation='leaky_relu'),
        )
        self.conv = nn.ModuleList([
            nn.Sequential(
                make_conv_block(nn.Conv2d(nf * 4 * coef, nf * 2, 3, 1, 1, bias=False), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(nf * 2, nf * 2, 3, 1, 1, bias=False), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(nf * 2, nf * 2, 3, 1, 1, bias=False), activation='leaky_relu'),
                # nn.Upsample(scale_factor=2, mode='nearest'),
            ),
            nn.Sequential(
                make_conv_block(nn.Conv2d(nf * 2 * coef, nf * 2, 3, 1, 1, bias=False), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(nf * 2, nf * 2, 3, 1, 1, bias=False), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(nf * 2, nf * 2, 3, 1, 1, bias=False), activation='leaky_relu'),
                # nn.Upsample(scale_factor=2, mode='nearest'),
            ),
            nn.Sequential(
                make_conv_block(nn.Conv2d(nf * 2 * coef, nf * 2, 3, 1, 1, bias=False), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=False), activation='leaky_relu'),
                # nn.Upsample(scale_factor=2, mode='nearest'),
            ),
            nn.Sequential(
                make_conv_block(nn.Conv2d(nf * coef, nf, 3, 1, 1, bias=False), activation='leaky_relu'),
                nn.ConvTranspose2d(nf, nc, 3, 1, 1, bias=False),
            ),
        ])
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, z, skip=None, sigmoid=False):
        """
        Parameters
        ----------
        z : torch.*.Tensor
            Decoder input.
        skip : list
            List of torch.*.Tensor representing skip connections in the same order as the decoder convolutional
            blocks. Must be None when skip connections are not allowed.
        sigmoid : bool
            Whether to apply a sigmoid at the end of the decoder.
        Returns
        -------
        torch.*.Tensor
            Decoder output as a frame of shape (batch, channels, width, height).
        """
        assert skip is None and not self.skip or self.skip and skip is not None
        h = self.first_upconv(z)
        for i, layer in enumerate(self.conv):
            if skip is not None:
                h = torch.cat([h, skip[i]], 1)
            h_res = layer(h)
            h = h + h_res
            if i in [1, 2]:
                h = self.upsample(h)
        x_ = h
        if sigmoid:
            x_ = torch.sigmoid(x_)
        return x_


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, nlayers):
        super(ConvNet, self).__init__()

        layers = []
        in_c = in_channels
        for _ in range(nlayers - 1):
            layers += [
                make_conv_block(nn.Conv2d(in_c, out_channels, 3, 1, 1, bias=False), activation='leaky_relu')
            ]
            in_c = out_channels
        layers += [SELayer(in_c)]
        layers += [make_conv_block(nn.Conv2d(in_c, out_channels, 3, 1, 1, bias=True), activation='none', bn=False)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
