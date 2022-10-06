from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """2D convolution followed by
         - an optional normalisation (batch norm or instance norm)
         - an optional activation (ReLU, LeakyReLU, or tanh)
    """

    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, norm='bn', activation='lrelu',
                 bias=False, transpose=False):
        super().__init__()
        out_channels = out_channels or in_channels
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d if not transpose else partial(nn.ConvTranspose2d)
        self.conv = self.conv(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)

        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            raise ValueError('Invalid norm {}'.format(norm))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            raise ValueError('Invalid activation {}'.format(activation))

    def forward(self, x):
        x = self.conv(x)

        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResBlock(nn.Module):
    """Residual block:
       x -> Conv -> norm -> act. -> Conv -> norm -> act. -> ADD -> out
         |                                                   |
          ---------------------------------------------------
    """

    def __init__(self, in_channels, out_channels=None, norm='bn', activation='lrelu', bias=False):
        super().__init__()
        out_channels = out_channels or in_channels

        self.layers = nn.Sequential(OrderedDict([
            ('conv_1', ConvBlock(in_channels, in_channels, 3, stride=1, norm=norm, activation=activation, bias=bias)),
            ('conv_2', ConvBlock(in_channels, out_channels, 3, stride=1, norm=norm, activation=activation, bias=bias)),
            ('dropout', nn.Dropout2d(0.25)),
        ]))

        if out_channels != in_channels:
            self.projection = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.projection = None

    def forward(self, x):
        x_residual = self.layers(x)

        if self.projection:
            x = self.projection(x)
        return x + x_residual


class SmallEncoder(nn.Module):
    def __init__(self, nc, nh, nf):
        super(SmallEncoder, self).__init__()

        self.blocks = nn.ModuleList([
            ResBlock(nc, nf),
            ResBlock(nf, nf * 2),
            ResBlock(nf * 2, nf * 2),
            ResBlock(nf * 2, nf * 2),
            ResBlock(nf * 2, nf * 4)
        ])
        self.last_conv = nn.Sequential(
            ConvBlock(nf * 4, nh, 3, stride=1, activation='tanh')
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x, return_skip=False):
        h = x
        skips = []
        for i, layer in enumerate(self.blocks):
            if i in [1, 2]:
                h = self.maxpool(h)
            h = layer(h)
            skips.append(h)
        h = self.last_conv(h)
        if return_skip:
            return h, skips[::-1]
        return h


class SmallDecoder(nn.Module):
    def __init__(self, nc, nh, nf, skip):
        super(SmallDecoder, self).__init__()
        coef = 2 if skip else 1
        self.skip = skip

        self.first_upconv = ConvBlock(nc, nf * 4, stride=1, transpose=True)

        self.blocks = nn.ModuleList([
            ResBlock(nf * 4 * coef, nf * 2),
            ResBlock(nf * 2 * coef, nf * 2),
            ResBlock(nf * 2 * coef, nf * 2),
            ResBlock(nf * 2 * coef, nf),
            ResBlock(nf * coef, nf)
        ])
        self.last_conv = nn.Sequential(
            ConvBlock(nf * coef, nf, 3, stride=1),
            ConvBlock(nf, nh, 3, stride=1, transpose=True, bias=True, norm='none'),

        )
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, z, skip=None, sigmoid=False):
        assert skip is None and not self.skip or self.skip and skip is not None
        h = self.first_upconv(z)
        for i, layer in enumerate(self.blocks):
            # print(i, h.shape, skip[i].shape)
            if skip is not None:
                h = torch.cat([h, skip[i]], 1)
            h = layer(h)
            if i in [2, 3]:
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
    def __init__(self, in_c, out_c, nlayers):
        super(ConvNet, self).__init__()
        self.model = nn.Sequential(
            ResBlock(in_c, out_c),
            SELayer(out_c),
            ResBlock(out_c, out_c),
            SELayer(out_c),
            ConvBlock(out_c, out_c, 3, stride=1, bias=True, norm='none'),
        )

    def forward(self, x):
        return self.model(x)
