# Copyright 2020 Mickael Chen, Edouard Delasalles, Jean-Yves Franceschi, Patrick Gallinari, Sylvain Lamprier

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch

import torch.distributions as distrib
import torch.nn as nn
import torch.nn.functional as F


def init_weight(m, init_type='normal', init_gain=0.02):
    """
    Initializes the input module with the given parameters.

    Only deals with `Conv2d`, `ConvTranspose2d`, `Linear` and `BatchNorm2d` layers.

    Parameters
    ----------
    m : torch.nn.Module
        Module to initialize.
    init_type : str
        'normal', 'xavier', 'kaiming', or 'orthogonal'. Orthogonal initialization types for convolutions and linear
        operations. Ignored for batch normalization which uses a normal initialization.
    init_gain : float
        Gain to use for the initialization.
    """
    classname = m.__class__.__name__
    if classname in ('Conv2d', 'ConvTranspose2d', 'Linear'):
        if init_type == 'normal':
            nn.init.normal_(m.weight.data, 0.0, init_gain)
        elif init_type == 'xavier':
            nn.init.xavier_normal_(m.weight.data, gain=init_gain)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(m.weight.data, gain=init_gain)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname == 'BatchNorm2d':
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


def make_normal_from_raw_params(raw_params, scale_stddev=1, dim=2, eps=1e-8, max_log_sigma=-10000, min_log_sigma=10000):
    """
    Creates a normal distribution from the given parameters.

    Parameters
    ----------
    raw_params : torch.*.Tensor
        Tensor containing the Gaussian mean and a raw scale parameter on a given dimension.
    scale_stddev : float
        Multiplier of the final scale parameter of the Gaussian.
    dim : int
        Dimensions of raw_params so that the first half corresponds to the mean, and the second half to the scale.
    eps : float
        Minimum possible value of the final scale parameter.

    Returns
    -------
    torch.distributions.Normal
        Normal distribution with the input mean and eps + softplus(raw scale) * scale_stddev as standard deviation.
    """
    dim = 2 if len(raw_params.shape) == 5 else 1
    loc, raw_scale = torch.chunk(raw_params, 2, dim)
    assert loc.shape[dim] == raw_scale.shape[dim], f'{loc.shape[dim]}, {raw_scale.shape[dim]}'
    # raw_scale = torch.clamp(raw_scale, min_log_sigma, max_log_sigma)
    scale = F.softplus(raw_scale) + eps
    normal = distrib.Normal(loc, scale * scale_stddev)
    return normal


def rsample_normal(raw_params, scale_stddev=1, max_log_sigma=-10000, min_log_sigma=10000):
    """
    Samples from a normal distribution with given parameters.

    Parameters
    ----------
    raw_params : torch.*.Tensor
        Tensor containing a Gaussian mean and a raw scale parameter on its last dimension.
    scale_stddev : float
        Multiplier of the final scale parameter of the Gaussian.

    Returns
    -------
    torch.*.Tensor
        Sample from the normal distribution with the input mean and eps + softplus(raw scale) * scale_stddev as
        standard deviation.
    """

    normal = make_normal_from_raw_params(raw_params, scale_stddev=scale_stddev)
    sample = normal.rsample()
    return sample


def neg_logprob(loc, data, scale=1):
    """
    Computes the negative log density function of a given input with respect to a normal distribution created from
    given parameters.

    Parameters
    ----------
    loc : torch.*.Tensor
        Tensor containing the mean of the Gaussian on its last dimension.
    data : torch.*.Tensor
        Computes the log density function of this tensor with respect to the Gaussian distribution of input mean and
        standard deviation.
    scale : float
        Standard deviation of the Gaussian.

    Returns
    -------
    torch.*.Tensor
        Sample from the normal distribution with the input mean and eps + softplus(raw scale) * scale_stddev as
        standard deviation.
    """
    obs_distrib = distrib.Normal(loc, scale)
    return -obs_distrib.log_prob(data)
