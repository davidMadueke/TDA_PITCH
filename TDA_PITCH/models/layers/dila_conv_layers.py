import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import numpy as np
from torchaudio.transforms import Spectrogram

# TAKEN FROM THE IMPLEMENTATION OF MRDC IN THE HARMOF0 Model (https://github.com/WX-Wei/HarmoF0.git)
# Multiple Rates Dilated Convolution
class MRDConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_list=[0, 12, 19, 24, 28, 31, 34, 36]):
        super().__init__()
        self.dilation_list = dilation_list
        self.conv_list = []
        for i in range(len(dilation_list)):
            self.conv_list += [nn.Conv2d(in_channels, out_channels, kernel_size=[1, 1])]
        self.conv_list = nn.ModuleList(self.conv_list)

    def forward(self, specgram):
        # input [b x C x T x n_freq]
        # output: [b x C x T x n_freq]
        dilation = self.dilation_list[0]
        y = self.conv_list[0](specgram)
        y = F.pad(y, pad=[0, dilation])
        y = y[:, :, :, dilation:]
        for i in range(1, len(self.conv_list)):
            dilation = self.dilation_list[i]
            x = self.conv_list[i](specgram)
            # => [b x T x (n_freq + dilation)]
            # x = F.pad(x, pad=[0, dilation])
            x = x[:, :, :, dilation:]
            n_freq = x.size()[3]
            y[:, :, :, :n_freq] += x

        return y


# Fixed Rate Dilated Casual Convolution
class FRDConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=[1, 3], dilation=[1, 1]) -> None:
        super().__init__()
        right = (kernel_size[1] - 1) * dilation[1]
        bottom = (kernel_size[0] - 1) * dilation[0]
        self.padding = nn.ZeroPad2d([0, right, 0, bottom])
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x):
        x = self.padding(x)
        x = self.conv2d(x)
        return x


def dila_conv_block(
        in_channel, out_channel,
        bins_per_octave,
        n_har,
        dilation_mode,
        dilation_rate,
        dil_kernel_size,
        kernel_size=[1, 3],
        padding=[0, 1],
):
    conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)
    batch_norm = nn.BatchNorm2d(out_channel)

    # dilation mode: 'log_scale', 'fixed'
    if (dilation_mode == 'log_scale'):
        a = np.log(np.arange(1, n_har + 1)) / np.log(2 ** (1.0 / bins_per_octave))
        dilation_list = a.round().astype(np.int)
        conv_log_dil = MRDConv(out_channel, out_channel, dilation_list)
        return nn.Sequential(
            conv, nn.ReLU(),
            conv_log_dil, nn.ReLU(),
            batch_norm,
            # pool
        )
    elif (dilation_mode == 'fixed_causal'):
        dilation_list = np.array([i * dil_kernel_size[1] for i in range(dil_kernel_size[1])])
        causal_conv = FRDConv(out_channel, out_channel, dil_kernel_size, dilation=[1, dilation_rate])
        return nn.Sequential(
            conv, nn.ReLU(),
            causal_conv, nn.ReLU(),
            batch_norm,
            # pool
        )
    elif (dilation_mode == 'fixed'):
        conv_dil = nn.Conv2d(out_channel, out_channel, kernel_size=dil_kernel_size, padding=[0, dilation_rate],
                             dilation=[1, dilation_rate])

        return nn.Sequential(
            conv, nn.ReLU(),
            conv_dil, nn.ReLU(),
            batch_norm,
            # pool
        )
    else:
        assert False, "unknown dilation type: " + dilation_mode