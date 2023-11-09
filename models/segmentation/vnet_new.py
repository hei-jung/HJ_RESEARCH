import torch
from torch import nn


class GroupNorm3D(nn.Module):
    def __init__(self, num_features, num_groups=16, eps=1e-5):
        super(GroupNorm3D, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1, 1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N, C, H, W, D = x.size()
        G = self.num_groups
        assert C % G == 0

        x = x.view(N, G, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)
        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, H, W, D)
        return x * self.weight + self.bias


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none', expand_chan=False):
        super(ResidualConvBlock, self).__init__()

        self.expand_chan = expand_chan
        if self.expand_chan:
            ops = []

            ops.append(nn.Conv3d(n_filters_in, n_filters_out, 1))

            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            if normalization == 'groupnorm':
                ops.append(GroupNorm3D(n_filters_out))

            # ops.append(nn.Dropout3d(p=0.5,inplace=False))

            ops.append(nn.ReLU(inplace=True))

            self.conv_expan = nn.Sequential(*ops)

        ops = []
        for i in range(n_stages):
            if normalization != 'none':
                ops.append(nn.Conv3d(n_filters_in, n_filters_out, 3, padding=1))
                if normalization == 'batchnorm':
                    ops.append(nn.BatchNorm3d(n_filters_out))
                if normalization == 'groupnorm':
                    ops.append(GroupNorm3D(n_filters_out))
            else:
                ops.append(nn.Conv3d(n_filters_in, n_filters_out, 3, padding=1))

            # ops.append(nn.Dropout3d(p=0.5,inplace=False))
            ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        if self.expand_chan:
            x = self.conv(x) + self.conv_expan(x)
        else:
            x = (self.conv(x) + x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            if normalization == 'groupnorm':
                ops.append(GroupNorm3D(n_filters_out))
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
        # ops.append(nn.Dropout3d(p=0.5,inplace=False))
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            if normalization == 'groupnorm':
                ops.append(GroupNorm3D(n_filters_out))
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
        # ops.append(nn.Dropout3d(p=0.5,inplace=False))
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class VNet_New(nn.Module):
    def __init__(self, n_channels, n_classes, n_filters=16, normalization='none', n_blocks=4):
        super(VNet_New, self).__init__()

        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        if n_channels > 1:
            self.encoder.add_module('block_1', ResidualConvBlock(1, n_channels, n_filters, normalization=normalization))
        else:
            n_stages = 1
            for i in range(1, n_blocks + 1):
                self.encoder.add_module('block_%d' % i,
                                        ResidualConvBlock(n_stages, n_channels, n_filters, normalization=normalization))
                self.encoder.add_module('block_%d_dw' % i,
                                        DownsamplingConvBlock(n_filters, n_filters * 2, normalization=normalization))
                n_filters *= 2
                n_channels = n_filters
                if n_stages < 3: n_stages += 1
            if n_blocks < 3 and n_stages == 3: n_stages -= 1
            for i in range(n_blocks + 1, 2 * n_blocks + 1):
                if i == 2 * n_blocks: n_stages -= 1
                self.decoder.add_module('block_%d' % i,
                                        ResidualConvBlock(n_stages, n_filters, n_filters, normalization=normalization))
                self.decoder.add_module('block_%d_up' % i,
                                        UpsamplingDeconvBlock(n_filters, n_filters // 2, normalization=normalization))
                n_filters //= 2
            i += 1
            self.decoder.add_module('block_%d' % i,
                                    ResidualConvBlock(1, n_filters, n_filters, normalization=normalization))
            self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

    def forward(self, inputs):
        x = []
        for i, (name, block) in enumerate(self.encoder.named_children()):
            # after iteration, x: x1, x1_dw, x2, x2_dw, x3, x3_dw, x4, x4_dw
            if i > 0:
                inputs = x[-1]
            out = block(inputs)
            x.append(out)
        y = x[-1]
        for i, (name, block) in enumerate(self.decoder.named_children()):
            y = block(y)
            if i % 2:  # if i is odd
                x.pop()  # pop dw output
                y = y + x.pop()  # up + x
        out = self.out_conv(y)
        return out
