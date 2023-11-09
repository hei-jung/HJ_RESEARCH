import torch
from torch import nn
import torch.nn.functional as F

"""
Ref link: https://github.com/KellerJordan/ResNet-PyTorch-CIFAR10
"""


def conv_3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True))


class ResNet(nn.Module):

    def __init__(self, in_channels=1, num_classes=1, num_layers=5, res_option='A', use_dropout=False):
        super(ResNet, self).__init__()
        self.res_option = res_option
        self.use_dropout = use_dropout
        self.conv1 = conv_3d(in_channels, 16)
        self.layer1 = self._make_layer(num_layers, 16, 16, 1)
        self.layer2 = self._make_layer(num_layers, 32, 16, 2)
        self.layer3 = self._make_layer(num_layers, 64, 32, 2)
        self.avgpool = nn.AvgPool3d(8)
        self.fc = nn.Linear(21952, num_classes)

    def _make_layer(self, layer_count, channels, channels_in, stride):
        return nn.Sequential(
            ResBlock(channels, channels_in, stride, res_option=self.res_option, use_dropout=self.use_dropout),
            *[ResBlock(channels) for _ in range(layer_count - 1)])

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ResBlock(nn.Module):

    def __init__(self, num_filters, channels_in=None, stride=1, res_option='A', use_dropout=False):
        super(ResBlock, self).__init__()

        # use 1x1 convolution for downsampling
        if not channels_in or channels_in == num_filters:
            channels_in = num_filters
            self.projection = None
        else:
            if res_option == 'A':
                self.projection = IdentityPadding(num_filters, channels_in, stride)
            elif res_option == 'B':
                self.projection = ConvProjection(num_filters, channels_in, stride)
            elif res_option == 'C':
                self.projection = AvgPoolPadding(num_filters, channels_in, stride)
        self.use_dropout = use_dropout

        self.conv1 = conv_3d(channels_in, num_filters, stride=stride)
        self.conv2 = nn.Conv3d(num_filters, num_filters, 3, stride=1, padding=1)
        self.bn = nn.BatchNorm3d(num_filters)
        if self.use_dropout:
            self.dropout = nn.Dropout(inplace=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(out)
        if self.use_dropout:
            out = self.dropout(out)
        if self.projection:
            residual = self.projection(x)
        out += residual
        out = self.relu(out)
        return out


# various projection options to change number of filters in residual connection
# option A from paper
class IdentityPadding(nn.Module):

    def __init__(self, num_filters, channels_in, stride):
        super(IdentityPadding, self).__init__()
        # with kernel_size=1, max pooling is equivalent to identity mapping with stride
        self.identity = nn.MaxPool3d(1, stride=stride)
        self.num_zeros = num_filters - channels_in

    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, 0, 0, self.num_zeros))
        out = self.identity(out)
        return out


# option B from paper
class ConvProjection(nn.Module):

    def __init__(self, num_filters, channels_in, stride):
        super(ConvProjection, self).__init__()
        self.conv = nn.Conv3d(channels_in, num_filters, kernel_size=1, stride=stride)

    def forward(self, x):
        out = self.conv(x)
        return out


# experimental option C
class AvgPoolPadding(nn.Module):

    def __init__(self, num_filters, channels_in, stride):
        super(AvgPoolPadding, self).__init__()
        self.identity = nn.AvgPool3d(stride, stride=stride)
        self.num_zeros = num_filters - channels_in

    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, 0, 0, self.num_zeros))
        out = self.identity(out)
        return out

    
def resnet20(**kwargs):
    # 1 + 2*(3 + 3 + 3) + 1 = 20
    model = ResNet(num_layers=3, **kwargs)
    return model


def resnet26(**kwargs):
    # 1 + 2*(4 + 4 + 4) + 1 = 26
    model = ResNet(num_layers=4, **kwargs)
    return model
    
    
def resnet32(**kwargs):
    # 1 + 2*(5 + 5 + 5) + 1 = 32
    model = ResNet(num_layers=5, **kwargs)
    return model