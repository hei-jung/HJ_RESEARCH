import torch
from torch import nn
from fds import FDS
import math
from .resnet import ResNet, BasicBlock, Bottleneck, get_inplanes


class ResNetFDS(ResNet):
    def __init__(self, block, layers, block_inplanes, in_channels=3, conv1_t_size=7, conv1_t_stride=1,
                 no_max_pool=False, shortcut_type='B', widen_factor=1.0, num_classes=400, fds=False, bucket_num=100,
                 bucket_start=3, start_update=0, start_smooth=1,
                 kernel='gaussian', ks=9, sigma=1, momentum=0.9):
        super(ResNetFDS, self).__init__(block=block, layers=layers, block_inplanes=block_inplanes,
                                        in_channels=in_channels, conv1_t_size=conv1_t_size,
                                        conv1_t_stride=conv1_t_stride, no_max_pool=no_max_pool,
                                        shortcut_type=shortcut_type, widen_factor=widen_factor, num_classes=num_classes)

        if fds:
            self.FDS = FDS(
                feature_dim=64, bucket_num=bucket_num, bucket_start=bucket_start,
                start_update=start_update, start_smooth=start_smooth, kernel=kernel, ks=ks, sigma=sigma,
                momentum=momentum
            )
        self.fds = fds
        self.start_smooth = start_smooth

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, targets=None, epoch=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        encoding = x.view(x.size(0), -1)
        encoding_s = encoding

        if self.training and self.fds:
            if epoch >= self.start_smooth:
                encoding_s = self.FDS.smooth(encoding_s, targets, epoch)

        x = self.fc(encoding_s)

        if self.training and self.fds:
            return x, encoding
        return x


def resnet_fds(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNetFDS(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNetFDS(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNetFDS(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNetFDS(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNetFDS(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNetFDS(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNetFDS(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model


def resnet20_fds(**kwargs):
    return ResNetFDS(BasicBlock, [1, 2, 5, 1], get_inplanes(), **kwargs)


def resnet22_fds(**kwargs):
    return ResNetFDS(BasicBlock, [2, 3, 3, 2], get_inplanes(), **kwargs)


def resnet26_fds(**kwargs):
    return ResNetFDS(BasicBlock, [3, 3, 3, 3], get_inplanes(), **kwargs)


def resnet30_fds(**kwargs):
    return ResNetFDS(BasicBlock, [2, 4, 6, 2], get_inplanes(), **kwargs)
