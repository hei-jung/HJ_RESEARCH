import torch
from torch import nn
import torch.nn.functional as F
from fds import FDS
import math
from .pretrained_cnn import RegNet


class RegFdsNet(RegNet):
    def __init__(self, n_channels, n_classes=2, n_filters=16, normalization='none', avg_shape=[5, 6, 5], out_features=1,
                 dropout_p=0., activation_fcn=True, fds=False, bucket_num=100, bucket_start=3, start_update=0,
                 start_smooth=1, kernel='gaussian', ks=9, sigma=1, momentum=0.9):
        super(RegFdsNet, self).__init__(n_channels, n_classes, n_filters, normalization, avg_shape, out_features,
                                        dropout_p, activation_fcn)

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
        x1 = self.block_one(x)
        x1_dw = self.block_one_dw(x1)
        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)
        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)
        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x = self.classifier(x4_dw)
        encoding = x.view(x.size(0), -1)
        encoding_s = encoding

        if self.training and self.fds:
            if epoch >= self.start_smooth:
                encoding_s = self.FDS.smooth(encoding_s, targets, epoch)

        x = self.fc(encoding_s)
        if self.activation_fcn:
            x = F.leaky_relu(x)

        if self.training and self.fds:
            return x, encoding
        return x
