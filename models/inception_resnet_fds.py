import torch
from torch import nn
import torch.nn.functional as F
from fds import FDS
import math
from .inception_resnet_v2 import InceptionResnetV2


class InceptionResnetV2FDS(InceptionResnetV2):
    def __init__(self, num_classes=1000, in_channels=3, drop_rate=0., output_stride=32, fds=False, bucket_num=100,
                 bucket_start=3, start_update=0, start_smooth=1, kernel='gaussian', ks=9, sigma=1, momentum=0.9):
        super(InceptionResnetV2FDS, self).__init__(num_classes=num_classes, in_channels=in_channels,
                                                   drop_rate=drop_rate, output_stride=output_stride)

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
        x = self.forward_features(x)
        x = self.avgpool(x)

        encoding = x.view(x.size(0), -1)
        encoding_s = encoding

        if self.training and self.fds:
            if epoch >= self.start_smooth:
                encoding_s = self.FDS.smooth(encoding_s, targets, epoch)

        if self.drop_rate > 0:
            encoding_s = F.dropout(encoding_s, p=self.drop_rate, training=self.training)

        x = self.fc(encoding_s)

        if self.training and self.fds:
            return x, encoding
        return x


def inception_resnet_v2_fds(**kwargs):
    return InceptionResnetV2FDS(**kwargs)
