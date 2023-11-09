import torch
import torch.nn as nn
import torch.nn.functional as F
from fds import FDS
import math
from .sfcn import SFCN


class SFCN_FDS(SFCN):
    def __init__(self, channel_number=[32, 64, 128, 256, 256, 64], avg_shape=[5, 2, 3], output_dim=1, dropout=True,
                 dropout_p=0.6, fds=False, bucket_num=100, bucket_start=3, start_update=0, start_smooth=1,
                 kernel='gaussian', ks=9, sigma=1, momentum=0.9):
        super(SFCN_FDS, self).__init__(channel_number=channel_number, avg_shape=avg_shape, output_dim=output_dim,
                                       dropout=dropout, dropout_p=dropout_p)
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
        x = self.feature_extractor(x)
        x = self.classifier(x)
        encoding = x.view(x.size(0), -1)
        encoding_s = encoding

        if self.training and self.fds:
            if epoch >= self.start_smooth:
                encoding_s = self.FDS.smooth(encoding_s, targets, epoch)

        x = self.fc(encoding_s)

        if self.training and self.fds:
            return x, encoding
        return x
