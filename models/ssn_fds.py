import torch
import torch.nn as nn
import torch.nn.functional as F
from fds import FDS
import math
from .ssn import SSN


class SSN_FDS(SSN):

    def __init__(self, depth=1, classes=4, fds=False, bucket_num=100, bucket_start=3, start_update=0, start_smooth=1,
                 kernel='gaussian', ks=9, sigma=1, momentum=0.9):
        super(SSN_FDS, self).__init__(depth=depth, classes=classes)

        if fds:
            self.FDS = FDS(
                feature_dim=768, bucket_num=bucket_num, bucket_start=bucket_start,
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

        x = self.fc1(encoding_s)

        if self.training and self.fds:
            return x, encoding
        return x
