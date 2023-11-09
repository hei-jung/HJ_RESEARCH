import torch
from torch import nn
import torch.nn.functional as F
from .segmentation.vnet_new import VNet_New
from fds import FDS
import math


class Regressor(VNet_New):

    def __init__(self, n_channels, n_classes=2, n_filters=16, normalization='none', n_blocks=4, out_features=1,
                 fc_features=[512], dropout_p=0., activation_fcn=True, avg_shape=3,
                 fds=False, bucket_num=100, bucket_start=3, start_update=0, start_smooth=1, kernel='gaussian', ks=9,
                 sigma=1, momentum=0.9):
        super(Regressor, self).__init__(n_channels, n_classes, n_filters, normalization, n_blocks)
        self.activation_fcn = activation_fcn
        self.n_filters = (2 ** n_blocks) * n_filters
        self.conv_out = nn.Sequential()
        self.conv_out.add_module('average_pool', nn.AvgPool3d(avg_shape))
        if dropout_p != 0.:
            self.conv_out.add_module('dropout', nn.Dropout(dropout_p))
        self.conv_out.add_module('conv', nn.Conv3d(self.n_filters, 1, 1, padding=0))
        self.fc = nn.Sequential()
        n = len(fc_features)
        for i in range(n - 1):
            self.fc.add_module('linear_%d' % (i + 1), nn.Linear(fc_features[i], fc_features[i + 1]))
            self.fc.append(nn.ReLU())
        if dropout_p != 0.:
            self.fc.add_module('dropout', nn.Dropout(dropout_p))
        self.fc.add_module('linear_out', nn.Linear(fc_features[n - 1], out_features))
        if fds:
            self.FDS = FDS(
                feature_dim=fc_features[0], bucket_num=bucket_num, bucket_start=bucket_start,
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

    def forward(self, inputs, targets=None, epoch=None):
        x = self.encoder(inputs)
        x = self.conv_out(x)
        encoding = x.view(x.size(0), -1)
        encoding_s = encoding

        if self.training and self.fds:
            if epoch >= self.start_smooth:
                encoding_s = self.FDS.smooth(encoding_s, targets, epoch)

        out = self.fc(encoding_s)
        if self.activation_fcn:
            out = F.leaky_relu(out)

        if self.training and self.fds:
            return out, encoding
        return out
