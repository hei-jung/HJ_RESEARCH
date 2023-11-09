import torch
from torch import nn
import torch.nn.functional as F
from .segmentation.vnet import VNet


class VNetRegressor(VNet):

    def __init__(self, n_channels, n_classes=2, n_filters=16, normalization='none', out_features=1,
                 fc_features=[512], activation_fcn=True):
        super(VNetRegressor, self).__init__(n_channels, n_classes, n_filters, normalization)
        self.activation_fcn = activation_fcn
        self.fc = nn.Sequential()
        n = len(fc_features)
        for i in range(n - 1):
            self.fc.add_module('linear_%d' % (i + 1), nn.Linear(fc_features[i], fc_features[i + 1]))
            self.fc.add_module('relu', nn.LeakyReLU())
        self.fc.add_module('linear_out', nn.Linear(fc_features[n - 1], out_features))

    def forward(self, inputs):
        # vnet encoder
        x1 = self.block_one(inputs)
        x1_dw = self.block_one_dw(x1)
        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)
        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)
        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)
        x = x4_dw.view(x4_dw.size(0), -1)
        out = self.fc(x)
        if self.activation_fcn:
            out = F.leaky_relu(out)
        return out
