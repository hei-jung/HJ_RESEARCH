import torch
from torch import nn
import torch.nn.functional as F
from .segmentation.vnet import VNet


class RegNet(VNet):

    def __init__(self, n_channels, n_classes=2, n_filters=16, normalization='none', avg_shape=[5, 6, 5], out_features=1,
                 dropout_p=0., activation_fcn=True):
        super(RegNet, self).__init__(n_channels, n_classes, n_filters, normalization)
        self.activation_fcn = activation_fcn
        self.classifier = nn.Sequential(
            nn.AvgPool3d(avg_shape),  # [5,6,5]
            nn.Dropout(p=dropout_p, inplace=False),
            nn.Conv3d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0)
        )
        self.fc = nn.Sequential()
        self.fc.add_module('linear_1', nn.Linear(512, out_features))

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
        x = self.classifier(x4_dw)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        if self.activation_fcn:
            out = F.leaky_relu(out)
        return out
