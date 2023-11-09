import torch
import torch.nn as nn
import torch.nn.functional as F

class MRINet(nn.Module):
    def __init__(self, channel_number=[64, 128, 256, 512, 1024, 2048], layer_number=[2, 2, 3, 3, 3, 3], dropout=True):
        super(MRINet, self).__init__()
        n_layer = len(channel_number) - 1
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = channel_number[i-1]
            out_channel = channel_number[i]
            self.feature_extractor.add_module('conv_%d_0' % i,
                                              self.conv_layer(in_channel,
                                                              out_channel,
                                                              maxpool=False,
                                                              kernel_size=3,
                                                              padding=1))
            for j in range(1, layer_number[i]):
                if j < layer_number[i] - 1:
                    self.feature_extractor.add_module('conv_%d_%d' % (i, j),
                                                      self.conv_layer(out_channel,
                                                                      out_channel,
                                                                      maxpool=False,
                                                                      kernel_size=3,
                                                                      padding=1))
                else:
                    self.feature_extractor.add_module('conv_%d_%d' % (i, j),
                                                      self.conv_layer(out_channel,
                                                                      out_channel,
                                                                      maxpool=True,
                                                                      kernel_size=3,
                                                                      padding=1))
        self.classifier = nn.Sequential()
        avg_shape = [3, 3, 3]
        self.classifier.add_module('average_pool', nn.AvgPool3d(avg_shape))
        if dropout is True:
            self.classifier.add_module('dropout', nn.Dropout(0.5))
        in_channel = out_channel
        out_channel = channel_number[-1]
        self.classifier.add_module('conv_%d_0' % n_layer,
                                       nn.Conv3d(in_channel, out_channel, padding=1, kernel_size=3))
        for i in range(1, layer_number[-1]):
            self.classifier.add_module('conv_%d_%d' % (n_layer, i),
                                       nn.Conv3d(out_channel, out_channel, padding=1, kernel_size=3))
        self.fc = nn.Linear(out_channel, 1)

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
        return layer

    def forward(self, x):
        x_f = self.feature_extractor(x)
        x = self.classifier(x_f)
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

