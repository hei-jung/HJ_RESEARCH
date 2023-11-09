import torch
import torch.nn as nn
import torch.nn.functional as F


# Simpler simple net

class SSN(nn.Module):

    def __init__(self, depth=1, classes=4, drop_p=0.5):
        super(SSN, self).__init__()

        self.feature_extractor = nn.Sequential()

        conv1 = nn.Sequential(
            nn.Conv3d(in_channels=depth, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.ReLU())

        conv2 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.ReLU())

        conv3 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=1, stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU())

        self.feature_extractor.add_module('conv1', conv1)
        self.feature_extractor.add_module('conv2', conv2)
        self.feature_extractor.add_module('conv3', conv3)

        self.classifier = nn.Sequential(
            nn.AvgPool3d([15, 6, 9]),
            nn.Dropout(p=drop_p, inplace=False),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=1, stride=1)
        )

        self.fc1 = nn.Linear(768, classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
