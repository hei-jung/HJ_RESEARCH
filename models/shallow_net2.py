from torch import nn


class ShallowNet2(nn.Module):

    def __init__(self, depth=1, classes=4):
        super(ShallowNet2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=depth, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU())

        self.classifier = nn.Sequential(
            nn.AvgPool3d([15, 6, 9]),
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=1, stride=1)
        )

        self.fc1 = nn.Linear(11200, classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
