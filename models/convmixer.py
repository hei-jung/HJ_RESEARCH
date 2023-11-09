from torch import nn


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixer(nn.Module):
    def __init__(self, h=256, depth=4, kernel_size=9, patch_size=7, n_classes=1):
        super(ConvMixer, self).__init__()

        self.embed = nn.Sequential(
            nn.Conv3d(1, h, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm3d(h))

        self.conv_mixer = nn.Sequential()

        for i in range(depth):
            self.conv_mixer.add_module("mixer_%d" % (i + 1),
                                       nn.Sequential(Residual(
                                           nn.Sequential(nn.Conv3d(h, h, kernel_size, groups=h, padding="same"),
                                                         nn.GELU(),
                                                         nn.BatchNorm3d(h))),
                                           nn.Conv3d(h, h, kernel_size=1),
                                           nn.GELU(),
                                           nn.BatchNorm3d(h)))

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(h, n_classes)

    def forward(self, x):
        x = self.embed(x)
        x = self.conv_mixer(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x
