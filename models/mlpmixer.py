import torch
from torch import nn
from torch.nn import functional as F


class MlpBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim):
        super(MlpBlock, self).__init__()
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, mlp_dim)

    def forward(self, x):
        y = self.fc1(x)
        y = self.gelu(y)
        return self.fc2(y)


class MixerBlock(nn.Module):
    def __init__(self, tokens_mlp_dim, channels_mlp_dim, tokens_hidden_dim, channels_hidden_dim):
        super(MixerBlock, self).__init__()
        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.tokens_mlpblock = MlpBlock(tokens_mlp_dim, tokens_hidden_dim)
        self.channels_mlpblock = MlpBlock(channels_mlp_dim, channels_hidden_dim)

    def forward(self, x):
        # token mixing
        y = self.norm(x)
        y = y.transpose(1, 2)
        y = self.tokens_mlpblock(y)
        # channel mixing
        y = y.transpose(1, 2)
        x = x + y
        y = self.norm(x)
        out = x + self.channels_mlpblock(y)
        return out


class MlpMixer(nn.Module):
    def __init__(self, in_dim=1, num_classes=1, num_blocks=8, patch_size=16, tokens_hidden_dim=300,
                 channels_hidden_dim=300, tokens_mlp_dim=300,
                 channels_mlp_dim=300, dropout_p=0.):
        super(MlpMixer, self).__init__()
        self.patch_size = patch_size
        self.tokens_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim
        self.dropout_p = dropout_p

        self.embed = nn.Conv3d(in_dim, channels_mlp_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(channels_mlp_dim)

        self.mixer_blocks = nn.Sequential()
        for i in range(num_blocks):
            self.mixer_blocks.add_module('mixer_%d' % i, MixerBlock(tokens_mlp_dim, channels_mlp_dim, tokens_hidden_dim,
                                                                    channels_hidden_dim))

        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(channels_mlp_dim, num_classes)

    def forward(self, x):
        y = self.embed(x)
        n, c, d, h, w = y.shape
        y = y.view(n, c, -1).transpose(1, 2)  # n, tokens, channels
        if self.tokens_mlp_dim != y.shape[1]:
            raise ValueError(
                f'tokens_mlp_dim={self.tokens_mlp_dim}, y.shape[1]={y.shape[1]}. tokens_mlp_dim is not correct.')
        y = self.mixer_blocks(y)
        y = self.norm(y)
        y = torch.mean(y, dim=1, keepdim=False)
        if self.dropout_p > 0:
            y = self.dropout(y)
        out = self.fc(y)
        return out
