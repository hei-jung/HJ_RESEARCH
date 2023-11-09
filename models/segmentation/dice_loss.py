import torch
from torch import nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def forward(self, inputs, targets, eps=0.0001, index=2):
        num = (inputs * targets).sum(dim=4, keepdim=True).sum(dim=3, keepdim=True).sum(dim=2, keepdim=True)

        assert index == 1 or index == 2, "power must be 1 or 2."

        den1 = inputs.pow(index).sum(dim=4, keepdim=True).sum(dim=3, keepdim=True).sum(dim=2, keepdim=True)
        den2 = targets.pow(index).sum(dim=4, keepdim=True).sum(dim=3, keepdim=True).sum(dim=2, keepdim=True)

        dice = (2.0 * num / (den1 + den2 + eps))

        return (1.0 - dice).mean(), dice.mean()
