import torch
from torch import nn
import torch.nn.functional as F

"""reference code: https://deep-learning-study.tistory.com/706"""


class BinaryDiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1e-05):
        # binary cross entropy loss
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='sum')

        inputs = F.sigmoid(inputs)
        intersection = (inputs * targets).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

        # dice coefficient
        dice = 2.0 * (intersection + smooth) / (union + smooth)

        # dice loss
        dice_loss = 1.0 - dice

        # total loss
        loss = bce + dice_loss

        return loss.sum(), dice.sum()
