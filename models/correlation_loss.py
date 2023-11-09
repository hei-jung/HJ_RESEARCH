import torch
from torch import nn
import torch.nn.functional as F


class CorrelationLoss(nn.Module):
    def forward(self, inputs, targets, eps=1e-12):
        x = inputs.clone()
        y = targets.clone()
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        cov = torch.sum(vx * vy)
        corr = cov / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + eps)
        corr = torch.max(torch.min(corr, torch.tensor(1)), torch.tensor(-1))
        return 1 - corr ** 2


class CorrelationMetric(nn.Module):
    def forward(self, inputs, targets, eps=1e-12):
        x = torch.Tensor(inputs)
        y = torch.Tensor(targets)
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        cov = torch.sum(vx * vy)
        corr = cov / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + eps)
        return corr
