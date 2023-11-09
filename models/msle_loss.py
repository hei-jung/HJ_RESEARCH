import torch
from torch import nn

""""Mean Squared Logarithmic Error"""


class MSLELoss(nn.Module):
    def __init__(self, eps=1e-03):
        super(MSLELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = self.mse(torch.log(yhat + self.eps), torch.log(y + self.eps))
        return loss
