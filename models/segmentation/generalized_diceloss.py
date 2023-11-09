import torch
from torch import nn
from torch.autograd import Variable


# multi-label
def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    transposed.contiguous()
    # Flatten: (C, N, D, H, W) ->
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.reshape(C, tensor.size(0) * tensor.size(2) * tensor.size(3) * tensor.size(4))


class GeneralizedDiceLoss(nn.Module):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=True):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def forward(self, inputs, targets):
        # get probabilities from logits
        #        input = self.normalization(input)

        assert inputs.size() == targets.size(), "'input' and 'target' must have the same shape"

        # mask ignore_index if present
        if self.ignore_index is not None:
            mask = targets.clone().ne_(self.ignore_index)
            mask.requires_grad = False

            inputs = inputs * mask
            targets = targets * mask

        targets = flatten(targets)
        inputs = flatten(inputs)

        targets = targets.float()
        target_sum = targets.sum(-1)
        class_weights = Variable(1. / (target_sum * target_sum).clamp(min=self.epsilon), requires_grad=False)

        intersect = (inputs * targets).sum(-1) * class_weights
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
            intersect = weight * intersect

        denominator = (inputs + targets).sum(-1) * class_weights

        return torch.mean(1. - 2. * intersect / denominator.clamp(min=self.epsilon))
