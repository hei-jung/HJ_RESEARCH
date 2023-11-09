from torch import nn


class SegMetrics(nn.Module):
    def forward(self, inputs, targets, eps=0.0001):
        tp = (inputs * targets).sum(dim=4, keepdim=True).sum(dim=3, keepdim=True).sum(dim=2, keepdim=True)
        fp = (inputs * (1 - targets)).sum(dim=4, keepdim=True).sum(dim=3, keepdim=True).sum(dim=2, keepdim=True)
        fn = ((1 - inputs) * targets).sum(dim=4, keepdim=True).sum(dim=3, keepdim=True).sum(dim=2, keepdim=True)
        tn = ((1 - inputs) * (1 - targets)).sum(dim=4, keepdim=True).sum(dim=3, keepdim=True).sum(dim=2, keepdim=True)

        pixel_acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)
        dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        specificity = (tn + eps) / (tn + fp + eps)

        return pixel_acc.mean(), dice.mean(), precision.mean(), recall.mean(), specificity.mean()


class SegLoss(nn.Module):
    def forward(self, inputs, targets, eps=0.0001, loss='dice'):
        assert loss == 'pa' or loss == 'dice', "loss argument must be 'pa' or 'dice'."

        tp = (inputs * targets).sum(dim=4, keepdim=True).sum(dim=3, keepdim=True).sum(dim=2, keepdim=True)
        fp = (inputs * (1 - targets)).sum(dim=4, keepdim=True).sum(dim=3, keepdim=True).sum(dim=2, keepdim=True)
        fn = ((1 - inputs) * targets).sum(dim=4, keepdim=True).sum(dim=3, keepdim=True).sum(dim=2, keepdim=True)
        tn = ((1 - inputs) * (1 - targets)).sum(dim=4, keepdim=True).sum(dim=3, keepdim=True).sum(dim=2, keepdim=True)

        if loss == 'pa':
            pixel_acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)
            return (1.0 - pixel_acc).mean(), pixel_acc.mean()
        else:
            dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
            return (1.0 - dice).mean(), dice.mean()
