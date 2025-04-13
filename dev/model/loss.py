import torch.nn as nn
import torch.nn.functional as F


class CustomLoss(nn.Module):
    def __init__(self, delta):
        super().__init__()

        self.delta = delta

    def forward(self, pred, target):
        huber = F.huber_loss(pred, target, delta=self.delta)
        return huber
