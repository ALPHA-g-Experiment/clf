import torch.nn as nn
import torch.nn.functional as F


class CustomLoss(nn.Module):
    """
    Custom loss function wrapper.

    Args:
        config (dict): Loss function configuration with keys:
            - `delta` (float): Threshold at which to switch between delta-scaled
              L1 and L2 loss.

    Inputs:
        pred (Tensor): Predicted values of shape (B,).
        target (Tensor): Target values of shape (B,).

    Returns:
        Tensor: Scalar loss value.
    """

    def __init__(self, config):
        super().__init__()

        self.delta = config["delta"]

    def forward(self, pred, target):
        huber = F.huber_loss(pred, target, delta=self.delta)
        return huber
