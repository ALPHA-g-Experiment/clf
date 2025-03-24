import torch.nn as nn
import torch.nn.functional as F
from reco.pointnet_utils import PointNetModEncoder


class get_model(nn.Module):
    def __init__(self, normal_channel=True, use_wireamp=False):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6 + use_wireamp
        else:
            channel = 3 + use_wireamp
        self.feat = PointNetModEncoder(
            global_feat=True, feature_transform=True, channel=channel
        )
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(p=0)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x, trans_feat


class get_loss(nn.Module):
    def __init__(self, delta=1.0, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale
        self.huber_loss = nn.HuberLoss(reduction="mean", delta=delta)

    def forward(self, pred, target, trans_feat):
        loss = self.huber_loss(pred, target)
        return loss
