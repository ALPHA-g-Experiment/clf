import torch
import torch.nn as nn


class _TNet(nn.Module):
    def __init__(self, in_dim, layers_1, layers_2, out_dim):
        super().__init__()

        self.layers_1 = nn.Sequential()
        for dim in layers_1:
            self.layers_1.append(nn.Conv1d(in_dim, dim, 1))
            self.layers_1.append(nn.BatchNorm1d(dim))
            self.layers_1.append(nn.ReLU())
            in_dim = dim

        self.layers_2 = nn.Sequential()
        for dim in layers_2:
            self.layers_2.append(nn.Linear(in_dim, dim))
            self.layers_2.append(nn.BatchNorm1d(dim))
            self.layers_2.append(nn.ReLU())
            in_dim = dim

        # Keep the last layer separate just to make it easier to e.g. initialize
        # it to produce the identity matrix.
        self.last = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.layers_1(x)
        x = x.max(dim=2).values
        x = self.layers_2(x)
        x = self.last(x)
        return x


class Regressor(nn.Module):
    def __init__(self, layers_1, layers_2, layers_3):
        super().__init__()
        in_dim = 3

        self.layers_1 = nn.Sequential()
        for dim in layers_1:
            self.layers_1.append(nn.Conv1d(in_dim, dim, 1))
            self.layers_1.append(nn.BatchNorm1d(dim))
            self.layers_1.append(nn.ReLU())
            in_dim = dim

        self.num_feats = in_dim
        self.feats_tnet = _TNet(in_dim, [64, 128, 1024], [512, 256], in_dim * in_dim)
        with torch.no_grad():
            self.feats_tnet.last.bias += torch.eye(in_dim).flatten()

        self.layers_2 = nn.Sequential()
        for dim in layers_2:
            self.layers_2.append(nn.Conv1d(in_dim, dim, 1))
            self.layers_2.append(nn.BatchNorm1d(dim))
            self.layers_2.append(nn.ReLU())
            in_dim = dim

        self.layers_3 = nn.Sequential()
        for dim in layers_3:
            self.layers_3.append(nn.Linear(in_dim, dim))
            self.layers_3.append(nn.BatchNorm1d(dim))
            self.layers_3.append(nn.ReLU())
            in_dim = dim

        self.layers_3.append(nn.Linear(in_dim, 1))

    def forward(self, x):
        x = self.layers_1(x)

        feat_trans = self.feats_tnet(x).view(-1, self.num_feats, self.num_feats)
        x = feat_trans.bmm(x)

        x = self.layers_2(x)
        x = x.max(dim=2).values
        x = self.layers_3(x)

        return x, feat_trans
