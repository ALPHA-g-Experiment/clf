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
        x = torch.max(x, dim=2).values
        x = self.layers_2(x)
        x = self.last(x)
        return x


class Regressor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
