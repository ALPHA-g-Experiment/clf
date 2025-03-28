import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = (
            Variable(
                torch.from_numpy(
                    np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)
                )
            )
            .view(1, 9)
            .repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = (
            Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)))
            .view(1, self.k * self.k)
            .repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetModEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetModEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.k = 64

        self.conv1_k = torch.nn.Conv1d(channel, self.k, 1)
        self.conv2_k = torch.nn.Conv1d(self.k, 128, 1)
        self.conv3_k = torch.nn.Conv1d(128, 1024, 1)

        self.bn1_k = nn.BatchNorm1d(self.k)
        self.bn2_k = nn.BatchNorm1d(128)
        self.bn3_k = nn.BatchNorm1d(1024)

        self.global_feat = global_feat

        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)
            self.fstn0 = STNkd(k=self.k)

    def forward(self, x):
        # Extract dimensions of the input tensor
        B, D, N = x.size()

        # Apply spatial transformation to the input
        trans = self.stn(x)

        # Transpose dimensions for subsequent operations
        x = x.transpose(2, 1)

        # If input has more than 3 channels, separate the first 3 channels (coordinates) and the rest as 'feature'
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]

        # If input has more than 3 channels, concatenate the coordinates with the 'feature'
        if D > 3:
            x = torch.cat([x, feature], dim=2)

        # Transpose dimensions again for subsequent operations
        x = x.transpose(2, 1)

        # Apply 1D convolution, batch normalization, and ReLU activation
        x = F.relu(self.bn1_k(self.conv1_k(x)))

        # If feature transformation is enabled
        if self.feature_transform:
            # Apply feature transformation
            trans_feat = self.fstn0(x)
            x = x.transpose(2, 1)
            # Apply the learned transformation to the features
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            # If feature transformation is not enabled, set trans_feat to None
            trans_feat = None

        # Save the features before the second convolutional block
        pointfeat = x

        # Apply another convolution, batch normalization, and ReLU activation
        x = F.relu(self.bn2_k(self.conv2_k(x)))

        # Apply the third convolution and batch normalization
        x = self.bn3_k(self.conv3_k(x))

        # Perform max pooling along the third dimension to obtain a global feature
        x = torch.max(x, 2, keepdim=True)[0]

        # Reshape the tensor
        x = x.view(-1, 1024)

        # If global features are requested, return global feature, spatial transformation, and feature transformation
        if self.global_feat:
            return x, trans, trans_feat
        else:
            # If not, concatenate global feature with the saved point features and return along with spatial transformation and feature transformation
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat
