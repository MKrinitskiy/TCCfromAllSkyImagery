import torch
import torch.nn as nn
from torch.nn import init


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']


    def __init__(self, in_features, out_features, downsample=None, activation=None, **kwargs):
        super(BasicBlock, self).__init__()

        if activation is None:
            initial_slope = kwargs['initial_slope']
            self.fc1 = nn.Linear(in_features, out_features)
            init.kaiming_normal_(self.fc1.weight, a=initial_slope)
            self.bn1 = nn.BatchNorm1d(out_features)
            self.act1 = nn.PReLU(init=initial_slope)
            self.fc2 = nn.Linear(out_features, out_features)
            init.kaiming_normal_(self.fc2.weight, a=initial_slope)
            self.bn2 = nn.BatchNorm1d(out_features)
            self.act2 = nn.PReLU(init=initial_slope)
            self.downsample = downsample
        else:
            self.fc1 = nn.Linear(in_features, out_features)
            init.kaiming_normal_(self.fc1.weight, a=activation.negative_slope)
            self.bn1 = nn.BatchNorm1d(out_features)
            self.act1 = activation
            self.fc2 = nn.Linear(out_features, out_features)
            init.kaiming_normal_(self.fc2.weight, a=activation.negative_slope)
            self.bn2 = nn.BatchNorm1d(out_features)
            self.act2 = activation
            self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.fc2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act2(out)

        return out
