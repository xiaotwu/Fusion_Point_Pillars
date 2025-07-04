import torch.nn as nn
import torch

class PillarFeatureNet(nn.Module):
    def __init__(self, num_input=9, num_filters=64):
        super().__init__()
        self.fc = nn.Linear(num_input, num_filters)
        self.bn = nn.BatchNorm1d(num_filters)
        self.relu = nn.ReLU()

    def forward(self, x):
        B, P, N, D = x.shape
        x = x.view(-1, D)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.view(B, P, N, -1)
        return torch.max(x, dim=2)[0]
