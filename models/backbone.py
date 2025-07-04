import torch.nn as nn

class Backbone(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)
