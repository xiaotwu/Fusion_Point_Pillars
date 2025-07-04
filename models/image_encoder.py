import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ImageEncoder(nn.Module):
    def __init__(self, output_dim=64):
        super().__init__()
        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.feature_map = nn.Sequential(*list(base_model.children())[:-2])
        self.reduce = nn.Conv2d(512, output_dim, kernel_size=1)

    def forward(self, x):
        x = self.feature_map(x)
        x = self.reduce(x)
        return x
