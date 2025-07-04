import torch.nn as nn

class DetectionHead(nn.Module):
    def __init__(self, in_channels=128, num_classes=3, num_anchors=6):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.cls_head = nn.Conv2d(in_channels, (num_classes + 1) * num_anchors, 1)
        self.reg_head = nn.Conv2d(in_channels, 7 * num_anchors, 1)

    def forward(self, x):
        cls = self.cls_head(x)  # (B, (C+1)*A, H, W)
        reg = self.reg_head(x)  # (B, 7*A, H, W)
        return cls, reg
