import torch
import torch.nn as nn
from models.pointpillars import PillarFeatureNet
from models.image_encoder import ImageEncoder
from models.backbone import Backbone
from models.ssd_head import DetectionHead
from utils.lidar_utils import scatter_to_bev
from utils.projection import sample_image_features

class FusionPointPillars(nn.Module):
    def __init__(self, num_classes=3, num_anchors=6):
        super().__init__()
        self.pillar_net = PillarFeatureNet()
        self.image_net = ImageEncoder()
        self.fusion_fc = nn.Linear(128, 64)
        self.backbone = Backbone()
        self.ssd_head = DetectionHead(in_channels=128, num_classes=num_classes, num_anchors=num_anchors)

    def forward(self, lidar_input, image_input, proj_uv_coords, pillar_coords):
        lidar_feat = self.pillar_net(lidar_input)
        image_feat_map = self.image_net(image_input)
        image_feat = sample_image_features(image_feat_map, proj_uv_coords)
        fused_feat = torch.cat([lidar_feat, image_feat], dim=-1)
        fused_feat = self.fusion_fc(fused_feat)
        bev = scatter_to_bev(fused_feat, pillar_coords)
        feat = self.backbone(bev)
        cls, reg = self.ssd_head(feat)
        return cls, reg
