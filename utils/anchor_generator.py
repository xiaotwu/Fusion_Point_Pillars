import numpy as np
import torch
from config import BEV_WIDTH, BEV_HEIGHT, VOXEL_SIZE, CLASSES, ANCHOR_ROTATIONS

ANCHOR_SIZES = {
    'Car': [3.9, 1.6, 1.56],
    'Pedestrian': [0.8, 0.6, 1.73],
    'Cyclist': [1.76, 0.6, 1.73]
}
ANCHOR_Z = -1.0

def generate_anchors(H=100, W=88):
    anchors = []
    for i in range(H):
        for j in range(W):
            x = j * VOXEL_SIZE + VOXEL_SIZE / 2
            y = i * VOXEL_SIZE + VOXEL_SIZE / 2 - 40
            for cls in CLASSES:
                l, w, h = ANCHOR_SIZES[cls]
                z = ANCHOR_Z + h / 2
                for yaw in ANCHOR_ROTATIONS:
                    anchors.append([x, y, z, l, w, h, yaw])
    return torch.tensor(anchors, dtype=torch.float32)
