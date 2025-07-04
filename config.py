import torch
import numpy as np

KITTI_ROOT = 'datasets/kitti'
MAX_POINTS_PER_PILLAR = 32
MAX_PILLARS = 12000
BEV_WIDTH = 176
BEV_HEIGHT = 200
VOXEL_SIZE = 0.16

CLASSES = ['Car', 'Pedestrian', 'Cyclist']
ANCHOR_ROTATIONS = [0, np.pi / 2]

NUM_CLASSES = len(CLASSES)
NUM_ROTATIONS = 2
ANCHORS_PER_LOC = NUM_CLASSES * NUM_ROTATIONS  # 6
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5
