import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from utils.calibration import load_kitti_calib, project_lidar_to_image
from config import *

class KITTIDataset(Dataset):
    def __init__(self, root, split='train'):
        self.root = root
        self.split = split
        self.velo_dir = os.path.join(root, 'velodyne')
        self.image_dir = os.path.join(root, 'image_2')
        self.label_dir = os.path.join(root, 'label_2')
        self.calib_dir = os.path.join(root, 'calib')

        self.filenames = sorted([f[:-4] for f in os.listdir(self.velo_dir) if f.endswith('.bin')])
        self.transform = transforms.Compose([
            transforms.Resize((256, 832)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        lidar = self.load_lidar(os.path.join(self.velo_dir, name + '.bin'))
        image = self.load_image(os.path.join(self.image_dir, name + '.png'))
        calib_file = os.path.join(self.calib_dir, name + '.txt')
        P2, Tr, R0 = load_kitti_calib(calib_file)

        # Get projection for each pillar center
        pillar_coords, lidar_tensor = self.make_pillars(lidar)
        proj_uv_coords = self.make_proj_coords(pillar_coords, P2, Tr, R0)
        label_path = os.path.join(self.label_dir, name + '.txt')
        targets = self.load_labels(label_path)

        return lidar_tensor, image, pillar_coords, proj_uv_coords, targets, name

    def load_lidar(self, path):
        scan = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        return scan

    def load_image(self, path):
        img = Image.open(path).convert('RGB')
        return self.transform(img)

    def make_pillars(self, points):
        mask = (points[:, 0] >= 0) & (points[:, 0] < BEV_WIDTH) & (points[:, 1] >= -40) & (points[:, 1] < 40)
        points = points[mask]
        pillars = {}
        for pt in points:
            x, y = int(pt[0] / 0.16), int((pt[1] + 40) / 0.16)
            key = (x, y)
            if key not in pillars:
                pillars[key] = []
            if len(pillars[key]) < MAX_POINTS_PER_PILLAR:
                pillars[key].append(pt)

        pillar_coords = []
        pillar_data = []
        for i, (k, pts) in enumerate(pillars.items()):
            if i >= MAX_PILLARS:
                break
            num_points = len(pts)
            pts = np.array(pts)
            mean = np.mean(pts, axis=0)
            feature = np.zeros((MAX_POINTS_PER_PILLAR, 9), dtype=np.float32)
            feature[:num_points, :4] = pts
            feature[:num_points, 4:7] = pts[:, :3] - mean[:3]
            feature[:num_points, 7:9] = [k[0], k[1]]
            pillar_data.append(feature)
            pillar_coords.append([k[0], k[1]])

        pillar_tensor = torch.tensor(np.array(pillar_data), dtype=torch.float32)
  # (P, N, 9)
        pillar_coords = torch.tensor(pillar_coords)  # (P, 2)

        return pillar_coords, pillar_tensor

    def make_proj_coords(self, pillar_coords, P2, Tr, R0):
        centers = []
        for xy in pillar_coords:
            x = xy[0].item() * 0.16
            y = xy[1].item() * 0.16 - 40
            z = -1.0
            centers.append([x, y, z])
        centers = np.array(centers)
        uv = project_lidar_to_image(centers, P2, Tr, R0)
        return torch.tensor(uv, dtype=torch.float32)

    def load_labels(self, label_file):
        with open(label_file, 'r') as f:
            lines = f.readlines()

        boxes = []
        classes = []
        for l in lines:
            obj = l.strip().split()
            cls = obj[0]
            if cls not in ['Car', 'Pedestrian', 'Cyclist']:
                continue
            if len(obj) < 18:
                continue
            cls_idx = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}[cls]
            box3d = [float(x) for x in obj[11:18]]  # x, y, z, h, w, l, ry
            boxes.append(box3d)
            classes.append(cls_idx)

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 7)),
            'labels': torch.tensor(classes, dtype=torch.long) if classes else torch.zeros((0,), dtype=torch.long)
        }
        return target
