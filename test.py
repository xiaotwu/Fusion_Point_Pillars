import os
import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.fusion_pointpillars import FusionPointPillars
from utils.kitti_dataset import KITTIDataset
from utils.anchor_generator import generate_anchors
from config import *

# 类别映射（从1开始，0为背景）
CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist']

def decode_boxes(anchors, deltas):
    xa, ya, za, la, wa, ha, ra = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3], anchors[:, 4], anchors[:, 5], anchors[:, 6]
    dx, dy, dz, dlogl, dlogw, dlogh, dtheta = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3], deltas[:, 4], deltas[:, 5], deltas[:, 6]

    xg = dx * la + xa
    yg = dy * wa + ya
    zg = dz * ha + za
    lg = torch.exp(dlogl) * la
    wg = torch.exp(dlogw) * wa
    hg = torch.exp(dlogh) * ha
    rg = dtheta + ra

    return torch.stack([xg, yg, zg, lg, wg, hg, rg], dim=1)

def nms_bev(boxes, scores, iou_thresh=0.5, topk=100):
    from shapely.geometry import box as rect
    keep = []
    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()
    indices = scores.argsort()[::-1]
    selected = np.zeros(len(scores), dtype=bool)

    for i in indices:
        if selected[i]:
            continue
        keep.append(i)
        xi, yi, li, wi = boxes[i][0], boxes[i][1], boxes[i][3], boxes[i][4]
        box_i = rect(xi - li/2, yi - wi/2, xi + li/2, yi + wi/2)

        for j in indices:
            if i == j or selected[j]:
                continue
            xj, yj, lj, wj = boxes[j][0], boxes[j][1], boxes[j][3], boxes[j][4]
            box_j = rect(xj - lj/2, yj - wj/2, xj + lj/2, yj + wj/2)
            inter = box_i.intersection(box_j).area
            union = box_i.union(box_j).area
            if inter / union > iou_thresh:
                selected[j] = True

    return keep[:topk]

def test():
    output_dir = "devkit_object/results/data"
    os.makedirs(output_dir, exist_ok=True)

    model = FusionPointPillars().to(DEVICE)
    model.load_state_dict(torch.load('fusion_pointpillars.pth'))
    model.eval()

    dataset = KITTIDataset(root=KITTI_ROOT, split='val')
    dataloader = DataLoader(dataset, batch_size=1)

    anchors = generate_anchors(H=100, W=88).to(DEVICE)

    with torch.no_grad():
        for i, (lidar, image, coords, uv_coords, _, name) in enumerate(dataloader):
            lidar = lidar[0].to(DEVICE)
            image = image[0].to(DEVICE)
            coords = coords[0].to(DEVICE)
            uv_coords = uv_coords[0].to(DEVICE)

            cls_pred, reg_pred = model(
                lidar.unsqueeze(0),
                image.unsqueeze(0),
                uv_coords.unsqueeze(0),
                coords.unsqueeze(0)
            )

            cls_pred = F.softmax(cls_pred, dim=1)[0]  # (C, H, W)
            reg_pred = reg_pred[0].permute(1, 2, 0).reshape(-1, 7)
            cls_pred = cls_pred.permute(1, 2, 0).reshape(-1, cls_pred.shape[0])  # (N, C)

            boxes = decode_boxes(anchors, reg_pred)
            scores, labels = cls_pred[:, 1:].max(dim=1)  # drop background

            mask = scores > 0.4
            boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

            keep = nms_bev(boxes, scores)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            save_path = os.path.join(output_dir, f"{name[0]}.txt")
            with open(save_path, 'w') as f:
                for box, score, label in zip(boxes.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy()):
                    cls = CLASS_NAMES[label]
                    x, y, z, l, w, h, ry = box
                    f.write(f"{cls} 0 0 -1 0 0 0 0 {h:.2f} {w:.2f} {l:.2f} {x:.2f} {y:.2f} {z:.2f} {ry:.2f} {score:.2f}\n")

            print(f"[{i}] Saved: {save_path}")

if __name__ == "__main__":
    test()
