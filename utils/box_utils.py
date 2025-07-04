import torch
import numpy as np
from shapely.geometry import Polygon

def compute_bev_iou(anchor_boxes, gt_boxes):
    def box_to_poly(box):
        x, y, _, l, w, _, yaw = box
        dx, dy = l / 2, w / 2
        corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        rot = corners @ R.T + [x, y]
        return Polygon(rot)

    anchor_boxes = anchor_boxes.cpu().numpy()
    gt_boxes = gt_boxes.cpu().numpy()
    N, M = len(anchor_boxes), len(gt_boxes)
    ious = np.zeros((N, M), dtype=np.float32)

    for i in range(N):
        p1 = box_to_poly(anchor_boxes[i])
        for j in range(M):
            p2 = box_to_poly(gt_boxes[j])
            if p1.is_valid and p2.is_valid:
                inter = p1.intersection(p2).area
                union = p1.union(p2).area
                if union > 0:
                    ious[i, j] = inter / union
    return torch.tensor(ious)

def encode_boxes(gt_boxes, anchors):
    xa, ya, za, la, wa, ha, ra = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3], anchors[:, 4], anchors[:, 5], anchors[:, 6]
    xg, yg, zg, lg, wg, hg, rg = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5], gt_boxes[:, 6]
    dx = (xg - xa) / la
    dy = (yg - ya) / wa
    dz = (zg - za) / ha
    dlogl = torch.log(lg / la)
    dlogw = torch.log(wg / wa)
    dlogh = torch.log(hg / ha)
    dtheta = rg - ra
    return torch.stack((dx, dy, dz, dlogl, dlogw, dlogh, dtheta), dim=1)
