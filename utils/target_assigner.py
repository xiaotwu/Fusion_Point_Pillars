import torch
from utils.box_utils import compute_bev_iou, encode_boxes

def assign_targets(anchors, gt_boxes, gt_labels, pos_iou_thresh=0.6, neg_iou_thresh=0.45, num_classes=3):
    N, M = anchors.shape[0], gt_boxes.shape[0]
    cls_targets = torch.zeros(N, dtype=torch.int64)
    reg_targets = torch.zeros((N, 7), dtype=torch.float32)
    reg_mask = torch.zeros(N, dtype=torch.bool)

    if M == 0:
        return cls_targets, reg_targets, reg_mask

    ious = compute_bev_iou(anchors, gt_boxes)
    max_iou, max_iou_idx = ious.max(dim=1)

    pos_mask = max_iou >= pos_iou_thresh
    neg_mask = max_iou < neg_iou_thresh
    assigned_gt = max_iou_idx[pos_mask]

    cls_targets[pos_mask] = gt_labels[assigned_gt] + 1
    reg_targets[pos_mask] = encode_boxes(gt_boxes[assigned_gt], anchors[pos_mask])
    reg_mask[pos_mask] = True

    return cls_targets, reg_targets, reg_mask
