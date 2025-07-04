import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.fusion_pointpillars import FusionPointPillars
from utils.kitti_dataset import KITTIDataset
from utils.anchor_generator import generate_anchors
from utils.target_assigner import assign_targets
from config import *
from tqdm import tqdm

def custom_collate_fn(batch):
    return tuple(zip(*batch))

def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    ce = F.cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-ce)
    loss = alpha * (1 - pt) ** gamma * ce
    return loss.mean()

def train():
    os.makedirs("results/logs", exist_ok=True)

    dataset = KITTIDataset(root=KITTI_ROOT, split='train')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)

    model = FusionPointPillars().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_reg = torch.nn.SmoothL1Loss(reduction='none')

    anchors = generate_anchors(H=100, W=88).to(DEVICE)
    best_loss = float('inf')
    best_state = None

    model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch in progress_bar:
            lidar_batch, image_batch, coords_batch, uv_batch, targets_batch, _ = batch

            lidar = lidar_batch[0].to(DEVICE)
            image = image_batch[0].to(DEVICE)
            coords = coords_batch[0].to(DEVICE)
            uv_coords = uv_batch[0].to(DEVICE)
            targets = targets_batch[0]
            gt_boxes = targets['boxes'].to(DEVICE)
            gt_labels = targets['labels'].to(DEVICE)

            cls_pred, reg_pred = model(
                lidar.unsqueeze(0),
                image.unsqueeze(0),
                uv_coords.unsqueeze(0),
                coords.unsqueeze(0)
            )

            B, cls_channels, H, W = cls_pred.shape
            cls_pred = cls_pred.view(1, (NUM_CLASSES + 1), ANCHORS_PER_LOC, H, W)
            cls_pred = cls_pred.permute(0, 3, 4, 2, 1).reshape(-1, NUM_CLASSES + 1)

            reg_pred = reg_pred.view(1, 7, ANCHORS_PER_LOC, H, W)
            reg_pred = reg_pred.permute(0, 3, 4, 2, 1).reshape(-1, 7)

            cls_targets, reg_targets, reg_mask = assign_targets(anchors, gt_boxes, gt_labels)

            cls_loss = focal_loss(cls_pred, cls_targets.to(DEVICE))
            reg_loss_all = criterion_reg(reg_pred, reg_targets.to(DEVICE))
            reg_loss = reg_loss_all[reg_mask.to(DEVICE)].mean() if reg_mask.any() else torch.tensor(0.0, device=DEVICE)

            loss = cls_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "cls": f"{cls_loss.item():.4f}",
                "reg": f"{reg_loss.item():.4f}"
            })

    log_line = f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}"
    print(log_line)
    with open("results/logs/training_log.txt", "a") as f:
        f.write(log_line + "\n")

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_state = model.state_dict()


        log_line = f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item():.4f}, Cls: {cls_loss.item():.4f}, Reg: {reg_loss.item():.4f}"
        print(log_line)
        with open("results/logs/training_log.txt", "a") as f:
            f.write(log_line + "\n")

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = model.state_dict()

    if best_state is not None:
        torch.save(best_state, "fusion_pointpillars.pth")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    train()
