import torch

def scatter_to_bev(fused_features, coords, H=200, W=176, C=64):
    B, P, D = fused_features.shape
    bev = torch.zeros((B, C, H, W), device=fused_features.device)
    for b in range(B):
        for p in range(P):
            x, y = coords[b, p]
            x = torch.clamp(x, 0, W - 1)
            y = torch.clamp(y, 0, H - 1)
            bev[b, :, y, x] = fused_features[b, p]
    return bev
