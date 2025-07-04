import torch

def project_pillars_to_image(pillar_coords, calib_matrix):
    B, P, _ = pillar_coords.shape
    xyz = torch.cat([pillar_coords, torch.ones(B, P, 1).to(pillar_coords.device)], dim=-1)
    uvw = torch.matmul(xyz, calib_matrix.T)
    u = uvw[..., 0] / uvw[..., 2]
    v = uvw[..., 1] / uvw[..., 2]
    return torch.stack([u, v], dim=-1)

def sample_image_features(image_feat_map, proj_uv_coords):
    B, C, H, W = image_feat_map.shape
    B2, P, _ = proj_uv_coords.shape
    assert B == B2

    feat = torch.zeros((B, P, C), device=image_feat_map.device)
    for b in range(B):
        for p in range(P):
            u, v = proj_uv_coords[b, p].long()
            u = torch.clamp(u, 0, W - 1)
            v = torch.clamp(v, 0, H - 1)
            feat[b, p] = image_feat_map[b, :, v, u]
    return feat
