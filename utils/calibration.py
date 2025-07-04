import numpy as np

def load_kitti_calib(calib_file):
    data = {}
    with open(calib_file, 'r') as f:
        for line in f:
            if line.strip():
                key, value = line.strip().split(':', 1)
                data[key] = np.array([float(x) for x in value.strip().split()])

    P2 = data['P2'].reshape(3, 4)
    Tr_velo_to_cam = data['Tr_velo_to_cam'].reshape(3, 4)
    R0_rect = data['R0_rect'].reshape(3, 3)
    return P2, Tr_velo_to_cam, R0_rect

def project_lidar_to_image(points, P2, Tr, R0):
    N = points.shape[0]
    pts_hom = np.hstack((points, np.ones((N, 1))))
    pts_cam = (R0 @ Tr[:, :3]) @ pts_hom[:, :3].T + Tr[:, 3:4]
    pts_img = P2 @ np.vstack((pts_cam, np.ones((1, N))))
    pts_img = pts_img[:2, :] / pts_img[2, :]
    return pts_img.T
