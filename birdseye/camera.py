import numpy as np


class CameraModel:
    def __init__(self, K, T_WC):
        self.K = K
        self.T_WC = T_WC  # world -> camera

    def project(self, X_world):
        X = np.hstack([X_world, 1.0])

        Xc = np.linalg.inv(self.T_WC) @ X
        x = self.K @ Xc[:3]

        return x[:2] / x[2]

    def ray(self, pixel):
        """
        returns world-space ray (origin, direction)
        """
        px = np.array([pixel[0], pixel[1], 1.0])

        dir_cam = np.linalg.inv(self.K) @ px
        dir_cam = dir_cam / np.linalg.norm(dir_cam)

        origin_cam = np.array([0, 0, 0, 1])

        origin_world = self.T_WC @ origin_cam
        origin_world = origin_world[:3]

        dir_world = self.T_WC[:3, :3] @ dir_cam
        dir_world = dir_world / np.linalg.norm(dir_world)

        return origin_world, dir_world
