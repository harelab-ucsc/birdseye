import numpy as np


@dataclass
class Ray:
    origin: np.ndarray      # (3,)
    direction: np.ndarray   # (3,)


class PinholeCameraModel:
    def __init__(self, K):
        self.K = K

    def image_to_rays(self, pixels, T_WC):
        """
        pixels: (N,2)
        returns: list[Ray]
        """
        pixels = np.asarray(pixels)

        R = T_WC[:3, :3]
        origin = T_WC[:3, 3]

        rays = []

        Kinv = np.linalg.inv(self.K)

        for u, v in pixels:
            p = np.array([u, v, 1.0])
            d_cam = Kinv @ p
            d_world = R.T @ d_cam
            d_world /= np.linalg.norm(d_world)

            rays.append(Ray(origin=origin.copy(), direction=d_world))

        return rays

    # Forward projection (3D → 2D)
    def world_to_image(self, world_points, T_WC):
        world_points = np.asarray(world_points)

        if world_points.shape[0] == 0:
            return np.zeros((0, 2))

        if world_points.shape[1] == 3:
            world_points = np.hstack([
                world_points,
                np.ones((len(world_points), 1))
            ])

        T_CW = np.linalg.inv(T_WC)
        cam = T_CW @ world_points.T
        cam = cam[:3,:]

        proj = self.K @ cam
        proj /= proj[2:3]

        return proj[:2].T
