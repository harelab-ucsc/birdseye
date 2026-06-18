from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import open3d as o3d

from camera import Ray, PinholeCameraModel


class GeometryBackend(ABC):
    """
    Unified interface for all geometry models:
    flat-world, DSM, point cloud, mesh, etc.
    """
    # Scene query (core primitive)
    @abstractmethod
    def rays_to_world(self, rays):
        """
        Intersect rays with world geometry.

        returns:
            (N,3) world points or None for misses
        """
        pass


class ProjectionEngine:
    def __init__(self, camera, backend):
        self.camera = camera
        self.backend = backend

    def image_to_world(self, pixels, T_WC):
        rays = self.camera.image_to_rays(pixels, T_WC)
        return self.backend.rays_to_world(rays)

    def world_to_image(self, points, T_WC):
        return self.camera.world_to_image(points, T_WC)

    def is_visible(self, points, T_WC):
        pix = self.world_to_image(points, T_WC)
        return len(pix) > 0


class FlatWorldBackend(GeometryBackend):

    def __init__(self, K, ground_z=0.0):
        self.ground_z = ground_z

    # ray → plane intersection
    def rays_to_world(self, rays):
        n = np.array([0, 0, 1.0])
        z0 = self.ground_z

        hits = []

        for r in rays:
            denom = r.direction @ n
            if abs(denom) < 1e-8:
                continue

            t = (z0 - r.origin @ n) / denom
            hits.append(r.origin + t * r.direction)

        return np.asarray(hits)


class MeshBackend(GeometryBackend):

    def __init__(self, scene):
        self.scene = scene

    def rays_to_world(self, rays):
        tmp = []
        for ray in rays:
            tmp.append([np.hstack((ray.origin, ray.direction)).tolist()])
        return self.scene.cast_rays(o3d.Tensor(tmp))
