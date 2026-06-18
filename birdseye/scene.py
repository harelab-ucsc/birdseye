import numpy as np
import open3d as o3d


class RTKTargets:
    def __init__(self, points_enu):
        """
        points_enu: Nx3 array in RTK world frame
        """
        self.points = np.asarray(points_enu)

    def get(self):
        return self.points


class SceneModel:
    def __init__(self, mesh_path=None, pcd_path=None):
        self.scene = o3d.t.geometry.RaycastingScene()

        self.has_mesh = False
        self.pcd = None

        if mesh_path is not None:
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            self.scene.add_triangles(mesh)
            self.has_mesh = True

        elif pcd_path is not None:
            # fallback: point cloud -> voxelized mesh approximation
            pcd = o3d.io.read_point_cloud(pcd_path)
            self.pcd = np.asarray(pcd.points)

    def raycast(self, origin, direction):
        """
        Returns:
            hit: bool
            t: float (distance)
        """

        rays = o3d.core.Tensor([np.hstack([origin, direction])],
                                dtype=o3d.core.Dtype.Float32)

        ans = self.scene.cast_rays(rays)

        t_hit = float(ans['t_hit'][0])
        hit = np.isfinite(t_hit)

        return hit, t_hit
