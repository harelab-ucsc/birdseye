import os
import sys
import laspy
import argparse
import hashlib
import glob2
import time

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import open3d as o3d
import pyvista as pv

from birdseye.camera.camera import CameraConfig, SensorConfigLoader, PinholeCameraModel
from birdseye.camera.geo_datasets import GeoTIFF, GeoPointCloud


def to_pyvista_mesh(o3d_mesh):
    """Convert Open3D mesh → PyVista mesh"""
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)

    # PyVista face format: [3, v0, v1, v2, 3, v0, v1, v2, ...]
    faces_pv = np.hstack(
        [np.full((faces.shape[0], 1), 3), faces]
    ).astype(np.int64).ravel()

    return pv.PolyData(vertices, faces_pv)


def add_rays(plotter, origins, dirs, hit_points=None, length=100.0):
    for i in range(len(origins)):
        o = origins[i]
        d = dirs[i]
        d = d / (np.linalg.norm(d) + 1e-12)
        end = o + d * length
        line = pv.Line(o, end)
        plotter.add_mesh(line, color="red", line_width=1)

        # origin
        plotter.add_mesh(pv.Sphere(radius=2.0, center=o), color="green")

        # hit
        if hit_points is not None and np.all(np.isfinite(hit_points[i])):
            plotter.add_mesh(
                pv.Sphere(radius=2.5, center=hit_points[i]),
                color="blue"
            )


class GeometryBackend(ABC):
    """
    Unified interface for all geometry models:
    flat-world, DSM, point cloud, mesh, etc.
    """
    # Scene query (core primitive)
    @abstractmethod
    def raycast(self, rays):
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
        origins, dirs = self.camera.image_to_rays(pixels, T_WC)
        return self.backend.raycast(origins, dirs)

    def world_to_image(self, points, T_WC):
        return self.camera.world_to_image(points, T_WC)

    def camera_frustum(self, image_shape, T_WC, stride='corners'):
        w, h = image_shape
        if stride == 'corners':
            pixels = np.array(
                [
                    [0.0, 0.0],
                    [0.0, h-1],
                    [w-1, 0.0],
                    [w-1, h-1]
                ]
            )
        else:
            u = np.arange(0, w, stride)
            v = np.arange(0, h, stride)
            uu, vv = np.meshgrid(u, v)
            pixels = np.stack([uu.ravel(), vv.ravel()], axis=1)
        origins, dirs = self.camera.image_to_rays(pixels, T_WC)
        return origins, dirs


class FlatWorldBackend(GeometryBackend):

    def __init__(self, K, ground_z=0.0):
        self.ground_z = ground_z

    # ray → plane intersection
    def raycast(self, rays):
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

    def raycast(self, origins, directions):
        """
        Vectorized raycast interface.

        Args:
            origins: (N,3)
            directions: (N,3)

        Returns:
            Open3D RaycastingScene.RaycastResult with the following keys:

        t_hit is the distance to the intersection. The unit is defined by the
            length of the ray direction. If no intersection this is inf
        geometry_ids gives the id of the geometry hit by the ray.
            If no geometry was hit this is RaycastingScene.INVALID_ID
        primitive_ids is the triangle index of the triangle that was hit
            or RaycastingScene.INVALID_ID
        primitive_uvs is the barycentric coordinates of the intersection
            point within the triangle.
        primitive_normals is the normal of the hit triangle.
        """
        origins = np.asarray(origins, dtype=np.float32)
        directions = np.asarray(directions, dtype=np.float32)
        rays = np.hstack((origins, directions)).astype(np.float32)
        tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
        return self.scene.cast_rays(tensor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo run for Projection Models in the BirdsEye ecosystem.")
    parser.add_argument("laz_filepath", help="file path to a target pointcloud")
    parser.add_argument("yaml_filepath", help="file path to a sensor configuration YAML")
    parser.add_argument("save_dir", help="file path to output directory (save target)")
    parser.add_argument("--downsample", action="store_true", default=False)

    args = parser.parse_args()

    filepath = args.laz_filepath
    ext = os.path.splitext(filepath)[-1].lower()
    if ext == ".tif":
        dataset = GeoTIFF(filepath)
    elif ext in [".las", ".laz"]:
        dataset = GeoPointCloud(filepath)
    else:
        print("Unsupported file type. Please provide a .tif, .las, or .laz file.")
        sys.exit(1)

    print("\n[PROC]    === Dataset Summary ===")
    summary = dataset.summary()
    for key in summary.keys():
        print(f'[PROC]    {key}: {summary[key]}')
    print('[PROC]')

    cached_mesh = False
    tmp = [str(summary[key]) for key in summary]
    tmp = "".join(tmp)
    hash_id = hashlib.sha256(tmp.encode()).hexdigest()
    filename = f'mesh_{hash_id}.ply'
    filename = os.path.join(args.save_dir, filename)
    files = glob2.glob(f'{args.save_dir}/*.ply')
    if filename in files:
        cached_mesh = True
        mesh = o3d.io.read_triangle_mesh(filename)
        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(tmesh)
        backend = MeshBackend(scene)
        print(f'[PROC]    Found cached mesh at: {filename}')
        print(
            f"[PROC]      Mesh: {len(mesh.vertices):,} vertices, "
            f"{len(mesh.triangles):,} triangles"
        )

    if not cached_mesh:
        print('[PROC]    Making new mesh from pointcloud...')
        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(dataset.points)

        if args.downsample:
            print('[PROC]    Downsampling...')
            # Optional: downsample if the cloud is very large
            voxel_size = 0.25
            pcd = pcd.voxel_down_sample(voxel_size)

        print(f"[PROC]        After downsampling: {len(pcd.points):,} points")

        # Estimate normals (required for Poisson reconstruction)
        print('[PROC]    Estimating pointcloud normals...')
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=5.0,
                max_nn=30,
            )
        )
        pcd.orient_normals_consistent_tangent_plane(50)

        # Generate mesh
        print("[PROC]    Running Poisson reconstruction...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=10,
        )

        # Remove low-density artifacts
        densities = np.asarray(densities)
        density_threshold = np.quantile(densities, 0.02)
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
        mesh.compute_vertex_normals()
        print(
            f"[PROC]        Mesh: {len(mesh.vertices):,} vertices, "
            f"{len(mesh.triangles):,} triangles"
        )

        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(tmesh)
        backend = MeshBackend(scene)
        o3d.io.write_triangle_mesh(filename, mesh)
        print('[PROC]    Mesh built and cached at: {filename}')

    loader = SensorConfigLoader(args.yaml_filepath)
    cam_cfg = loader.get_camera("rgb_1")
    cam = PinholeCameraModel.from_config(cam_cfg)
    engine = ProjectionEngine(cam, backend)

    # TODO: add mean subtraction of the dataset, so that numbers are smaller

    T_cam_ins = cam_cfg.T_cam_ins
    T_ins_world = np.eye(4)
    T_ins_world[:3,:3] = np.array([[0,1,0],[1,0,0],[0,0,-1]])  # NED INS to ENU viz
    T_ins_world[:3,3] = np.array([584150, 4093350, 115])
    T_cam_world = T_ins_world @ T_cam_ins
    start = time.time()
    image_shape = engine.camera.image_shape()
    # stride = 1
    stride='corners'
    origins, dirs = engine.camera_frustum(
        image_shape,
        T_cam_world,
        stride=stride,
    )
    if stride != 'corners':
        print(f'[DEBUG]    Rays cast: ',
            f'{(image_shape[0]//stride)*(image_shape[1]//stride)}',
            f'({image_shape[0]//stride} x {image_shape[1]//stride})')
    print(f'[DEBUG]    elapsed: {time.time() - start}')
    hit_result = engine.backend.raycast(origins, dirs)
    t_hit = hit_result["t_hit"].numpy()
    hit_points = np.where(
        np.isfinite(t_hit)[:, None],
        origins + t_hit[:, None] * dirs,
        np.nan
    )

    # Visualize
    pv_mesh = to_pyvista_mesh(mesh)
    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, color="lightgray", opacity=1.0)
    add_rays(plotter, origins, dirs, hit_points)
    plotter.set_background("white")
    plotter.show_axes()
    plotter.enable_eye_dome_lighting()

    plotter.show()
