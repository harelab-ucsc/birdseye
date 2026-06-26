import os
import sys
import laspy
import argparse
import hashlib
import glob2
import time
import utm
import csv

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import open3d as o3d
import pyvista as pv

from birdseye.camera.camera import CameraConfig, SensorConfigLoader, PinholeCameraModel
from birdseye.camera.geo_datasets import GeoTIFF, GeoPointCloud


STRIDE = 200


def csv_read(clicks_csv):
    print(f"[PROC]    Reading clicks CSV: {clicks_csv}...")
    data = []
    with open(clicks_csv) as clicks:
        reader = csv.reader(clicks)
        for line in reader:
            # returns easting, northing, zone number, zone letter
            u = utm.from_latlon(float(line[0]), float(line[1]))
            tag = int(line[-1][-1])
            data.append(  # Easting, Northing, Number, Letter, EPS, MSL, tag
                [u[0], u[1], u[2], u[3], float(line[2]), float(line[3]), tag]
            )
    data = np.array(data)
    data = data[:, [0,1,4]].astype(float)
    print("[PROC]      ... Done.")
    return data


def to_pyvista_mesh(o3d_mesh):
    """Convert Open3D mesh → PyVista mesh"""
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)

    # PyVista face format: [3, v0, v1, v2, 3, v0, v1, v2, ...]
    faces_pv = np.hstack(
        [np.full((faces.shape[0], 1), 3), faces]
    ).astype(np.int64).ravel()

    pv_mesh = pv.PolyData(vertices, faces_pv)

    if o3d_mesh.has_vertex_colors():
        colors = np.asarray(o3d_mesh.vertex_colors)
        # PyVista expects RGB colors as 8-bit integers (0-255)
        colors_8bit = (colors * 255).astype(np.uint8)

        # Attach colors directly to the vertices
        pv_mesh.point_data["Colors"] = colors_8bit

    return pv_mesh


def draw_camera(loader, engine, cam_name, color, T_ins_world, stride=STRIDE, verbose=False):
    cam_cfg = loader.get_camera(cam_name)
    cam = PinholeCameraModel.from_config(cam_cfg)
    engine.camera = cam

    T_cam_ins = cam_cfg.T_cam_ins
    T_cam_world = T_ins_world @ T_cam_ins
    start = time.time()
    image_shape = engine.camera.image_shape()
    stride = stride

    origins, dirs = engine.camera_frustum(
        image_shape,
        T_cam_world,
        stride=stride,
    )
    if stride != 'corners' and verbose:
        print(f'[PROC] [DEBUG]    Rays cast: ',
            f'{(image_shape[0]//stride)*(image_shape[1]//stride)}',
            f'({image_shape[0]//stride} x {image_shape[1]//stride})')
    print(f'[PROC] [DEBUG]    elapsed: {time.time() - start}')
    hit_result = engine.backend.raycast(origins, dirs)
    t_hit = hit_result["t_hit"].numpy()
    hit_points = np.where(
        np.isfinite(t_hit)[:, None],
        origins + t_hit[:, None] * dirs,
        np.nan
    )
    add_rays(plotter, origins, dirs, hit_points, color=color)


def add_rays(plotter, origins, dirs, hit_points=None, color=None):
    for i in range(len(origins)):
        o = origins[i]
        d = dirs[i]
        d = d / (np.linalg.norm(d) + 1e-12)
        # end = o + d * length
        line = pv.Line(o, hit_points[i])
        if color is None:
            plotter.add_mesh(line, color='red', line_width=1)
        else:
            plotter.add_mesh(line, color=color, line_width=1)

        # origin
        plotter.add_mesh(pv.Sphere(radius=0.25, center=o), color="green")

        # hit
        if hit_points is not None and np.all(np.isfinite(hit_points[i])):
            if color is None:
                plotter.add_mesh(
                    pv.Sphere(radius=1.5, center=hit_points[i]),
                    color='blue'
                )
            else:
                plotter.add_mesh(
                    pv.Sphere(radius=1.5, center=hit_points[i]),
                    color=color
                )


def add_spheres(world_points, radius=None, color=None):
    if color is None:
        color = "blue"
    if radius is None:
        radius = 0.5
    for point in world_points:
        plotter.add_mesh(pv.Sphere(radius=radius, center=point), color=color)


class GeometryBackend(ABC):
    """
    Unified interface for all geometry models:
    flat-world, DSM, point cloud, mesh, etc.
    """
    # Scene query (core primitive)
    @abstractmethod
    def raycast(self, origins, directions):
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

    def image_to_world(self, pixels, T_cam_world):
        origins, dirs = self.camera.image_to_rays(pixels, T_cam_world)
        return self.backend.raycast(origins, dirs)

    def world_to_image(self, points, T_cam_world):
        return self.camera.world_to_image(points, T_cam_world)

    def camera_frustum(self, image_shape, T_cam_world, stride='corners'):
        # TODO: change so that we don't assume rectangular frame
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
        origins, dirs = self.camera.image_to_rays(pixels, T_cam_world)
        return origins, dirs

    def pixels_in_view(self, pixels):
        return self.camera.valid_path.contains_points(pixels)

    def occlusion_mask(self, world_points, T_cam_world, epsilon):
        cam_origin = T_cam_world[:3, 3]
        origins = np.repeat(cam_origin[None, :], len(world_points), axis=0)
        vectors = world_points - origins
        target_dist = np.linalg.norm(vectors, axis=1)
        dirs = vectors / (target_dist[:, None] + 1e-12)
        hit_result = self.backend.raycast(origins, dirs)
        t_hit = hit_result["t_hit"].numpy()

        # visible if:
        #   first thing hit is the target itself
        not_occluded = (
            np.isfinite(t_hit)
            &
            np.abs(t_hit - target_dist) <= epsilon
        )
        return not_occluded

    def visible_world_points(
        self,
        world_points,
        T_cam_world,
        occlusion_check=True,
        epsilon=0.25,
    ):
        pixels, in_front = self.world_to_image(world_points, T_cam_world)
        in_frame = self.pixels_in_view(pixels)
        visible = in_front & in_frame
        if not occlusion_check:
            return pixels, visible

        not_occluded = self.occlusion_mask(world_points, T_cam_world, epsilon)
        visible &= not_occluded
        return pixels, visible


class FlatWorldBackend(GeometryBackend):

    def __init__(self, ground_z=0.0):
        self.ground_z = ground_z

    # ray → plane intersection
    def raycast(self, origins, directions):
        denom = directions[:,2]
        valid = np.abs(denom) > 1e-8
        t = np.full(len(origins), np.nan)
        t[valid] = (
            self.ground_z - origins[valid,2]
        ) / denom[valid]
        hits = np.full_like(origins, np.nan)
        hits[valid] = (
            origins[valid]
            + directions[valid] * t[valid,None]
        )
        return hits


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
    parser.add_argument("clicks_csv", help="file path to geotag data CSV")
    parser.add_argument("save_dir", help="file path to output directory (save target)")
    parser.add_argument("--downsample", action="store_true", default=False)

    args = parser.parse_args()
    clicks = csv_read(args.clicks_csv)

    filepath = args.laz_filepath
    ext = os.path.splitext(filepath)[-1].lower()
    if ext == ".tif":
        dataset = GeoTIFF(filepath)
    elif ext in [".las", ".laz"]:
        dataset = GeoPointCloud(filepath)
    else:
        print("[PROC] [ERROR]    Unsupported file type. Please provide a .tif, .las, or .laz file.")
        sys.exit(1)

    print("[PROC]\n[PROC]    === Dataset Summary ===")
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
        if dataset.colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(dataset.colors)

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
        
        if dataset.colors is not None:
            print("[PROC]    Coloring mesh from pointcloud.")
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)

            pcd_colors = np.asarray(pcd.colors)
            vertex_colors = np.empty((len(mesh.vertices), 3), dtype=np.float64)

            for i, v in enumerate(mesh.vertices):
                _, idx, _ = pcd_tree.search_knn_vector_3d(v, 1)
                vertex_colors[i] = pcd_colors[idx[0]]

            mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

        print('[PROC]    Building Open3D RaycastingScene')
        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(tmesh)
        backend = MeshBackend(scene)
        o3d.io.write_triangle_mesh(filename, mesh)
        print('[PROC]    Mesh built and cached at: {filename}')

    # TODO: add mean subtraction of the dataset, so that numbers are smaller

    T_ins_world = np.eye(4)
    T_ins_world[:3,:3] = np.array([[0,1,0],[1,0,0],[0,0,-1]])  # NED INS to ENU viz
    T_ins_world[:3,3] = np.array([584150, 4093350, 115])

    # Visualize
    pv_mesh = to_pyvista_mesh(mesh)
    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, opacity=1.0)

    loader = SensorConfigLoader(args.yaml_filepath)
    cam_cfg = loader.get_camera("rgb_1")
    cam = PinholeCameraModel.from_config(cam_cfg)
    engine = ProjectionEngine(cam, backend)

    draw_camera(loader, engine, "rgb_1", "red", T_ins_world)
    draw_camera(loader, engine, "rgb_2", "blue", T_ins_world)
    draw_camera(loader, engine, "rgb_3", "green", T_ins_world)
    draw_camera(loader, engine, "rgb_4", "yellow", T_ins_world)
    draw_camera(loader, engine, "multispec_1", "misty_rose", T_ins_world)
    draw_camera(loader, engine, "multispec_2", "lavender", T_ins_world)
    draw_camera(loader, engine, "multispec_3", "honeydew", T_ins_world)
    draw_camera(loader, engine, "multispec_4", "light_goldenrod", T_ins_world)

    add_spheres(clicks, color='magenta')

    plotter.set_background("white")
    plotter.show_axes()
    plotter.enable_eye_dome_lighting()

    plotter.show()
