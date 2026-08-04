import os
import sys
import laspy
import argparse
import hashlib
import glob2
import time
import utm
import csv
import cv2

from abc import ABC, abstractmethod
from dataclasses import dataclass

import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="matplotlib.projections",
)

import numpy as np
import open3d as o3d
import pyvista as pv
# import matplotlib.pyplot as plt

from open3d.visualization import rendering
from dataclasses import dataclass

from birdseye.camera.camera import CameraConfig, SensorConfigLoader, PinholeCameraModel
from birdseye.camera.geo_datasets import GeoTIFF, GeoPointCloud
from birdseye.geometry.se3 import SE3


# STRIDE = 200
STRIDE = 'corners'


@dataclass
class SyntheticView:
    contour_pixels: np.ndarray      # (N,2)
    contour_world: np.ndarray       # (N,3)
    contour_normals: np.ndarray     # (N,2)
    edge_map: np.ndarray
    distance: np.ndarray
    gx: np.ndarray
    gy: np.ndarray
    normals: np.ndarray
    depth: np.ndarray


@dataclass
class RaycastResult:
    origins: np.ndarray      # (N,3)
    directions: np.ndarray   # (N,3)
    t: np.ndarray            # (N,)
    points: np.ndarray       # (N,3)
    valid: np.ndarray        # (N,)


def csv_read(clicks_csv):
    print(f"[PROC]    Reading clicks CSV: {clicks_csv}...")
    data = []
    with open(clicks_csv) as clicks:
        reader = csv.reader(clicks)
        for line in reader:
            # returns easting, northing, zone number, zone letter
            u = utm.from_latlon(float(line[0]), float(line[1]))
            tag = line[-1][-1]
            data.append(  # Easting, Northing, Number, Letter, EPS, MSL, tag
                [u[0], u[1], u[2], u[3], float(line[2]), float(line[3]), tag]
            )
    data = np.array(data)
    data = data[:, [0, 1, 4]].astype(float)
    print("[PROC]      ... Done.")
    return data


def to_pyvista_mesh(o3d_mesh):
    """Convert Open3D mesh → PyVista mesh"""
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)

    # PyVista face format: [3, v0, v1, v2, 3, v0, v1, v2, ...]
    faces_pv = (
        np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64).ravel()
    )

    pv_mesh = pv.PolyData(vertices, faces_pv)

    if o3d_mesh.has_vertex_colors():
        colors = np.asarray(o3d_mesh.vertex_colors)
        # PyVista expects RGB colors as 8-bit integers (0-255)
        colors_8bit = (colors * 255).astype(np.uint8)

        # Attach colors directly to the vertices
        pv_mesh.point_data["Colors"] = colors_8bit

    return pv_mesh


def draw_camera(
    loader, proj, cam_name, color, T_ins_world, stride=STRIDE, verbose=False
):
    cam_cfg = loader.get_camera(cam_name)
    cam = PinholeCameraModel.from_config(cam_cfg)
    proj.camera = cam
    image_shape = proj.camera.image_shape()
    stride = stride

    T_cam_ins = cam_cfg.T_cam_ins
    T_cam_world = T_ins_world @ T_cam_ins

    origins, dirs = proj.camera_frustum(
        image_shape,
        T_cam_world,
        stride=stride,
    )

    hit_result = proj.backend.raycast(origins, dirs)
    t_hit = hit_result["t_hit"].numpy()
    hit_points = np.where(
        np.isfinite(t_hit)[:, None], origins + t_hit[:, None] * dirs, np.nan
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
            plotter.add_mesh(line, color="red", line_width=1)
        else:
            plotter.add_mesh(line, color=color, line_width=1)

        # origin
        plotter.add_mesh(pv.Sphere(radius=0.25, center=o), color="green")

        # hit
        if hit_points is not None and np.all(np.isfinite(hit_points[i])):
            if color is None:
                plotter.add_mesh(
                    pv.Sphere(radius=0.5, center=hit_points[i]), color="blue"
                )
            else:
                plotter.add_mesh(
                    pv.Sphere(radius=0.5, center=hit_points[i]), color=color
                )


def add_spheres(world_points, radius=None, color=None):
    if color is None:
        color = "blue"
    if radius is None:
        radius = 0.5
    for point in world_points:
        plotter.add_mesh(pv.Sphere(radius=radius, center=point), color=color)


def normal_gradient_image(
    normals,
    valid_mask=None,
    eps = 1e-6,
):
    H, W, _ = normals.shape

    dnx = np.zeros((H, W), np.float32)
    dny = np.zeros((H, W), np.float32)

    # central differences of the normals
    dx = normals[:, 2:, :] - normals[:, :-2, :]
    dy = normals[2:, :, :] - normals[:-2, :, :]

    # magnitude of each derivative
    dnx[:, 1:-1] = np.linalg.norm(dx, axis=2)
    dny[1:-1, :] = np.linalg.norm(dy, axis=2)

    if valid_mask is not None:
        valid = (
            valid_mask[1:-1,1:-1] &
            valid_mask[1:-1,:-2] &
            valid_mask[1:-1,2:] &
            valid_mask[:-2,1:-1] &
            valid_mask[2:,1:-1]
        )

        dnx[1:-1,1:-1] *= valid
        dny[1:-1,1:-1] *= valid

    mag = np.sqrt(dnx**2 + dny**2)
    gx_syn = dnx / (mag + eps)
    gy_syn = dny / (mag + eps)

    return mag, gx_syn, gy_syn


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
        origins, directions = self.camera.image_to_rays(
            pixels,
            T_cam_world,
        )

        ans = self.backend.raycast(origins, directions)

        t = ans["t_hit"].numpy()
        valid = np.isfinite(t)

        points = origins + directions * t[:, None]

        return RaycastResult(
            origins=origins,
            directions=directions,
            t=t,
            points=points,
            valid=valid,
        )

    def world_to_image(self, points, T_cam_world):
        return self.camera.world_to_image(points, T_cam_world)

    def camera_frustum(self, image_shape, T_cam_world, stride="corners"):
        # TODO: change so that we don't assume rectangular frame
        w, h = image_shape
        if stride == "corners":
            pixels = np.array([[0.0, 0.0], [0.0, h - 1], [w - 1, 0.0], [w - 1, h - 1]])
        else:
            u = np.arange(0, w, stride)
            v = np.arange(0, h, stride)
            uu, vv = np.meshgrid(u, v)
            pixels = np.stack([uu.ravel(), vv.ravel()], axis=1)
        origins, dirs = self.camera.image_to_rays(pixels, T_cam_world)
        return origins, dirs

    def pixels_in_view(self, pixels):
        u = pixels[:, 0]
        v = pixels[:, 1]

        return (
            (u >= 0)
            & (u < self.camera.width)
            & (v >= 0)
            & (v < self.camera.height)
        )

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
        finite = np.isfinite(t_hit)
        close = np.abs(t_hit - target_dist) <= epsilon
        not_occluded = finite & close
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
        candidate_mask = in_front & in_frame
        if not occlusion_check:
            return pixels, candidate_mask
        if not np.any(candidate_mask):
            return pixels, candidate_mask
        candidate_points = world_points[candidate_mask]
        not_occluded = self.occlusion_mask(
            candidate_points,
            T_cam_world,
            epsilon,
        )
        visible = np.zeros(len(world_points), dtype=bool)
        visible[candidate_mask] = not_occluded
        return pixels, visible

    def render(
        self,
        T_cam_world,
        grad_thresh=25.5,
        min_contour_length=40,
    ):
        """
        Render a synthetic observation suitable for contour registration.

        Returns
        -------
        SyntheticView
            contour_pixels : (N,2) float32 (u,v)
            contour_world  : (N,3)
            contour_normals: (N,2)
            edge_map
            distance
            gx
            gy
            normals
            depth
        """

        # Render mesh
        rays = self.backend.scene.create_rays_pinhole(
            self.camera.K,
            np.linalg.inv(T_cam_world),
            self.camera.width,
            self.camera.height,
        )

        ans = self.backend.scene.cast_rays(rays)
        depth = ans["t_hit"].numpy()
        normals = ans["primitive_normals"].numpy()
        valid = np.isfinite(depth)
        grad, gx_syn, gy_syn = normal_gradient_image(
            normals,
            valid,
        )
        grad = cv2.normalize(
            grad,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        ).astype(np.uint8)

        _, edge = cv2.threshold(
            grad,
            grad_thresh,
            255,
            cv2.THRESH_BINARY,
        )

        # Remove tiny contours
        clean = np.zeros_like(edge)
        contours, _ = cv2.findContours(
            edge,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_NONE,
        )
        for c in contours:
            if cv2.arcLength(c, False) > min_contour_length:
                cv2.drawContours(clean, [c], -1, 255, 1)
        edge = cv2.ximgproc.thinning(clean)

        # Distance transform of synthetic contour
        distance = cv2.distanceTransform(
            255 - edge,
            cv2.DIST_L2,
            5,
        )

        # Extract every contour pixel
        vv, uu = np.nonzero(edge)
        contour_pixels = np.stack(
            (uu, vv),
            axis=1,
        ).astype(np.float32)

        # Recover corresponding 3D points directly from rendered depth
        d = depth[vv, uu]
        finite = np.isfinite(d)
        contour_pixels = contour_pixels[finite]
        uu = uu[finite]
        vv = vv[finite]
        d = d[finite]

        # Get contour 3D points
        contour_rc = self.image_to_world(contour_pixels, T_cam_world)
        contour_world = contour_rc.points
        # contour_world /= np.linalg.norm(contour_world, axis=1, keepdims=True)

        # Edge normal directions
        contour_normals = np.stack((gx_syn[vv, uu], gy_syn[vv, uu]), axis=1)

        return SyntheticView(
            contour_pixels=contour_pixels,
            contour_world=contour_rc.points,
            contour_normals=contour_normals,
            edge_map=edge,
            distance=distance,
            gx=gx_syn,
            gy=gy_syn,
            normals=((normals + 1.0) * 127.5).astype(np.uint8),
            depth=depth,
        )

    def analytic_projection_jacobian(self, T, Pw):
        N = len(Pw)
        R = T[:3,:3]
        t = T[:3,3]

        xc = (R @ Pw.T).T + t
        x = xc[:,0]
        y = xc[:,1]
        z = xc[:,2]

        J_xi = np.zeros((N,3,6))
        J_xi[:,:3,:3] = np.broadcast_to(np.eye(3),(N,3,3))
        J_xi[:,:,3:] = -SE3.skew(xc)

        J_pixel = np.zeros((N,2,3))
        J_pixel[:,0,0] = self.camera.K[0,0]/z
        J_pixel[:,0,2] = -self.camera.K[0,0]*x/z**2
        J_pixel[:,1,1] = self.camera.K[1,1]/z
        J_pixel[:,1,2] = -self.camera.K[1,1]*y/z**2

        return np.einsum("nij,njk->nik", J_pixel, J_xi)

    def numerical_projection_jacobian(self, T, Pw, eps=1e-6):
        J = np.zeros((2,6))
        u0 = self.camera.project(T, Pw)

        for i in range(6):
            d = eps*np.ones(6)
            T_pert = SE3.perturb(T, d)
            u1 = self.camera.world_to_image(Pw, T_pert)
            J[:,i] = (u1 - u0) / eps

        return J


class FlatWorldBackend(GeometryBackend):
    def __init__(self, ground_z=0.0):
        self.ground_z = ground_z

    # ray → plane intersection
    def raycast(self, origins, directions):
        denom = directions[:, 2]
        valid = np.abs(denom) > 1e-8
        t = np.full(len(origins), np.nan)
        t[valid] = (self.ground_z - origins[valid, 2]) / denom[valid]
        hits = np.full_like(origins, np.nan)
        hits[valid] = origins[valid] + directions[valid] * t[valid, None]
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

    def snap_points_along_direction(
        self,
        points,
        direction=np.array([0.0, 0.0, 1.0]),
        max_distance=np.inf,
    ):
        """
        Project points onto the mesh along a specified direction.

        Rays are cast in both +direction and -direction, and the nearest
        intersection is chosen.

        Parameters
        ----------
        points : (N,3) ndarray
            World coordinates.
        direction : (3,) array_like
            Projection direction (need not be unit length).
            Examples:
                ENU vertical : [0,0,1]
                NED vertical : [0,0,-1]
                East-West    : [1,0,0]
        max_distance : float
            Ignore intersections farther than this distance.

        Returns
        -------
        snapped : (N,3) ndarray
            Snapped coordinates. Rows are NaN if no intersection exists.
        """
        points = np.asarray(points, dtype=np.float32)
        direction = np.asarray(direction, dtype=np.float32)

        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            raise ValueError("direction must have nonzero length")

        direction /= norm

        n = len(points)

        rays_pos = np.hstack((
            points,
            np.tile(direction, (n, 1))
        )).astype(np.float32)

        rays_neg = np.hstack((
            points,
            np.tile(-direction, (n, 1))
        )).astype(np.float32)

        hit_pos = self.scene.cast_rays(
            o3d.core.Tensor(rays_pos, dtype=o3d.core.Dtype.Float32)
        )

        hit_neg = self.scene.cast_rays(
            o3d.core.Tensor(rays_neg, dtype=o3d.core.Dtype.Float32)
        )

        t_pos = hit_pos["t_hit"].numpy()
        t_neg = hit_neg["t_hit"].numpy()

        snapped = np.full_like(points, np.nan)

        valid_pos = np.isfinite(t_pos) & (t_pos <= max_distance)
        valid_neg = np.isfinite(t_neg) & (t_neg <= max_distance)

        choose_pos = valid_pos & (~valid_neg | (t_pos <= t_neg))
        choose_neg = valid_neg & (~valid_pos | (t_neg < t_pos))

        snapped = np.full_like(points, np.nan)

        snapped[choose_pos] = (
            points[choose_pos]
            + t_pos[choose_pos, None] * direction
        )

        snapped[choose_neg] = (
            points[choose_neg]
            - t_neg[choose_neg, None] * direction
        )

        # remove lingering NaNs (arise when the click is outside mesh extent)
        snapped = snapped[~np.isnan(snapped).any(axis=1)]

        return snapped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo run for Projection Models in the BirdsEye ecosystem."
    )
    parser.add_argument("laz_filepath", help="file path to a target pointcloud")
    parser.add_argument(
        "yaml_filepath", help="file path to a sensor configuration YAML"
    )
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
        print(
            "[PROC] [ERROR]    Unsupported file type. Please provide a .tif, .las, or .laz file."
        )
        sys.exit(1)

    print("[PROC]\n[PROC]    === Dataset Summary ===")
    summary = dataset.summary()
    for key in summary.keys():
        print(f"[PROC]    {key}: {summary[key]}")
    print("[PROC]")

    cached_mesh = False
    tmp = [str(summary[key]) for key in summary]
    tmp = "".join(tmp)
    hash_id = hashlib.sha256(tmp.encode()).hexdigest()
    filename = f"mesh_{hash_id}.ply"
    filename = os.path.join(args.save_dir, filename)
    files = glob2.glob(f"{args.save_dir}/*.ply")
    if filename in files:
        cached_mesh = True
        mesh = o3d.io.read_triangle_mesh(filename)
        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(tmesh)
        backend = MeshBackend(scene)
        print(f"[PROC]    Found cached mesh at: {filename}")
        print(
            f"[PROC]      Mesh: {len(mesh.vertices):,} vertices, "
            f"{len(mesh.triangles):,} triangles"
        )

    if not cached_mesh:
        print("[PROC]    Making new mesh from pointcloud...")
        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        print('[PROC]        Loading points...')
        pcd.points = o3d.utility.Vector3dVector(dataset.points)
        if dataset.colors is not None:
            print('[PROC]        Loading colors...')
            pcd.colors = o3d.utility.Vector3dVector(dataset.colors)

        if args.downsample:
            print("[PROC]    Downsampling...")
            # Optional: downsample if the cloud is very large
            voxel_size = 0.15
            pcd = pcd.voxel_down_sample(voxel_size)

        print(f"[PROC]        After downsampling: {len(pcd.points):,} points")

        # Estimate normals (required for Poisson reconstruction)
        print("[PROC]    Estimating pointcloud normals...")
        origin = np.mean(np.asarray(pcd.points), axis=0)
        pcd.translate(-origin)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=1.0,
                max_nn=30,
            )
        )
        pcd.orient_normals_consistent_tangent_plane(30)

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
        mesh.translate(origin)

        print("[PROC]    Building Open3D RaycastingScene")
        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        tmesh.fill_holes()
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(tmesh)
        backend = MeshBackend(scene)
        o3d.io.write_triangle_mesh(filename, mesh)
        print("[PROC]    Mesh built and cached at: {filename}")

    verts = np.asarray(mesh.vertices)

    T_ins_world = np.eye(4)
    T_ins_world[:3, :3] = np.array(
        [[0, 1, 0], [1, 0, 0], [0, 0, -1]]
    )  # NED INS to ENU viz
    x = (verts[:,0].max() - verts[:,0].min()) / 2 + verts[:,0].min()
    y = (verts[:,1].max() - verts[:,1].min()) / 2 + verts[:,1].min()
    z = verts[:,2].max() + 10
    T_ins_world[:3, 3] = np.array([x,y,z])

    # TODO: registrator.register()

    # Visualize
    print(
        f"[PROC] Mesh bounds:\n"
        f"[PROC]    X: {verts[:,0].min():.2f} -> {verts[:,0].max():.2f}\n"
        f"[PROC]    Y: {verts[:,1].min():.2f} -> {verts[:,1].max():.2f}\n"
        f"[PROC]    Z: {verts[:,2].min():.2f} -> {verts[:,2].max():.2f}"
    )
    print(
        f"[PROC] Click bounds (pre-snap):\n"
        f"[PROC]    X: {clicks[:,0].min():.2f} -> {clicks[:,0].max():.2f}\n"
        f"[PROC]    Y: {clicks[:,1].min():.2f} -> {clicks[:,1].max():.2f}\n"
        f"[PROC]    Z: {clicks[:,2].min():.2f} -> {clicks[:,2].max():.2f}"
    )

    pv_mesh = to_pyvista_mesh(mesh)
    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, scalars="Colors", rgb=True, opacity=1.0)

    loader = SensorConfigLoader(args.yaml_filepath)
    cam_cfg = loader.get_camera("rgb_1")
    cam = PinholeCameraModel.from_config(cam_cfg)
    proj = ProjectionEngine(cam, backend)
    clicks = backend.snap_points_along_direction(
        clicks,
        direction=[0, 0, 1],   # ENU "up"
        max_distance=20.0,
    )
    print(
        f"[PROC] Click bounds (post-snap):\n"
        f"[PROC]    X: {clicks[:,0].min():.2f} -> {clicks[:,0].max():.2f}\n"
        f"[PROC]    Y: {clicks[:,1].min():.2f} -> {clicks[:,1].max():.2f}\n"
        f"[PROC]    Z: {clicks[:,2].min():.2f} -> {clicks[:,2].max():.2f}"
    )

    cam_cfg = loader.get_camera('rgb_1')
    cam = PinholeCameraModel.from_config(cam_cfg)
    proj.camera = cam
    # T_cam_ins = cam_cfg.T_cam_ins
    # T_cam_world = T_ins_world @ T_cam_ins
    T_cam_world = np.array(
        [[ 2.44943705e-01 ,-5.68428045e-01, -7.85424892e-01 , 5.84144897e+05],
         [-9.61736962e-01, -2.45006580e-01, -1.22612415e-01,  4.09335174e+06],
         [-1.22737924e-01 , 7.85405289e-01 ,-6.06691082e-01 , 1.22659203e+02],
         [ 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]]
    )
    synth = proj.render(T_cam_world)

    landmarks, _ = synth.keypoints_3D
    Pw = landmarks[0:1,:]
    T_world_cam = np.linalg.inv(T_cam_world)
    J_analytic = proj.analytic_projection_jacobian(T_world_cam, Pw)
    J_fd = np.zeros((2,6))

    for k in range(6):
        delta = np.zeros(6)
        delta[k] = 1e-6

        T2 = SE3.apply_se3_update(T_world_cam, delta)
        with np.printoptions(precision=3):
            print("T_world_cam: \n", T_world_cam)
            print("T2: \n", T2)
            print("diff of T's: \n", T_world_cam - T2)

        p1, _ = proj.world_to_image(Pw, T_world_cam)
        p2, _ = proj.world_to_image(Pw, T2)
        print('points: \n', p1, p2)

        J_fd[:,k] = (p2[0] - p1[0]) / 1e-6
    with np.printoptions(precision=3):
        print("J_analytic: \n", J_analytic)
        print("J_fd: \n: ", J_fd)
        print("diff of J's: \n", J_analytic - J_fd)

    draw_camera(loader, proj, "rgb_1", "red", T_ins_world)
    # draw_camera(loader, proj, "rgb_2", "blue", T_ins_world)
    # draw_camera(loader, proj, "rgb_3", "green", T_ins_world)
    draw_camera(loader, proj, "rgb_4", "yellow", T_ins_world)
    # draw_camera(loader, proj, "multispec_1", "misty_rose", T_ins_world)
    # draw_camera(loader, proj, "multispec_2", "lavender", T_ins_world)
    # draw_camera(loader, proj, "multispec_3", "honeydew", T_ins_world)
    # draw_camera(loader, proj, "multispec_4", "light_goldenrod", T_ins_world)

    add_spheres(clicks, color="magenta")

    plotter.set_background("white")
    plotter.show_axes()
    plotter.enable_eye_dome_lighting()

    plotter.show()
