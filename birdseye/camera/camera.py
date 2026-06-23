import yaml

import numpy as np

from dataclasses import dataclass


@dataclass
class Ray:
    origin: np.ndarray      # (3,)
    direction: np.ndarray   # (3,)


@dataclass
class CameraConfig:
    name: str
    K: np.ndarray
    width: int
    height: int
    T_cam_ins: np.ndarray

    def image_shape(self):
        return (self.width, self.height)


class SensorConfigLoader:
    def __init__(self, yaml_path):
        with open(yaml_path, "r") as f:
            self.raw = yaml.safe_load(f)

    def get_camera(self, name: str) -> CameraConfig:
        cam = self.raw["cameras"][name]

        intr = cam["intrinsics"]
        res = cam["resolution"]

        K = np.array([
            [intr["fx"], 0.0, intr["cx"]],
            [0.0, intr["fy"], intr["cy"]],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        T_cam_ins = np.array(cam["T_cam_ins"], dtype=np.float32)

        return CameraConfig(
            name=name,
            K=K,
            width=res["width"],
            height=res["height"],
            T_cam_ins=T_cam_ins
        )

    def list_cameras(self):
        return list(self.raw["cameras"].keys())


class PinholeCameraModel:
    def __init__(self, K, res, name):
        self.K = K
        width, height = res
        self.width = width
        self.height = height
        self.name = name

        self.Kinv = np.linalg.inv(K)

    @classmethod
    def from_config(cls, cam_cfg):
        """
        Build camera from CameraConfig or dict-like YAML object.
        """

        # Support both dataclass + raw dict usage
        if hasattr(cam_cfg, "K"):
            K = cam_cfg.K
            width = cam_cfg.width
            height = cam_cfg.height
            name = cam_cfg.name
        else:
            intr = cam_cfg["intrinsics"]
            res = cam_cfg["resolution"]

            K = np.array([
                [intr["fx"], 0.0, intr["cx"]],
                [0.0, intr["fy"], intr["cy"]],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)

            width = res["width"]
            height = res["height"]
            name = cam_cfg.get("frame", "unknown")

        return cls(
            K=K,
            res=(width, height),
            name=name
        )

    def image_shape(self):
        return (self.width, self.height)

    def image_to_rays(self, pixels, T_WC):
        """
        Vectorized version:
        pixels: (N,2)
        returns:
            origins: (N,3)
            directions: (N,3)
        """
        pixels = np.asarray(pixels, dtype=np.float32)

        # 1. Backproject to camera frame
        ones = np.ones((pixels.shape[0], 1), dtype=np.float32)
        pix_h = np.hstack([pixels, ones])  # (N,3)
        dirs_cam = (self.Kinv @ pix_h.T).T  # (N,3)
        dirs_cam /= (np.linalg.norm(dirs_cam, axis=1, keepdims=True) + 1e-12)

        # 2. Transform to world
        R = T_WC[:3, :3]
        t = T_WC[:3, 3]
        origins = np.repeat(t[None, :], dirs_cam.shape[0], axis=0)
        dirs_world = (R @ dirs_cam.T).T
        dirs_world /= (np.linalg.norm(dirs_world, axis=1, keepdims=True) + 1e-12)
        return origins.astype(np.float32), dirs_world.astype(np.float32)

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
