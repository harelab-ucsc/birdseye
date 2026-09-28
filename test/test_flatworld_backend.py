import numpy as np
from fixtures.simple_camera import make_camera
from birdseye.camera.projection_models import FlatWorldBackend


def test_flat_world_intersection():
    cam, T_cam_world = make_camera()
    backend = FlatWorldBackend(ground_z=0.0)
    pixels = np.array([[320, 240]])
    rays = cam.image_to_rays(pixels, T_cam_world)
    pts = backend.rays_to_world(rays)
    assert pts.shape == (1, 3)
    # camera at z=10 looking down → ground hit must be z=0
    assert np.isclose(pts[0, 2], 0.0, atol=1e-6)


def test_image_to_world_consistency():
    cam, T_cam_world = make_camera()
    backend = FlatWorldBackend(ground_z=0.0)
    pixels = np.array([[320, 240], [300, 240], [340, 240]])
    origins, dirs = cam.image_to_rays(pixels, T_cam_world)
    world = backend.raycast(origins, dirs)
    assert world.shape[0] == len(pixels)
    assert world.shape[1] == 3


def test_projection_consistency():
    cam, T_cam_world = make_camera()
    backend = FlatWorldBackend(ground_z=0.0)
    pixels = np.array([[320, 240]])
    origins, dirs = cam.image_to_rays(pixels, T_cam_world)
    world = backend.raycast(origins, dirs)
    reproject, in_front = cam.world_to_image(world, T_cam_world)
    assert reproject.shape == (1, 2)
    assert np.allclose(reproject[0], pixels[0], atol=1e-3)


def test_flat_world_intersection():
    backend = FlatWorldBackend(ground_z=0.0)
    origins = np.array([[0, 0, 0]])
    dirs = np.array([[0, 0, -1]])
    hits = backend.raycast(origins, dirs)
    assert hits.shape == (1, 3)
    assert np.allclose(hits[0], np.array([0, 0, 0]))
