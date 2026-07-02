import numpy as np
from fixtures.simple_camera import make_camera
from fixtures.simple_mesh import make_plane
from birdseye.camera.projection_models import MeshBackend


def test_flat_world_intersection():
    cam, T_cam_world = make_camera()
    scene = make_plane()
    backend = MeshBackend(scene)
    pixels = np.array([[320, 240]])
    origins, dirs = cam.image_to_rays(pixels, T_cam_world)
    result = backend.raycast(origins, dirs)

    test = origins + result["t_hit"].numpy() * dirs
    assert test.shape == (1, 3)
    # camera at z=10 looking down → ground hit must be z=0
    assert np.isclose(test[0, 2], 0.0, atol=1e-6)


def test_image_to_world_consistency():
    cam, T_cam_world = make_camera()
    scene = make_plane()
    backend = MeshBackend(scene)
    pixels = np.array([[320, 240], [300, 240], [340, 240]])
    origins, dirs = cam.image_to_rays(pixels, T_cam_world)
    result = backend.raycast(origins, dirs)

    test = origins + result["t_hit"].numpy() * dirs
    assert test.shape[0] == len(pixels)
    assert test.shape[1] == 3


def test_projection_consistency():
    cam, T_cam_world = make_camera()
    scene = make_plane()
    backend = MeshBackend(scene)
    pixels = np.array([[320, 240]])
    origins, dirs = cam.image_to_rays(pixels, T_cam_world)
    result = backend.raycast(origins, dirs)

    origins, dirs = cam.image_to_rays(pixels, T_cam_world)
    result = backend.raycast(origins, dirs)

    test = origins + result["t_hit"].numpy() * dirs
    reproject, in_front = cam.world_to_image(test, T_cam_world)
    assert reproject.shape == (1, 2)
    assert np.allclose(reproject[0], pixels[0], atol=1e-3)


def test_mesh_raycast_hit():
    scene = make_plane()
    backend = MeshBackend(scene)
    origins = np.array([[0, 0, 10]])
    dirs = np.array([[0, 0, -1]])
    result = backend.raycast(origins, dirs)
    t_hit = result["t_hit"].numpy()
    assert np.isfinite(t_hit[0])
    assert np.isclose(t_hit[0], 10.0, atol=1e-2)


def test_mesh_raycast_miss():
    scene = make_plane()
    backend = MeshBackend(scene)
    origins = np.array([[0, 0, 10]])
    dirs = np.array([[1, 0, 0]])
    result = backend.raycast(origins, dirs)
    t_hit = result["t_hit"].numpy()
    assert np.isinf(t_hit[0])
