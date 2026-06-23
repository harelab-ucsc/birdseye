import numpy as np
from fixtures.simple_camera import make_camera
from birdseye.camera.projection_models import FlatWorldBackend

def test_image_to_world_roundtrip_flat():
    cam, T_WC = make_camera()
    backend = FlatWorldBackend(K=cam.K, ground_z=0.0)

    pixels = np.array([
        [320, 240],
        [300, 240],
        [340, 240]
    ])

    rays = cam.image_to_rays(pixels, T_WC)
    world = backend.rays_to_world(rays)

    assert world.shape[0] == len(pixels)
    assert world.shape[1] == 3


def test_projection_consistency():
    cam, T_WC = make_camera()
    backend = FlatWorldBackend(K=cam.K, ground_z=0.0)

    pixels = np.array([[320, 240]])

    rays = cam.image_to_rays(pixels, T_WC)
    world = backend.rays_to_world(rays)

    reproject = cam.world_to_image(world, T_WC)

    assert reproject.shape == (1, 2)
    assert np.allclose(reproject[0], pixels[0], atol=1e-3)
