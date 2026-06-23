import numpy as np
from fixtures.simple_camera import make_camera

def test_ray_determinism():
    cam, T_WC = make_camera()

    pixels = np.array([[320, 240]])  # principal point

    rays1 = cam.image_to_rays(pixels, T_WC)
    rays2 = cam.image_to_rays(pixels, T_WC)

    assert np.allclose(rays1[0].direction, rays2[0].direction)
    assert np.allclose(rays1[0].origin, rays2[0].origin)


def test_ray_normalization():
    cam, T_WC = make_camera()

    pixels = np.array([[100, 200], [400, 300]])
    rays = cam.image_to_rays(pixels, T_WC)

    for r in rays:
        norm = np.linalg.norm(r.direction)
        assert np.isclose(norm, 1.0, atol=1e-6)
