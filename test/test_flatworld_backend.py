import numpy as np
from fixtures.simple_camera import make_camera
from birdseye.camera.projection_models import FlatWorldBackend

def test_flat_world_intersection():
    cam, T_WC = make_camera()
    backend = FlatWorldBackend(K=cam.K, ground_z=0.0)

    pixels = np.array([[320, 240]])

    rays = cam.image_to_rays(pixels, T_WC)
    pts = backend.rays_to_world(rays)

    assert pts.shape == (1, 3)

    # camera at z=10 looking down → ground hit must be z=0
    assert np.isclose(pts[0, 2], 0.0, atol=1e-6)
