import numpy as np
from fixtures.simple_camera import make_camera


def test_principal_ray_identity_pose():
    cam, T_cam_world = make_camera()
    pixels = np.array([[320, 240]])
    origins, dirs = cam.image_to_rays(pixels, T_cam_world)
    assert origins.shape == (1, 3)
    assert np.allclose(
        dirs[0],
        np.array([0.0, 0.0, -1.0]),
        atol=1e-6,
    )


def test_ray_determinism():
    cam, T_cam_world = make_camera()
    pixels = np.array([[320, 240]])  # principal point
    origins1, dirs1 = cam.image_to_rays(pixels, T_cam_world)
    origins2, dirs2 = cam.image_to_rays(pixels, T_cam_world)
    assert np.allclose(dirs1[0], dirs2[0])
    assert np.allclose(origins1[0], origins2[0])


def test_ray_normalization():
    cam, T_cam_world = make_camera()
    pixels = np.array([[100, 200], [400, 300]])
    _, dirs = cam.image_to_rays(pixels, T_cam_world)
    for d in dirs:
        assert np.isclose(np.linalg.norm(d), 1.0, atol=1e-6)


def test_world_to_image_consistency():
    cam, T_cam_world = make_camera()
    pts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [-1, 0, 0],
        ]
    )
    pixels, in_front = cam.world_to_image(pts, T_cam_world)
    assert np.all(in_front)

    origins, dirs = cam.image_to_rays(pixels, T_cam_world)
    vecs = pts - origins
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    assert np.allclose(vecs, dirs, atol=1e-5)
