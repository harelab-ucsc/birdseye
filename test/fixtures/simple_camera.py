import numpy as np
from birdseye.camera.camera import PinholeCameraModel


def make_camera():
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
    D = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    T_cam_ins = np.eye(4)

    T_WC = np.eye(4)
    T_WC[2, 2] = -1
    T_WC[:3, 3] = np.array([0, 0, 10])  # camera at z=10 looking down

    return PinholeCameraModel(K, D, T_cam_ins, (640, 480), "test"), T_WC
