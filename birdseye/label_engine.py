import numpy as np


class LabelEngine:
    def __init__(self, scene, camera, targets):
        self.scene = scene
        self.camera = camera
        self.targets = targets

    def process_frame(self, image, T_WC):
        """
        returns:
            labels, projections, visibility
        """

        self.camera.T_WC = T_WC

        labels = []
        projections = []
        visibility = []

        for Xw in self.targets.get():

            # 1. project
            uv = self.camera.project(Xw)

            # 2. raycast visibility test
            origin, direction = self.camera.ray(uv)

            hit, t_hit = self.scene.raycast(origin, direction)

            # distance to target
            t_target = np.linalg.norm(Xw - origin)

            visible = hit and abs(t_hit - t_target) < 1.0  # tolerance (meters)

            labels.append(int(visible))
            projections.append(uv)
            visibility.append(visible)

        return np.array(labels), np.array(projections), np.array(visibility)
