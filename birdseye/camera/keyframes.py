import numpy as np


class TriangleCoverageKeyframeSelector:
    """
    Geometry-driven keyframe selection.

    A frame is accepted if it observes enough mesh area that has
    not yet been sufficiently covered.

    Coverage is tracked per-triangle.
    """

    def __init__(
        self,
        projection_engine,
        mesh,
        min_novel_area_fraction=0.10,
        max_hits_per_triangle=3,
        ray_stride=20,
    ):
        self.engine = projection_engine

        self.min_novel_area_fraction = min_novel_area_fraction
        self.max_hits_per_triangle = max_hits_per_triangle
        self.ray_stride = ray_stride

        self.triangle_area = self._compute_triangle_areas(mesh)

        self.triangle_observation_count = np.zeros(
            len(self.triangle_area),
            dtype=np.uint16,
        )

    def should_keep(self, T_WC):
        """
        Returns:
            keep: bool
            stats: dict
        """

        origins, dirs = self.engine.camera_frustum(
            self.engine.camera.image_shape(),
            T_WC,
            stride=self.ray_stride,
        )

        result = self.engine.backend.raycast(
            origins,
            dirs,
        )

        tri_ids = result["primitive_ids"].numpy()

        valid = tri_ids >= 0
        visible_triangles = tri_ids[valid]

        if len(visible_triangles) == 0:
            return False, {
                "visible_area": 0.0,
                "novel_area": 0.0,
                "novel_fraction": 0.0,
            }

        #
        # remove duplicates
        #
        visible_triangles = np.unique(visible_triangles)

        visible_area = self.triangle_area[visible_triangles].sum()

        under_observed_mask = (
            self.triangle_observation_count[visible_triangles]
            < self.max_hits_per_triangle
        )

        novel_triangles = visible_triangles[under_observed_mask]

        novel_area = self.triangle_area[novel_triangles].sum()

        novel_fraction = novel_area / visible_area if visible_area > 0 else 0.0

        keep = novel_fraction >= self.min_novel_area_fraction

        if keep:
            self.triangle_observation_count[visible_triangles] += 1

        return keep, {
            "visible_area": visible_area,
            "novel_area": novel_area,
            "novel_fraction": novel_fraction,
            "n_visible_triangles": len(visible_triangles),
            "n_novel_triangles": len(novel_triangles),
        }

    @staticmethod
    def _compute_triangle_areas(mesh):
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]

        return 0.5 * np.linalg.norm(
            np.cross(v1 - v0, v2 - v0),
            axis=1,
        )
