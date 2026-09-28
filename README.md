# birdsEye

> **Version note** — This is v2 (3-D ray casting / SE(3)), under active development. The code used in the manuscript ([arXiv:2609.28767](https://arxiv.org/abs/2609.28767)) is the v1 SQLite line — use tag [`v1.0.0`](https://github.com/harelab-ucsc/birdseye/tree/v1.0.0) / branch [`v1`](https://github.com/harelab-ucsc/birdseye/tree/v1) to reproduce it.

## Overview

birdsEye is a ROS 2 (`ament_python`) package for human-in-the-loop geospatial annotation of UAV camera-payload imagery. v2 projects RTK-surveyed annotations into images by casting camera rays against a 3-D model of the scene instead of a flat ground plane.

- `birdseye/camera/projection_models.py` — `ProjectionEngine` over a `GeometryBackend` interface, with `FlatWorldBackend` (ray–plane intersection) and `MeshBackend` (Open3D mesh ray casting) backends.
- `birdseye/camera/` — `PinholeCameraModel` and sensor-calibration loading (`camera.py`), keyframe selection by triangle coverage (`keyframes.py`), GeoJSON / point-cloud / GeoTIFF readers (`geo_datasets.py`).
- `birdseye/geometry/se3.py` — `SE3` rigid-body pose type.
- `birdseye/annotator_node.py` — `AnnotatorNode`, the ROS 2 annotation node.
- `birdseye/hloc_localizer.py` — `HlocLocalizationNode`, a ROS 2 node for hloc-based visual localization.
- `birdseye/PathPlanner/` — flight-plan generation from surveyed clicks (`GPS_path_planner.py`, `kml_filegen.py`, `plan_filegen.py`).
- `birdseye/cnn/` — tile/frame loaders, dataset generator, training and tiled inference.
- `scripts/` — bag utilities (`images_from_bag.py`, `ffc_from_bag.py` flat-field correction, `altimetry_DEM.py`) and `gen_leg_masks.py` (static drone-leg occlusion masks for COLMAP).

## Usage

Build in a ROS 2 workspace (`colcon build --packages-select birdseye`, then `source install/setup.bash`) and run the annotator:

```bash
ros2 run birdseye annotator_node --ros-args \
  -p yaml_filepath:=/path/to/payload_calibration.yaml \
  -p mesh_filepath:=/path/to/reconstruction_mesh \
  -p click_filepath:=/path/to/clicks.csv \
  -p save_dir:=/path/to/output
```

- `yaml_filepath` — payload calibration YAML.
- `mesh_filepath` — 3-D reconstruction of the scene.
- `click_filepath` — RTK annotations and GCPs (CSV).
- `save_dir` — output directory (labels are written to `labels.txt`).

## Tests

```bash
python3 -m pytest test/
```

## Branches and versions

| Ref | What | Status |
|---|---|---|
| `main` | v2 — 3-D ray casting, SE(3) poses, `annotator_node` | active development |
| `v1` / tag `v1.0.0` | v1 — `sub_node` → SQLite, flat-world projection; manuscript code | frozen, bug fixes only |
| `pose-bound` | v1 snapshot linked by the current manuscript draft | frozen; removed after the revised manuscript cites `v1.0.0` |
| tags `archive/*` | retired experiment branches (YOLO/CNN training, ceres pose optimizer, early KML/CSV path planners, ortho mosaicking prototypes) | read-only history |

## Citation

Citation metadata is in [`CITATION.cff`](CITATION.cff) (GitHub: "Cite this repository"). Please cite the article:

```bibtex
@misc{masters2026birdseye,
  title         = {Human-in-the-Loop Geospatial Annotation for Rapid Dataset Construction in Field-Deployed {UAV} Systems},
  author        = {Masters, Morgan and Korycki, Adam and Bender, Nikolaas and Altaffer, T. Luca and Kuipers, Nick and Josephson, Colleen and McGuire, Steve},
  year          = {2026},
  eprint        = {2609.28767},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2609.28767}
}
```

## License

MIT — see [`LICENSE`](LICENSE).
