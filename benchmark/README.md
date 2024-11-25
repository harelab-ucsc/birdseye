# birdseye benchmark

**Visual-inertial system validator and tester for birdseye**

## Installation

This software uses [OpenCV](https://github.com/opencv/opencv) for vison tools, [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) for linear algebra operations, , and [Pangolin](https://github.com/stevenlovegrove/Pangolin) for smooth visualization. Install these software dependencies using the following command on Ubuntu. 

```bash
sudo apt-get install cmake libeigen3-dev libpangolin-dev libsophus-dev libyaml-cpp-dev nlohmann-json3-dev
```

To build this simulator, run
```bash
git clone https://github.com/harelab-ucsc/birdseye.git
cd birdseye/benchmark
mkdir build
cd build
cmake ..
make
```

## Usage

The simulation and system parameters are specified in a yaml file. An example is provided in the config directory. To run a benchmark, simply type the following in the root of this subdirectory. These include the camera intrinsics. 

```bash
./build/sim3D config/sim.yaml
```

The provided data is expexcted in the [Nerfstudio](https://docs.nerf.studio/) format with a directpry of images (images/*.png) and a JSON (poses.json) specifying the camera poses for each image. See the example below.

```json
            "w": 1920,
            "h": 1200,
            "fl_x": 4205.423600947519,
            "fl_y": 4225.524941785368,
            "cx": 955.1910100382079,
            "cy": 320.49628086862134,
            "timestamp": 1731710646.8194637,
            "file_path": "images/1731710646.819463634.png",
            "transform_matrix": [
                [
                    0.1626978022551909,
                    -0.9770937934787769,
                    -0.13717559508376637,
                    8274487.293483774
                ],
                [
                    0.9558551066107057,
                    0.12160755231100541,
                    0.26749321185249453,
                    19995929.883672442
                ],
                [
                    -0.24468436874383348,
                    -0.17464055074976345,
                    0.9537453736329484,
                    -0.040102166821745965
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ]
        }
```


The program provides Euclidean backprojection error terms in a file called l2_errors.txt. To anaylze this output, we provide a script that fits a gamma function to the error distribution.

```bash
python3 scripts/analyze.py --bins 250 l2_errors.txt
```
