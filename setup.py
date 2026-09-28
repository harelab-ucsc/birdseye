from setuptools import find_packages, setup

package_name = "birdseye"

setup(
    name=package_name,
    version="2.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
    ],
    install_requires=[
        "setuptools",
    ],
    zip_safe=True,
    maintainer="Morgan Masters",
    maintainer_email="mwmaster@ucsc.edu",
    description="birdsEye v2: human-in-the-loop geospatial annotation of UAV imagery against 3-D models (ray casting, SE(3) poses).",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "annotator_node = birdseye.annotator_node:main",
        ],
    },
)
