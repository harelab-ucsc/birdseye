from setuptools import find_packages, setup

package_name = 'birdseye'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
    ],
    install_requires=['setuptools',],
    zip_safe=True,
    maintainer='Morgan Masters',
    maintainer_email='mwmaster@ucsc.edu',
    description='birdsEye v1: ROS 2 UAV imagery ingestion to SQLite and human-in-the-loop geospatial annotation (flat-world projection).',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sub_node = birdseye.sub_node:main',
        ],
    },
)
