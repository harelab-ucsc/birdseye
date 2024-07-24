from launch import LaunchDescription
import launch_ros.actions
import os
import yaml
from launch.substitutions import EnvironmentVariable
import pathlib
import launch.actions
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import sys

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('bagpath', default_value='/home/mwmaster/catch/Thesis_polygons/polygon_flight_7'),
        DeclareLaunchArgument('rl_launch_path', default_value='/home/mwmaster/ros2_ws/launch/dualEKF_navsat_gx5.launch.py'),
        DeclareLaunchArgument('be_launch_path', default_value='/home/mwmaster/ros2_ws/src/birdseye/launch/subNode.launch.py'),
        launch_ros.actions.Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments = ['-0.0013182011389418422', '0.048743269467030205', '0.0355464324080468', '0.9997894', '-0.0065577', '-0.018069', '-0.0071814', 'base_link', 'cam0']
            ),
        launch_ros.actions.Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments = ['-0.05', '0', '0.33', '0', '0', '0', '1', 'base_link', 'gps']
            ),
        launch_ros.actions.Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments = ['0', '0', '-0.025', '0.8939967', '0', '0', '-0.4480736', 'base_link', 'radalt']
            ),
        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', launch.substitutions.LaunchConfiguration('bagpath'), '--read-ahead-queue-size', '10000'],
            output='screen'
        ),
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource([launch.substitutions.LaunchConfiguration('rl_launch_path')])
        ),
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource([launch.substitutions.LaunchConfiguration('be_launch_path')])
        ),
    ])
