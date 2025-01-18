from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os

def generate_launch_description():
    # Declare arguments for parameters
    yaml_param_file_arg = DeclareLaunchArgument(
        'yaml_param_file',
        default_value='/home/pi5-alpha/ros2_iron/src/inertial-sense-sdk/ros2/launch/example_params.yaml',
        description='Path to the YAML parameter file for the inertial_sense_ros2 node'
    )

    antenna_offset_gps1_arg = DeclareLaunchArgument(
        'antenna_offset_gps1',
        default_value='[-0.11, 0.0, -0.33]',  # Default value; update as needed
        description='Offset for GPS1 antenna'
    )

    mag_declination_arg = DeclareLaunchArgument(
        'mag_declination',
        default_value='0.222',  # Default value; update as needed
        description='Magnetic declination value'
    )

    # Get the values of the arguments
    yaml_param_file = LaunchConfiguration('yaml_param_file')
    antenna_offset_gps1 = LaunchConfiguration('antenna_offset_gps1')
    mag_declination = LaunchConfiguration('mag_declination')

    # Include the previous launch script and pass the arguments
    current_dir = os.path.dirname(os.path.realpath(__file__))
    previous_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(current_dir, 'sensor_launch.py')
        ),
        launch_arguments={
            'yaml_param_file': yaml_param_file,
            'antenna_offset_gps1': antenna_offset_gps1,
            'mag_declination': mag_declination
        }.items()
    )

    # Execute `ros2 topic hz /image_raw`
    topic_hz_process = ExecuteProcess(
        cmd=['ros2', 'topic', 'hz', '/camera_driver_gv_example/vis/image_raw'],
        output='screen'
    )

    return LaunchDescription([
        yaml_param_file_arg,
        antenna_offset_gps1_arg,
        mag_declination_arg,
        previous_launch,
        topic_hz_process,
    ])
