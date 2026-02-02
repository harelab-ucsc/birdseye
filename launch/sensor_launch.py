from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.actions import ExecuteProcess

def generate_launch_description():
    # Declare arguments
    yaml_param_file_arg = DeclareLaunchArgument(
        'yaml_param_file',
        default_value='/home/pi5-alpha/ros2_iron/src/inertial-sense-sdk/ros2/launch/example_params.yaml',
        description='Path to the YAML parameter file for the inertial_sense_ros2 node'
    )

    antenna_offset_gps1_arg = DeclareLaunchArgument(
        'antenna_offset_gps1',
        default_value='[-0.02, 0.48, -0.36]',  # Default value; update as needed
        description='Offset for GPS1 antenna'
    )

    gps2_enable_arg = DeclareLaunchArgument(
        'gps2_enable',
        default_value='True',
        description='Boolean, whether or not to enable the second GPS antenna for compassing. Default: True'
    )

    antenna_offset_gps2_arg = DeclareLaunchArgument(
        'antenna_offset_gps2',
        default_value='[-0.02, -0.48, -0.36]',  # Default value; update as needed
        description='Offset for GPS2 antenna'
    )

    mag_declination_arg = DeclareLaunchArgument(
        'mag_declination',
        default_value='0.222',  # Default value; update as needed
        description='Magnetic declination value'
    )

    # Get the argument values
    yaml_param_file = LaunchConfiguration('yaml_param_file')
    antenna_offset_gps1 = LaunchConfiguration('antenna_offset_gps1')
    antenna_offset_gps2 = LaunchConfiguration('antenna_offset_gps2')
    gps2_enable = LaunchConfiguration('gps2_enable')
    mag_declination = LaunchConfiguration('mag_declination')

    # Camera driver node
    camera_driver = ExecuteProcess(
        cmd=['ros2', 'launch', 'camera_aravis2', 'camera_driver_gv_example.launch.py'],
        output='screen'
    )

    # Radar altimeter node
    radalt_node = Node(
        package='radalt',
        executable='radalt',
        name='radalt',
        output='screen'
    )

    # Inertial Sense node with parameters
    inertial_sense_node = Node(
        package='inertial_sense_ros2',
        executable='new_target',
        name='inertial_sense_node',
        output='screen',
        arguments=[yaml_param_file],
        parameters=[
            {'antenna_offset_gps1': antenna_offset_gps1},
            {'antenna_offset_gps2': antenna_offset_gps2},
            {'msg/gps2/enable': gps2_enable},
            {'mag_declination': mag_declination}
        ]
    )

    return LaunchDescription([
        yaml_param_file_arg,
        antenna_offset_gps1_arg,
        antenna_offset_gps2_arg,
        gps2_enable_arg,
        mag_declination_arg,
        camera_driver,
        radalt_node,
        inertial_sense_node
    ])
