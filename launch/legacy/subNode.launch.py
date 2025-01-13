import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch_ros.actions import Node
import os

def generate_launch_description():
    return LaunchDescription([
          # Declare the launch arguments
        DeclareLaunchArgument('sensorID', default_value='cam0'),
        # DeclareLaunchArgument('cam_topic', default_value='/cam0/rgb_cam/image_rect'),
        DeclareLaunchArgument('cam_topic', default_value='/cam0/rgb_cam/image_raw'),
        DeclareLaunchArgument('sensors_yaml', default_value=os.path.join(os.path.expanduser('~'), 'ros2_ws/src/birdseye/config/birdsEyeSensorParams.yaml')),
        DeclareLaunchArgument('clicks_csv', default_value=os.path.join(os.path.expanduser('~'),'catch/data_farmAlt_20240722.csv')),
        # DeclareLaunchArgument('clicks_csv', default_value=os.path.join(os.path.expanduser('~'),'catch/data_mima_0_20240805.csv')),
        # DeclareLaunchArgument('clicks_csv', default_value=os.path.join(os.path.expanduser('~'),'farmFlight20240808_1/data_farmFlight20240808.csv')),

        DeclareLaunchArgument('dir_name', default_value='parsed_flight'),

        # Commented out image_proc node
        # Node(
        #     package='image_proc',
        #     executable='image_proc',
        #     namespace=launch.substitutions.LaunchConfiguration('sensorID'),
        #     name='image_proc',
        #     remappings=[
        #         ('image', '/' + launch.substitutions.LaunchConfiguration('sensorID') + '/rgb_cam/image_raw'),
        #         ('camera_info', '/' + launch.substitutions.LaunchConfiguration('sensorID') + '/rgb_cam/camera_info')
        #     ]
        # ),

        # BirdsEye sub_node
        Node(
            package='birdseye',
            executable='sub_node',
            namespace=launch.substitutions.LaunchConfiguration('sensorID'),
            name='sub_node',
            remappings=[
                ('/image', launch.substitutions.LaunchConfiguration('cam_topic')),
                ('/rtk/fix', '/ublox_gps_node/fix'),
                ('/rtk/fix_status', '/ublox_gps_node/navpvt')
            ],
            parameters=[
                {'sensorID': launch.substitutions.LaunchConfiguration('sensorID')},
                {'sensors_yaml': launch.substitutions.LaunchConfiguration('sensors_yaml')},
                {'clicks_csv': launch.substitutions.LaunchConfiguration('clicks_csv')}
            ]
        ),

        # Execute rqt_graph
        # ExecuteProcess(
        #     cmd=['rqt_graph'],
        #     output='screen'
        # ),
    ])
