#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped, Quaternion
from nav_msgs.msg import Odometry
import tf2_ros
from tf2_ros import Buffer, TransformListener
from tf_transformations import quaternion_from_euler

from inertial_sense_ros2.msg import DIDINS2


class TfBroadcasterNode(Node):
    def __init__(self):
        super().__init__('imu_cam_tf_broadcaster')

        # Initialize broadcaster, odometry publisher, and subscriber
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.camera_odom_publisher = self.create_publisher(Odometry, '/camera_odom', 10)

        # Subscriber to INS message
        self.subscription = self.create_subscription(
            DIDINS2,  # Replace with actual INS message type
            '/ins/data',  # Replace with actual topic
            self.ins_callback,
            10
        )

        t = TransformStamped()
        t.header.stamp = current_time
        t.header.frame_id = 'imu_base_link'
        t.child_frame_id = 'camera_link'

        # Static translation (adjust these values as needed)
        t.transform.translation.x = 0.00572684955927241
        t.transform.translation.y = 0.09686755356261537
        t.transform.translation.z = 0.0033388945470177764

        # Static rotation (Euler angles: roll, pitch, yaw)
        q = [-0.003443, 0.0042536, 0.9999809, 0.0028842]
        t.transform.rotation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        # Send transform
        self.tf_broadcaster.sendTransform(t)

        t = TransformStamped()
        t.header.stamp = current_time
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'

        # Static translation (adjust these values as needed)
        t.transform.translation.x = 0
        t.transform.translation.y = 0
        t.transform.translation.z = 0

        # Static rotation (Euler angles: roll, pitch, yaw)
        q = [0., 0., 0., 1]
        t.transform.rotation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        # Send transform
        self.tf_broadcaster.sendTransform(t)


    def ins_callback(self, msg):
        """Callback for INS data processing."""
        # Extract quaternion (NED: W, X, Y, Z)
        q_ned = np.array([msg.pose.pose.orientation.w,
                          msg.pose.pose.orientation.x,
                          msg.pose.pose.orientation.y,
                          msg.pose.pose.orientation.z])

        # Convert Quaternion (NED → ENU)
        q_enu = np.array([q_ned[0], q_ned[2], q_ned[1], -q_ned[3]])  # (W, Y, X, -Z)

        # Extract Position (Latitude, Longitude, Altitude)
        latitude = msg.pose.pose.position.x
        longitude = msg.pose.pose.position.y
        altitude = msg.pose.pose.position.z  # Ellipsoid height

        # Convert to UTM or Local ENU (not shown here, but needed for real deployment)

        # Create TF TransformStamped message
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "odom"
        t.child_frame_id = "imu_base_link"

        # Set Position (Assume pre-converted to ENU)
        t.transform.translation.x = latitude   # Replace with actual ENU conversion
        t.transform.translation.y = longitude  # Replace with actual ENU conversion
        t.transform.translation.z = altitude   # Replace with actual ENU conversion

        # Set Rotation (ENU)
        t.transform.rotation.x = q_enu[1]
        t.transform.rotation.y = q_enu[2]
        t.transform.rotation.z = q_enu[3]
        t.transform.rotation.w = q_enu[0]

        # Publish Transform
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = TfBroadcasterNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
