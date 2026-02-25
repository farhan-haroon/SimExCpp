import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped
from builtin_interfaces.msg import Time

class TwistToTwistStamped(Node):
    def __init__(self):
        super().__init__('twist_to_twiststamped')

        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        self.publisher = self.create_publisher(
            TwistStamped,
            '/diff_drive_controller/cmd_vel',
            10
        )

        self.get_logger().info('Converting /cmd_vel (Twist) to /diff_drive_controller/cmd_vel (TwistStamped)')

    def cmd_vel_callback(self, twist_msg: Twist):
        twist_stamped = TwistStamped()
        twist_stamped.header.stamp = self.get_clock().now().to_msg()
        twist_stamped.header.frame_id = 'base_link'  # Or whatever frame you want
        twist_stamped.twist = twist_msg

        self.publisher.publish(twist_stamped)

def main(args=None):
    rclpy.init(args=args)
    node = TwistToTwistStamped()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
