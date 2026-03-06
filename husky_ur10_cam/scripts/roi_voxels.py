#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

class RoiVoxelPublisher(Node):

    def __init__(self):
        super().__init__('roi_voxel_publisher')

        self.pub = self.create_publisher(Marker, '/roi_voxels', 10)
        self.timer = self.create_timer(1.0, self.publish_voxels)

        self.resolution = 0.05

        self.xmin, self.xmax = -2.0, 2.0
        self.ymin, self.ymax = -2.0, 2.0
        self.zmin, self.zmax = 0.0, 2.0

    def publish_voxels(self):

        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "roi_voxels"
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD

        marker.scale.x = self.resolution
        marker.scale.y = self.resolution
        marker.scale.z = self.resolution

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.5

        x = self.xmin
        while x <= self.xmax:
            y = self.ymin
            while y <= self.ymax:
                z = self.zmin
                while z <= self.zmax:
                    p = Point()
                    p.x = x
                    p.y = y
                    p.z = z
                    marker.points.append(p)

                    # height normalized
                    t = (z - self.zmin) / (self.zmax - self.zmin)
                    c = ColorRGBA()
                    c.r = t
                    c.g = 1.0 - abs(t - 0.5) * 2
                    c.b = 1.0 - t
                    c.a = 1.0
                    marker.colors.append(c)

                    z += self.resolution
                y += self.resolution
            x += self.resolution

        self.pub.publish(marker)


def main():
    rclpy.init()
    node = RoiVoxelPublisher()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()