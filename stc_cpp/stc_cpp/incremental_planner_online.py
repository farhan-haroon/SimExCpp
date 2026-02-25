import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from collections import defaultdict, deque
import math

class MapSubscriber(Node):
    def __init__(self):
        super().__init__('anchored_grid_spanning_tree')

        self.cell_size = 0.8  # Example: 2D × 2D where D=0.4m → cell_size=0.8m
        self.robot_position = (0.0, 0.0)  # assume starting at odom (0,0)

        # Tree state persistence between map updates
        self.tree_edges = []
        self.visited_nodes = set()
        self.root_node = (0.0, 0.0)  # fixed root at map origin (0,0)

        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/diff_drive_controller/odom', self.odom_callback, 10)

        self.marker_pub = self.create_publisher(MarkerArray, '/grid_centers', 10)
        self.get_logger().info('Initialized anchored grid & incremental spanning tree node.')

    def odom_callback(self, msg: Odometry):
        self.robot_position = (
            msg.pose.pose.position.x, msg.pose.pose.position.y
        )

    def map_callback(self, msg: OccupancyGrid):
        resolution = msg.info.resolution
        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y
        w, h = msg.info.width, msg.info.height
        data = msg.data

        # Map bounds in world coords
        map_min_x = ox
        map_min_y = oy
        map_max_x = ox + w * resolution
        map_max_y = oy + h * resolution

        # Grid anchored at (0,0) — expand in ± directions
        num_x_pos = int(math.ceil((map_max_x) / self.cell_size))
        num_x_neg = int(math.ceil(abs(map_min_x) / self.cell_size))
        num_y_pos = int(math.ceil((map_max_y) / self.cell_size))
        num_y_neg = int(math.ceil(abs(map_min_y) / self.cell_size))

        samples_per_cell = max(1, int(self.cell_size / resolution))

        marker_array = MarkerArray()
        blue_cells = []
        blue_set = set()
        cell_positions = {}

        # Iterate over anchored grid
        for ix in range(-num_x_neg, num_x_pos + 1):
            for iy in range(-num_y_neg, num_y_pos + 1):
                cx_world = ix * self.cell_size
                cy_world = iy * self.cell_size

                free_cnt = total_cnt = 0
                for dx in range(-samples_per_cell // 2, samples_per_cell // 2 + 1):
                    for dy in range(-samples_per_cell // 2, samples_per_cell // 2 + 1):
                        mx = int((cx_world + dx * resolution - ox) / resolution)
                        my = int((cy_world + dy * resolution - oy) / resolution)
                        if 0 <= mx < w and 0 <= my < h:
                            val = data[my * w + mx]
                            total_cnt += 1
                            if val == 0:
                                free_cnt += 1

                cell_key = (round(cx_world, 3), round(cy_world, 3))
                if free_cnt == total_cnt and total_cnt > 0:
                    blue_cells.append((cx_world, cy_world))
                    blue_set.add(cell_key)
                    cell_positions[cell_key] = (cx_world, cy_world)
                    # Color root black if this is (0,0)
                    if cell_key == self.root_node:
                        color = (0.0, 0.0, 0.0)
                    else:
                        color = (0.0, 0.0, 1.0)  # Blue
                elif free_cnt > 0:
                    color = (0.0, 1.0, 0.0)  # Green
                else:
                    color = (1.0, 0.0, 0.0)  # Red

                m = Marker()
                m.header.frame_id = "map"
                m.header.stamp = self.get_clock().now().to_msg()
                m.ns = "cell_centers"
                m.id = ix * 10000 + iy
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.pose.position.x = cx_world
                m.pose.position.y = cy_world
                m.pose.position.z = 0.1
                m.scale.x = m.scale.y = m.scale.z = 0.25
                m.color.r, m.color.g, m.color.b = color
                m.color.a = 1.0
                marker_array.markers.append(m)

        # Build adjacency for blue cells
        adjacency = defaultdict(list)
        for x, y in blue_cells:
            p = (round(x, 3), round(y, 3))
            for dx, dy in [(self.cell_size, 0), (-self.cell_size, 0),
                           (0, self.cell_size), (0, -self.cell_size)]:
                nb = (round(x + dx, 3), round(y + dy, 3))
                if nb in blue_set:
                    adjacency[p].append(nb)

        # If root node is blue, expand the tree incrementally
        if self.root_node in blue_set:
            if not self.visited_nodes:
                # First time init
                self.visited_nodes.add(self.root_node)
                dq = deque([self.root_node])
                while dq:
                    cur = dq.popleft()
                    for nb in adjacency[cur]:
                        if nb not in self.visited_nodes:
                            self.visited_nodes.add(nb)
                            self.tree_edges.append((cur, nb))
                            dq.append(nb)
            else:
                # Incrementally grow tree
                new_blue_nodes = blue_set - self.visited_nodes
                for nb in list(new_blue_nodes):
                    for dx, dy in [(self.cell_size, 0), (-self.cell_size, 0),
                                   (0, self.cell_size), (0, -self.cell_size)]:
                        neighbor = (round(nb[0] + dx, 3), round(nb[1] + dy, 3))
                        if neighbor in self.visited_nodes:
                            self.visited_nodes.add(nb)
                            self.tree_edges.append((neighbor, nb))
                            break

        # Publish tree edges as orange lines
        tree_marker = Marker()
        tree_marker.header.frame_id = "map"
        tree_marker.header.stamp = self.get_clock().now().to_msg()
        tree_marker.ns = "spanning_tree"
        tree_marker.id = 0
        tree_marker.type = Marker.LINE_LIST
        tree_marker.action = Marker.ADD
        tree_marker.scale.x = 0.05
        tree_marker.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0)  # orange
        for a, b in self.tree_edges:
            pt1 = Point(x=a[0], y=a[1], z=0.1)
            pt2 = Point(x=b[0], y=b[1], z=0.1)
            tree_marker.points.append(pt1)
            tree_marker.points.append(pt2)
        marker_array.markers.append(tree_marker)

        self.marker_pub.publish(marker_array)
        self.get_logger().info(f'Published anchored grid + spanning tree, root={self.root_node}, edges={len(self.tree_edges)}')

def main(args=None):
    rclpy.init(args=args)
    node = MapSubscriber()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
