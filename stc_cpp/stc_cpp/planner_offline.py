import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from collections import defaultdict, deque
import math

class MapSubscriber(Node):
    def __init__(self):
        super().__init__('map_subscriber')

        self.cell_size = 0.8  # 2D x 2D cell (D = 1.5 m)
        self.grid_received = False

        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10)

        self.marker_pub = self.create_publisher(MarkerArray, '/grid_centers', 10)
        self.get_logger().info('Waiting for map...')

    def map_callback(self, msg):
        if self.grid_received:
            return

        self.grid_received = True
        self.get_logger().info('Received map')

        resolution = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y
        map_width = msg.info.width
        map_height = msg.info.height
        data = msg.data

        full_width = map_width * resolution
        full_height = map_height * resolution

        map_center_x = origin_x + full_width / 2.0
        map_center_y = origin_y + full_height / 2.0

        half_cells_x = math.floor((full_width / self.cell_size) / 2)
        half_cells_y = math.floor((full_height / self.cell_size) / 2)

        grid_origin_x = map_center_x - (half_cells_x + 0.5) * self.cell_size
        grid_origin_y = map_center_y - (half_cells_y + 0.5) * self.cell_size

        num_cells_x = math.ceil(full_width / self.cell_size)
        num_cells_y = math.ceil(full_height / self.cell_size)

        samples_per_cell = int(self.cell_size / resolution)

        marker_array = MarkerArray()
        marker_id = 0

        blue_cells = []  # List of (x, y)
        blue_set = set()  # For fast lookup

        for cx in range(num_cells_x):
            for cy in range(num_cells_y):
                center_x = grid_origin_x + (cx + 0.5) * self.cell_size
                center_y = grid_origin_y + (cy + 0.5) * self.cell_size

                free_count = 0
                total_count = 0

                for dx in range(-samples_per_cell // 2, samples_per_cell // 2):
                    for dy in range(-samples_per_cell // 2, samples_per_cell // 2):
                        map_x = int((center_x + dx * resolution - origin_x) / resolution)
                        map_y = int((center_y + dy * resolution - origin_y) / resolution)

                        if 0 <= map_x < map_width and 0 <= map_y < map_height:
                            index = map_y * map_width + map_x
                            occ = data[index]
                            if occ == 0:
                                free_count += 1
                            total_count += 1

                # Determine marker color
                if free_count == total_count and total_count > 0:
                    color = (0.0, 0.0, 1.0)  # Blue
                    blue_cells.append((center_x, center_y))
                    blue_set.add((round(center_x, 3), round(center_y, 3)))
                elif free_count > 0:
                    color = (0.0, 1.0, 0.0)  # Green
                else:
                    color = (1.0, 0.0, 0.0)  # Red

                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "cell_centers"
                marker.id = marker_id
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = center_x
                marker.pose.position.y = center_y
                marker.pose.position.z = 0.1
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.3
                marker.scale.y = 0.3
                marker.scale.z = 0.3
                marker.color.r = color[0]
                marker.color.g = color[1]
                marker.color.b = color[2]
                marker.color.a = 1.0
                marker.lifetime.sec = 0

                marker_array.markers.append(marker)
                marker_id += 1

        # === SPANNING TREE FROM BLUE CELLS ===
        adjacency = defaultdict(list)

        for x, y in blue_cells:
            neighbors = [
                (round(x + self.cell_size, 3), round(y, 3)),
                (round(x - self.cell_size, 3), round(y, 3)),
                (round(x, 3), round(y + self.cell_size, 3)),
                (round(x, 3), round(y - self.cell_size, 3))
            ]
            for nx, ny in neighbors:
                if (nx, ny) in blue_set:
                    adjacency[(round(x, 3), round(y, 3))].append((nx, ny))

        # Find closest blue cell to map center
        start = None
        min_dist = float('inf')
        for x, y in blue_cells:
            dist = math.hypot(x - map_center_x, y - map_center_y)
            if dist < min_dist:
                min_dist = dist
                start = (round(x, 3), round(y, 3))

        # BFS to build rooted spanning tree
        tree_edges = []
        if start:
            visited = set()
            queue = deque([start])
            visited.add(start)

            while queue:
                current = queue.popleft()
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        tree_edges.append((current, neighbor))
                        queue.append(neighbor)

        # Marker for the spanning tree (yellow lines)
        tree_marker = Marker()
        tree_marker.header.frame_id = "map"
        tree_marker.header.stamp = self.get_clock().now().to_msg()
        tree_marker.ns = "spanning_tree"
        tree_marker.id = marker_id
        marker_id += 1
        tree_marker.type = Marker.LINE_LIST
        tree_marker.action = Marker.ADD
        tree_marker.scale.x = 0.05
        tree_marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)  # Yellow
        tree_marker.lifetime.sec = 0

        for (x1, y1), (x2, y2) in tree_edges:
            pt1 = Point(x=x1, y=y1, z=0.1)
            pt2 = Point(x=x2, y=y2, z=0.1)
            tree_marker.points.append(pt1)
            tree_marker.points.append(pt2)

        marker_array.markers.append(tree_marker)

        # Marker for the ROOT node (black sphere)
        if start:
            root_marker = Marker()
            root_marker.header.frame_id = "map"
            root_marker.header.stamp = self.get_clock().now().to_msg()
            root_marker.ns = "root_node"
            root_marker.id = marker_id
            marker_id += 1
            root_marker.type = Marker.SPHERE
            root_marker.action = Marker.ADD
            root_marker.pose.position.x = start[0]
            root_marker.pose.position.y = start[1]
            root_marker.pose.position.z = 0.15
            root_marker.pose.orientation.w = 1.0
            root_marker.scale.x = 0.35
            root_marker.scale.y = 0.35
            root_marker.scale.z = 0.35
            root_marker.color.r = 0.0
            root_marker.color.g = 0.0
            root_marker.color.b = 0.0
            root_marker.color.a = 1.0
            root_marker.lifetime.sec = 0
            marker_array.markers.append(root_marker)

        self.marker_pub.publish(marker_array)
        self.get_logger().info(f'Published {marker_id} markers including rooted spanning tree and black root marker')

def main(args=None):
    rclpy.init(args=args)
    node = MapSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
