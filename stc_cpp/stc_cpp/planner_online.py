import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from collections import defaultdict, deque
from std_msgs.msg import ColorRGBA
import math

class MapSubscriber(Node):
    def __init__(self):
        super().__init__('grid_marker_publisher')

        self.cell_size = 0.8  # cell size in meters
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)

        self.marker_pub = self.create_publisher(
            MarkerArray, '/grid_centers', 10)

        self.get_logger().info('Initialized grid marker node anchored at (0,0).')

    def map_callback(self, msg: OccupancyGrid):
        resolution = msg.info.resolution
        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y
        w, h = msg.info.width, msg.info.height
        data = msg.data

        # Map boundaries in map frame
        min_x = ox
        max_x = ox + w * resolution
        min_y = oy
        max_y = oy + h * resolution

        # Determine how many steps to take in +x/-x and +y/-y from (0,0)
        max_pos_x = math.ceil((max_x) / self.cell_size)
        max_neg_x = math.floor((min_x) / self.cell_size)
        max_pos_y = math.ceil((max_y) / self.cell_size)
        max_neg_y = math.floor((min_y) / self.cell_size)

        samples_per_cell = max(1, int(self.cell_size / resolution))

        marker_array = MarkerArray()
        marker_id = 0
        blue_cells = []
        blue_set = set()

        # Loop through the grid centered at (0,0)
        for gx in range(max_neg_x, max_pos_x + 1):
            for gy in range(max_neg_y, max_pos_y + 1):
                center_x = gx * self.cell_size
                center_y = gy * self.cell_size

                free_cnt = 0
                total_cnt = 0

                # Check occupancy of this cell
                for dx in range(-samples_per_cell // 2, samples_per_cell // 2 + 1):
                    for dy in range(-samples_per_cell // 2, samples_per_cell // 2 + 1):
                        wx = center_x + dx * resolution
                        wy = center_y + dy * resolution

                        mx = int((wx - ox) / resolution)
                        my = int((wy - oy) / resolution)

                        if 0 <= mx < w and 0 <= my < h:
                            val = data[my * w + mx]
                            total_cnt += 1
                            if val == 0:
                                free_cnt += 1

                # Color assignment
                if total_cnt > 0 and free_cnt == total_cnt:
                    color = (0.0, 0.0, 1.0)  # Blue - all free
                    blue_cells.append((center_x, center_y))
                    blue_set.add((round(center_x,3), round(center_y,3)))
                elif free_cnt > 0:
                    color = (0.0, 1.0, 0.0)  # Green - partially free
                else:
                    color = (1.0, 0.0, 0.0)  # Red - outside map or all occupied

                # Marker creation
                m = Marker()
                m.header.frame_id = "map"
                m.header.stamp = self.get_clock().now().to_msg()
                m.ns = "cell_centers"
                m.id = marker_id
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.pose.position.x = center_x
                m.pose.position.y = center_y
                m.pose.position.z = 0.1
                m.scale.x = m.scale.y = m.scale.z = 0.25
                m.color.r, m.color.g, m.color.b = color
                m.color.a = 1.0
                marker_array.markers.append(m)

                marker_id += 1

        adjacency = defaultdict(list)
        for x,y in blue_cells:
            p = (round(x,3), round(y,3))
            for dx, dy in [(self.cell_size,0),(-self.cell_size,0),(0,self.cell_size),(0,-self.cell_size)]:
                nb = (round(x+dx,3), round(y+dy,3))
                if nb in blue_set:
                    adjacency[p].append(nb)

        # BFS Tree
        start = (0.0, 0.0)
        tree_edges = []
        if start:
            vis = {start}
            dq = deque([start])
            while dq:
                cur = dq.popleft()
                for nb in adjacency[cur]:
                    if nb not in vis:
                        vis.add(nb)
                        tree_edges.append((cur, nb))
                        dq.append(nb)

        # Publish TREE lines as orange LINE_LIST
        tree_marker = Marker()
        tree_marker.header.frame_id = "map"
        tree_marker.header.stamp = self.get_clock().now().to_msg()
        tree_marker.ns = "spanning_tree"
        tree_marker.id = 0
        tree_marker.type = Marker.LINE_LIST
        tree_marker.action = Marker.ADD
        tree_marker.scale.x = 0.05
        tree_marker.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0)  # orange
        for a,b in tree_edges:
            pt1 = Point(x=a[0], y=a[1], z=0.1)
            pt2 = Point(x=b[0], y=b[1], z=0.1)
            tree_marker.points.append(pt1)
            tree_marker.points.append(pt2)
        marker_array.markers.append(tree_marker)

        # Publish markers
        self.marker_pub.publish(marker_array)
        self.get_logger().info(f'Published {marker_id} cell markers anchored at (0,0)')

def main(args=None):
    rclpy.init(args=args)
    node = MapSubscriber()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
