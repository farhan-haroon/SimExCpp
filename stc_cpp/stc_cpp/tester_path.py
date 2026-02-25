import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from custom_interfaces.msg import TreeEdges
from collections import defaultdict

class PathPlanner(Node):
    def __init__(self):
        super().__init__('path_planner')
        self.path_publisher = self.create_publisher(
            Path, '/coverage_path', 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/diff_drive_controller/odom', self.odom_callback, 10
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10
        )
        self.tree_sub = self.create_subscription(
            TreeEdges, '/spanning_tree', self.tree_callback, 10
        )

        self.cell_size = 0.8
        self.odom = Odometry()
        self.map = OccupancyGrid()
        self.adjacency = defaultdict(list)

    def odom_callback(self, msg: Odometry):
        self.odom = msg

    def map_callback(self, msg: OccupancyGrid):
        self.map = msg

    def tree_callback(self, msg: TreeEdges):
        self.adjacency.clear()
        for start, end in zip(msg.start_points, msg.end_points):
            start_node = (start.x, start.y)
            end_node = (end.x, end.y)
            self.adjacency[start_node].append(end_node)
            self.adjacency[end_node].append(start_node)
    
        cur = (0.0, 0.0)  # Make it the blue node closest to robot pose at T = 0
        sub = prev = start = (cur[0] - self.cell_size / 4, cur[1] + self.cell_size / 4)
        print(f'Sub: {sub}, Start: {start}, Cur: {cur}, Prev: {prev}')
        path = []
        goal = (0.0, 0.0)
        dist = float('inf')

        nb = self.get_neighbours(cur)
        next_subs = defaultdict(list)

        for x, y in nb:
            node = (round(x, 3), round(y, 3))
            subnodes = self.get_subnodes(node)
            next_subs[node].extend(subnodes)

        for key in next_subs:
            closest_subnode = self.get_closest_subnode(sub, next_subs[key])
            if closest_subnode in path or closest_subnode == start:
                continue
            dx, dy = closest_subnode[0], closest_subnode[1]
            d = math.hypot(sub[0] - dx, sub[1] - dy)
            if d < dist:
                dist = d
                goal = closest_subnode
                goal_key = key

        midpoint = ((sub[0] + goal[0]) / 2, (sub[1] + goal[1]) / 2)
        if midpoint not in path:
            path.append(midpoint)
        if goal not in path:            
            path.append(goal)
        cur = goal_key
        sub = goal
        print(f'Sub: {sub}, Start: {start}, Cur: {cur}, Prev: {prev}')

        while (sub != start):

            print('Inside While Loop')
            print(f'Path: {path}')

            # Handle leaf node
            if len(self.adjacency[cur]) == 1:
                print(f'Leaf Node Detected: {cur}')
                if (sub[0] < cur[0] and sub[1] > cur[1] and prev[1] == sub[1]):
                    prev = sub
                    sub = (sub[0] + self.cell_size / 2, sub[1])
                    path.append((round(sub[0], 3), round(sub[1], 3)))
                    print('adding1', (round(sub[0], 3), round(sub[1], 3)))
                    prev = sub
                    sub = (sub[0], sub[1] - self.cell_size / 2)
                    path.append((round(sub[0], 3), round(sub[1], 3)))
                    print('adding', (round(sub[0], 3), round(sub[1], 3)))

                elif (sub[0] > cur[0] and sub[1] > cur[1] and prev[0] == sub[0]):
                    prev = sub
                    sub = (sub[0], sub[1] - self.cell_size / 2)
                    path.append((round(sub[0], 3), round(sub[1], 3)))
                    print('adding2', (round(sub[0], 3), round(sub[1], 3)))
                    prev = sub
                    sub = (sub[0] - self.cell_size / 2, sub[1])
                    path.append((round(sub[0], 3), round(sub[1], 3)))
                    print('adding', (round(sub[0], 3), round(sub[1], 3)))

                elif (sub[0] > cur[0] and sub[1] < cur[1] and prev[1] == sub[1]):
                    prev = sub
                    sub = (sub[0] - self.cell_size / 2, sub[1])
                    path.append((round(sub[0], 3), round(sub[1], 3)))
                    print('adding3', (round(sub[0], 3), round(sub[1], 3)))
                    prev = sub
                    sub = (sub[0], sub[1] + self.cell_size / 2)
                    path.append((round(sub[0], 3), round(sub[1], 3)))
                    print('adding', (round(sub[0], 3), round(sub[1], 3)))

                elif (sub[0] < cur[0] and sub[1] < cur[1] and prev[0] == sub[0]):
                    prev = sub
                    sub = (sub[0], sub[1] + self.cell_size / 2)
                    path.append((round(sub[0], 3), round(sub[1], 3)))
                    print('adding4', (round(sub[0], 3), round(sub[1], 3)))
                    prev = sub
                    sub = (sub[0] + self.cell_size / 2, sub[1])
                    path.append((round(sub[0], 3), round(sub[1], 3)))
                    print('adding', (round(sub[0], 3), round(sub[1], 3)))

                elif (sub[0] < cur[0] and sub[1] < cur[1] and prev[1] == sub[1]):
                    prev = sub
                    sub = (sub[0] + self.cell_size / 2, sub[1])
                    path.append((round(sub[0], 3), round(sub[1], 3)))
                    print('adding5', (round(sub[0], 3), round(sub[1], 3)))
                    prev = sub
                    sub = (sub[0], sub[1] + self.cell_size / 2)
                    path.append((round(sub[0], 3), round(sub[1], 3)))
                    print('adding', (round(sub[0], 3), round(sub[1], 3)))

                elif (sub[0] > cur[0] and sub[1] < cur[1] and prev[0] == sub[0]):
                    prev = sub
                    sub = (sub[0], sub[1] + self.cell_size / 2)
                    path.append((round(sub[0], 3), round(sub[1], 3)))
                    print('adding6', (round(sub[0], 3), round(sub[1], 3)))
                    prev = sub
                    sub = (sub[0] - self.cell_size / 2, sub[1])
                    path.append((round(sub[0], 3), round(sub[1], 3)))
                    print('adding', (round(sub[0], 3), round(sub[1], 3)))

                elif (sub[0] > cur[0] and sub[1] > cur[1] and prev[1] == sub[1]):
                    prev = sub
                    sub = (sub[0] - self.cell_size / 2, sub[1])
                    path.append((round(sub[0], 3), round(sub[1], 3)))
                    print('adding7', (round(sub[0], 3), round(sub[1], 3)))
                    prev = sub
                    sub = (sub[0], sub[1] - self.cell_size / 2)
                    path.append((round(sub[0], 3), round(sub[1], 3)))
                    print('adding', (round(sub[0], 3), round(sub[1], 3)))

                elif (sub[0] < cur[0] and sub[1] > cur[1] and prev[0] == sub[0]):
                    prev = sub
                    sub = (sub[0], sub[1] - self.cell_size / 2)
                    path.append((round(sub[0], 3), round(sub[1], 3)))
                    print('adding8', (round(sub[0], 3), round(sub[1], 3)))
                    prev = sub
                    sub = (sub[0] + self.cell_size / 2, sub[1])
                    path.append((round(sub[0], 3), round(sub[1], 3)))
                    print('adding', (round(sub[0], 3), round(sub[1], 3)))

                print(f'Sub: {sub}, Prev: {prev}, Cur: {cur}')

                nb = self.get_neighbours(cur)
                next_subs = defaultdict(list)

                for x, y in nb:
                    node = (round(x, 3), round(y, 3))
                    subnodes = self.get_subnodes(node)
                    next_subs[node].extend(subnodes)

                dist = float('inf')
                for key in next_subs:
                    closest_subnode = self.get_closest_subnode(sub, next_subs[key])
                    if closest_subnode in path:
                        continue
                    dx, dy = closest_subnode[0], closest_subnode[1]
                    d = math.hypot(sub[0] - dx, sub[1] - dy)
                    if d < dist:
                        dist = d
                        goal = closest_subnode
                        goal_key = key

                midpoint = ((sub[0] + goal[0]) / 2, (sub[1] + goal[1]) / 2)
                if midpoint not in path:
                    path.append(midpoint)
                if goal not in path:            
                    path.append(goal)
                cur = goal_key
                prev = sub
                sub = goal

                print(f'midpoint: {midpoint}, goal: {goal}')

                print(f'New Cur: {cur}, New Prev: {prev}, New Sub: {sub}')
                continue

            print(f'Current Node: {cur}, Previous SubNode: {prev}, Current Subnode: {sub}')
            
            nb = self.get_neighbours(cur)
            next_subs.clear()

            for x, y in nb:
                node = (round(x, 3), round(y, 3))
                subnodes = self.get_subnodes(node)
                next_subs[node].extend(subnodes)

            dist = float('inf')
            for key in next_subs:
                closest_subnode = self.get_closest_subnode(sub, next_subs[key])
                dx, dy = closest_subnode[0], closest_subnode[1]
                d = math.hypot(sub[0] - dx, sub[1] - dy)
                if d < dist:
                    dist = d
                    goal = closest_subnode
                    goal_key = key

            # Move by cell_size / 2 distance in prev -> cur direction
            dist = round(dist, 3)
            print(f'Checking distance: {dist} against cell size / 2: {self.cell_size / 2}')
            if dist != self.cell_size / 2:
                if prev[0] > cur[0] and prev[1] == cur[1]:
                    print('moving down by 1 unit')
                    target_sub = (sub[0] - self.cell_size / 2, sub[1])
                elif prev[0] < cur[0] and prev[1] == cur[1]:
                    print('moving up by 1 unit')
                    target_sub = (sub[0] + self.cell_size / 2, sub[1])
                elif prev[1] > cur[1] and prev[0] == cur[0]:
                    print('moving right by 1 unit')
                    target_sub = (sub[0], sub[1] - self.cell_size / 2)
                elif prev[1] < cur[1] and prev[0] == cur[0]:
                    print('moving left 1 unit')
                    target_sub = (sub[0], sub[1] + self.cell_size / 2)

                midpoint = ((sub[0] + target_sub[0]) / 2, (sub[1] + target_sub[1]) / 2)
                if midpoint not in path:
                    path.append(midpoint)
                if target_sub not in path:
                    path.append(target_sub)
                prev = sub
                sub = target_sub

                # Move to closest subnode

                nb = self.get_neighbours(cur)
                next_subs = defaultdict(list)

                for x, y in nb:
                    node = (round(x, 3), round(y, 3))
                    subnodes = self.get_subnodes(node)
                    next_subs[node].extend(subnodes)

                dist = float('inf')                
                for key in next_subs:
                    closest_subnode = self.get_closest_subnode(sub, next_subs[key])
                    if closest_subnode in path:
                        continue
                    dx, dy = closest_subnode[0], closest_subnode[1]
                    d = math.hypot(sub[0] - dx, sub[1] - dy)
                    if d < dist:
                        dist = d
                        goal = closest_subnode
                        goal_key = key

                midpoint = ((sub[0] + goal[0]) / 2, (sub[1] + goal[1]) / 2)
                if midpoint not in path:
                    path.append(midpoint)
                if goal not in path:
                    path.append(goal)
                cur = goal_key
                prev = sub
                sub = goal

                print(f'IF -> Current Node: {cur}, Previous SubNode: {prev}, Current Subnode: {sub}')
                
            else:
                # If we are already at the correct distance, just move to closest subnode
                nb = self.get_neighbours(cur)
                next_subs = defaultdict(list)

                for x, y in nb:
                    node = (round(x, 3), round(y, 3))
                    subnodes = self.get_subnodes(node)
                    next_subs[node].extend(subnodes)

                dist = float('inf')                                
                for key in next_subs:
                    closest_subnode = self.get_closest_subnode(sub, next_subs[key])
                    if closest_subnode in path:
                        continue
                    dx, dy = closest_subnode[0], closest_subnode[1]
                    d = math.hypot(sub[0] - dx, sub[1] - dy)
                    if d < dist:
                        dist = d
                        goal = closest_subnode
                        goal_key = key

                midpoint = ((sub[0] + goal[0]) / 2, (sub[1] + goal[1]) / 2)
                if midpoint not in path:
                    path.append(midpoint)
                if goal not in path:
                    path.append(goal)
                cur = goal_key
                prev = sub
                sub = goal

                print(f'midpoint: {midpoint}, goal: {goal}')

                print(f'ELSE -> Current Node: {cur}, Previous SubNode: {prev}, Current Subnode: {sub}')

        self.get_logger().info('Out of While Loop')
        
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for (x, y) in path:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.1
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.path_publisher.publish(path_msg)
        
        self.get_logger().info('Path planning completed')
                       

    def get_neighbours(self, node: tuple) -> list:
        neighbours = self.adjacency[node]
        if not neighbours:
            self.get_logger().warn(f'No neighbours found for node {node}')
            return
        neighbours = [(round(x, 3), round(y, 3)) for x, y in neighbours]
        return neighbours
    
    def get_subnodes(self, node: tuple) -> list:
        sub_nodes = []
        for dx, dy in [(self.cell_size / 4, self.cell_size / 4), (-self.cell_size / 4, -self.cell_size / 4), (-self.cell_size / 4, self.cell_size / 4), (self.cell_size / 4, -self.cell_size / 4)]:
            sub_node = (round(node[0] + dx, 3), round(node[1] + dy, 3))
            sub_nodes.append(sub_node)
        return sub_nodes
    
    def get_closest_subnode(self, sub: tuple, nb_subs: list) -> tuple:
        closest = None
        min_dist = float('inf')
        for subnode in nb_subs:
            dist = math.hypot(sub[0] - subnode[0], sub[1] - subnode[1])
            if dist < min_dist:
                min_dist = dist
                closest = subnode
        return closest   


def main(args=None):
    rclpy.init(args=args)
    node = PathPlanner()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
