import math
from typing import Optional, List, Tuple
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Path, Odometry
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
        self.tree_sub = self.create_subscription(
            TreeEdges, '/spanning_tree', self.tree_callback, 10
        )

        self.cell_size = 0.8
        self.odom = Odometry()
        self.adjacency = defaultdict(list)
        self.start = (-self.cell_size / 4, self.cell_size / 4)
        self.path = []

    def odom_callback(self, msg: Odometry):
        self.odom = msg

    def tree_callback(self, msg: TreeEdges):
        self.adjacency.clear()
        for start, end in zip(msg.start_points, msg.end_points):
            start_node = (start.x, start.y)
            end_node = (end.x, end.y)
            self.adjacency[start_node].append(end_node)
            self.adjacency[end_node].append(start_node)

        maj_cur = (0.0, 0.0)  # Make it the blue node closest to robot pose at T = 0
        sub_cur = sub_prev = self.start
        self.path.append(sub_cur)
        obj = self.get_closest_subnode(sub_cur, maj_cur)
        goal_sub, goal_cur = obj[0], obj[1]
        if goal_sub is None:
            self.get_logger().info('No subnodes found, exiting path planning')
            return
        midpoint = (round((goal_sub[0] + sub_cur[0]) / 2, 3), round((goal_sub[1] + sub_cur[1]) / 2, 3))
        print(f'Adding midpoint: {midpoint} and goal_sub: {goal_sub} to path')
        self.path.append(midpoint)
        self.path.append(goal_sub)
        sub_prev = sub_cur
        sub_cur = goal_sub
        maj_cur = goal_cur

        while True:
            print()
            print(f'Path: {self.path}')
            
            if len(self.adjacency[maj_cur]) == 1:
                self.get_logger().info(f'Leaf Node: {maj_cur}')
                if (sub_cur[0] < maj_cur[0] and sub_cur[1] > maj_cur[1] and sub_prev[1] == sub_cur[1]):
                    sub_prev = sub_cur
                    sub_cur = (round(sub_cur[0] + self.cell_size / 2, 3), round(sub_cur[1], 3))
                    self.path.append(sub_cur)
                    print(f'adding {sub_cur} to path')
                    sub_prev = sub_cur
                    sub_cur = (round(sub_cur[0], 3), round(sub_cur[1] - self.cell_size / 2, 3))
                    self.path.append(sub_cur)
                    print(f'adding {sub_cur} to path')

                elif (sub_cur[0] > maj_cur[0] and sub_cur[1] > maj_cur[1] and sub_prev[0] == sub_cur[0]):
                    sub_prev = sub_cur
                    sub_cur = (round(sub_cur[0], 3), round(sub_cur[1] - self.cell_size / 2, 3))
                    self.path.append(sub_cur)
                    print(f'adding {sub_cur} to path')
                    sub_prev = sub_cur
                    sub_cur = (round(sub_cur[0] - self.cell_size / 2, 3), round(sub_cur[1], 3))
                    self.path.append(sub_cur)
                    print(f'adding {sub_cur} to path')

                elif (sub_cur[0] > maj_cur[0] and sub_cur[1] < maj_cur[1] and sub_prev[1] == sub_cur[1]):
                    sub_prev = sub_cur
                    sub_cur = (round(sub_cur[0] - self.cell_size / 2, 3), round(sub_cur[1], 3))
                    self.path.append(sub_cur)
                    print(f'adding {sub_cur} to path')
                    sub_prev = sub_cur
                    sub_cur = (round(sub_cur[0], 3), round(sub_cur[1] + self.cell_size / 2,3))
                    self.path.append(sub_cur)
                    print(f'adding {sub_cur} to path')

                elif (sub_cur[0] < maj_cur[0] and sub_cur[1] < maj_cur[1] and sub_prev[0] == sub_cur[0]):
                    sub_prev = sub_cur
                    sub_cur = (round(sub_cur[0], 3), round(sub_cur[1] + self.cell_size / 2, 3))
                    self.path.append(sub_cur)
                    print(f'adding {sub_cur} to path')
                    sub_prev = sub_cur
                    sub_cur = (round(sub_cur[0] + self.cell_size / 2, 3), round(sub_cur[1], 3))
                    self.path.append(sub_cur)
                    print(f'adding {sub_cur} to path')

                elif (sub_cur[0] < maj_cur[0] and sub_cur[1] < maj_cur[1] and sub_prev[1] == sub_cur[1]):
                    sub_prev = sub_cur
                    sub_cur = (round(sub_cur[0] + self.cell_size / 2, 3), round(sub_cur[1], 3))
                    self.path.append(sub_cur)
                    print(f'adding {sub_cur} to path')
                    sub_prev = sub_cur
                    sub_cur = (round(sub_cur[0], 3), round(sub_cur[1] + self.cell_size / 2, 3))
                    self.path.append(sub_cur)
                    print(f'adding {sub_cur} to path')

                elif (sub_cur[0] > maj_cur[0] and sub_cur[1] < maj_cur[1] and sub_prev[0] == sub_cur[0]):
                    sub_prev = sub_cur
                    sub_cur = (round(sub_cur[0], 3), round(sub_cur[1] + self.cell_size / 2, 3))
                    self.path.append(sub_cur)
                    print(f'adding {sub_cur} to path')
                    sub_prev = sub_cur
                    sub_cur = (round(sub_cur[0] - self.cell_size / 2, 3), round(sub_cur[1], 3))
                    self.path.append(sub_cur)
                    print(f'adding {sub_cur} to path')

                elif (sub_cur[0] > maj_cur[0] and sub_cur[1] > maj_cur[1] and sub_prev[1] == sub_cur[1]):
                    sub_prev = sub_cur
                    sub_cur = (round(sub_cur[0] - self.cell_size / 2, 3), round(sub_cur[1], 3))
                    self.path.append(sub_cur)
                    print(f'adding {sub_cur} to path')
                    sub_prev = sub_cur
                    sub_cur = (round(sub_cur[0], 3), round(sub_cur[1] - self.cell_size / 2, 3))
                    self.path.append(sub_cur)
                    print(f'adding {sub_cur} to path')

                elif (sub_cur[0] < maj_cur[0] and sub_cur[1] > maj_cur[1] and sub_prev[0] == sub_cur[0]):
                    sub_prev = sub_cur
                    sub_cur = (round(sub_cur[0], 3), round(sub_cur[1] - self.cell_size / 2, 3))
                    self.path.append(sub_cur)
                    print(f'adding {sub_cur} to path')
                    sub_prev = sub_cur
                    sub_cur = (round(sub_cur[0] + self.cell_size / 2, 3), round(sub_cur[1], 3))
                    self.path.append(sub_cur)
                    print(f'adding {sub_cur} to path')

                print(f'Leaf Node: {maj_cur}, Subnode: {sub_cur}, Previous Subnode: {sub_prev}')

                obj = self.get_closest_subnode(sub_cur, maj_cur)
                goal_sub, goal_cur = obj[0], obj[1]
                if goal_sub is None:
                    self.get_logger().info('Break Point 2')
                    break
                midpoint = (round((goal_sub[0] + sub_cur[0]) / 2, 3), round((goal_sub[1] + sub_cur[1]) / 2, 3))
                print(f'(AFTER LEAFNODE) Adding midpoint: {midpoint} and goal_sub: {goal_sub} to path')
                self.path.append(midpoint)
                self.path.append(goal_sub)
                sub_prev = sub_cur
                sub_cur = goal_sub
                maj_cur = goal_cur
                if maj_cur == self.start:
                    print(f'Reached Start Node: {self.start}, Ending Path Planning')
                    break
                print(f'(FINALLY) Current Subnode: {sub_cur}, Previous Subnode: {sub_prev}, Current Major Node: {maj_cur}')
                continue



            # If next subnode distance is not cell_size/2, then we add a jump
            
            obj = self.get_closest_subnode(sub_cur, maj_cur)
            goal_sub, goal_cur = obj[0], obj[1]
            if goal_sub is None:
                self.get_logger().info('Break Point 3')
                break

            d = math.hypot(goal_sub[0] - sub_cur[0], goal_sub[1] - sub_cur[1])
            if round(d, 3) != self.cell_size / 2:
                if sub_prev[0] > sub_cur[0] and sub_prev[1] == sub_cur[1]:
                    print('moving down by 1 unit')
                    target_sub = (sub_cur[0] - self.cell_size / 2, sub_cur[1])
                elif sub_prev[0] < sub_cur[0] and sub_prev[1] == sub_cur[1]:
                    print('moving up by 1 unit')
                    target_sub = (sub_cur[0] + self.cell_size / 2, sub_cur[1])
                elif sub_prev[1] > sub_cur[1] and sub_prev[0] == sub_cur[0]:
                    print('moving right by 1 unit')
                    target_sub = (sub_cur[0], sub_cur[1] - self.cell_size / 2)
                elif sub_prev[1] < sub_cur[1] and sub_prev[0] == sub_cur[0]:
                    print('moving left 1 unit')
                    target_sub = (sub_cur[0], sub_cur[1] + self.cell_size / 2)
                
                midpoint = (round((target_sub[0] + sub_cur[0]) / 2, 3), round((target_sub[1] + sub_cur[1]) / 2, 3))
                self.path.append(midpoint)
                self.path.append(target_sub)
                sub_prev = sub_cur
                sub_cur = target_sub
                print(f'After jumping L/R/U/D - Current Subnode: {sub_cur}, Previous Subnode: {sub_prev}, Current Major Node: {maj_cur}')
                if maj_cur == self.start:
                    print(f'Reached Start Node: {self.start}, Ending Path Planning')
                    break

                # After jumping, move to next closest subnode                

                obj = self.get_closest_subnode(sub_cur, maj_cur)
                goal_sub, goal_cur = obj[0], obj[1]
                if goal_sub is None:
                    self.get_logger().info('Break Point 4')
                    break
                midpoint = (round((goal_sub[0] + sub_cur[0]) / 2, 3), round((goal_sub[1] + sub_cur[1]) / 2, 3))
                self.path.append(midpoint)
                self.path.append(goal_sub)
                sub_prev = sub_cur
                sub_cur = goal_sub
                maj_cur = goal_cur
                if maj_cur == self.start:
                    print(f'Reached Start Node: {self.start}, Ending Path Planning')
                    break

            # If next subnode distance is cell_size/2, move to next subnode directly
            else:
                obj = self.get_closest_subnode(sub_cur, maj_cur)
                goal_sub, goal_cur = obj[0], obj[1]
                if goal_sub is None:
                    self.get_logger().info('Break Point 5')
                    break
                midpoint = (round((goal_sub[0] + sub_cur[0]) / 2, 3), round((goal_sub[1] + sub_cur[1]) / 2, 3))
                self.path.append(midpoint)
                self.path.append(goal_sub)
                sub_prev = sub_cur
                sub_cur = goal_sub
                maj_cur = goal_cur
                if maj_cur == self.start:
                    print(f'Reached Start Node: {self.start}, Ending Path Planning')
                    break

            print(f'(FINALLY) Current Subnode: {sub_cur}, Previous Subnode: {sub_prev}, Current Major Node: {maj_cur}')

        self.get_logger().info('Out of While Loop')
        
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for (x, y) in self.path:
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
        self.path.clear()

    def get_closest_subnode(self, sub_node: tuple, major_node: tuple) -> Optional[List[Tuple[float, float]]]:
        closest_subnode = closest_majornode = target = None
        nb = self.get_neighbours(major_node)
        print(f'Neighbours of: {major_node}: {nb}')
        distance = float('inf')
        for i in nb:
            subnodes = self.get_subnodes(i)
            # print(f'Subnodes for {i}: {subnodes}')
            dist = float('inf')
            for sn in subnodes:
                if sn in self.path:
                    print(f'{sn} in path')
                    continue
                d = math.hypot(sub_node[0] - sn[0], sub_node[1] - sn[1])
                if d < dist:
                    dist = d
                    target = sn
            if target and dist < distance:
                distance = dist
                closest_subnode = target
                closest_majornode = i
        print(f'Closest Subnode: {closest_subnode}, Closest Majornode: {closest_majornode}')
        return [closest_subnode, closest_majornode]


    def get_neighbours(self, node: tuple) -> list:
        neighbours = self.adjacency[node]
        if not neighbours:
            self.get_logger().error(f'No neighbours found for {node}')
            return
        return neighbours

    def get_subnodes(self, node: tuple) -> list:
        sub_nodes = []
        for dx, dy in [(self.cell_size / 4, self.cell_size / 4), (-self.cell_size / 4, -self.cell_size / 4), (-self.cell_size / 4, self.cell_size / 4), (self.cell_size / 4, -self.cell_size / 4)]:
            sub_node = (round(node[0] + dx, 3), round(node[1] + dy, 3))
            sub_nodes.append(sub_node)
        return sub_nodes


def main(args=None):
    rclpy.init(args=args)
    node = PathPlanner()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()        