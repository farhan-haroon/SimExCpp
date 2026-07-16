#!/usr/bin/env python3

import time
import xml.etree.ElementTree as ET

import rclpy
from rclpy.executors import MultiThreadedExecutor
from threading import Thread

from moveit_msgs.action import MoveGroup as MoveGroupAction
from moveit_msgs.msg import (
    MotionPlanRequest,
    Constraints,
    JointConstraint,
    PlanningOptions,
    MoveItErrorCodes
)

from rclpy.action import ActionClient
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory


PLANNING_GROUP = "ur10_ur_manipulator"


def parse_named_states(srdf_path):

    root = ET.parse(srdf_path).getroot()

    states = {}

    for gs in root.findall("group_state"):

        name = gs.get("name")

        states[name] = {
            j.get("name"): float(j.get("value"))
            for j in gs.findall("joint")
        }

    return states


class MoveitInspector(Node):

    def __init__(self):

        super().__init__("moveit_inspector")

        self._ac = ActionClient(
            self,
            MoveGroupAction,
            "/move_action"
        )

        srdf_path = (
            get_package_share_directory(
                "husky_ur10_moveit_config"
            )
            + "/srdf/ur.srdf"
        )

        self.named_states = parse_named_states(srdf_path)

    def move_to(self, label, joint_values):

        if not self._ac.wait_for_server(timeout_sec=10.0):

            self.get_logger().error(
                "MoveIt action server not available"
            )

            return False

        jcs = []

        for name, pos in joint_values.items():

            jc = JointConstraint()

            jc.joint_name = name
            jc.position = pos
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0

            jcs.append(jc)

        req = MotionPlanRequest()

        req.group_name = PLANNING_GROUP
        req.num_planning_attempts = 10
        req.allowed_planning_time = 10.0
        req.goal_constraints = [
            Constraints(
                joint_constraints=jcs
            )
        ]

        opts = PlanningOptions()
        opts.plan_only = False

        goal = MoveGroupAction.Goal()

        goal.request = req
        goal.planning_options = opts

        future = self._ac.send_goal_async(goal)

        rclpy.spin_until_future_complete(
            self,
            future
        )

        goal_handle = future.result()

        if goal_handle is None:
            return False

        result_future = goal_handle.get_result_async()

        rclpy.spin_until_future_complete(
            self,
            result_future
        )

        result = result_future.result()

        if result is None:
            return False

        return (
            result.result.error_code.val
            == MoveItErrorCodes.SUCCESS
        )


class ObjectInspector:

    def __init__(self):

        self.node = MoveitInspector()

        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)

        self.thread = Thread(
            target=self.executor.spin,
            daemon=True
        )

        self.thread.start()

    def inspect(self):

        viewpoints = [
            "front_low",
            "front_right_low",
            "right_low",
            "right_rear_low",
            "rear_low",
            "rear_left_low",
            "left_low",
            "left_front_low"
        ]

        for pose in viewpoints:

            if pose not in self.node.named_states:

                print(
                    f"{pose} not found in SRDF"
                )

                continue

            print(
                f"\nMoving to {pose}"
            )

            success = self.node.move_to(
                pose,
                self.node.named_states[pose]
            )

            if not success:

                print(
                    f"Failed to reach {pose}"
                )

                continue

            time.sleep(1.0)

            print(
                f"Captured image at {pose}"
            )

        success = self.node.move_to(
            "front_low",
            self.node.named_states["front_low"]
        )

        print(
            "Inspection complete"
        )