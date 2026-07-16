#!/usr/bin/env python3

import time
from threading import Event, Thread

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor

from moveit_msgs.action import MoveGroup as MoveGroupAction
from moveit_msgs.msg import (
    MotionPlanRequest,
    Constraints,
    JointConstraint,
    PlanningOptions,
    MoveItErrorCodes,
)


POSITIONS = [
    "front_low",
    "right_low",
    "rear_low",
    "left_low",
    "front_high",
    "right_high",
    "rear_high",
    "left_high",
]

PLANNING_GROUP = "ur10_ur_manipulator"

MOVE_TIMEOUT = 60.0
ALLOWED_PLANNING_TIME = 15.0

NAMED_STATES = {

    "front_low": {
        "ur10_elbow_joint": 2.30,
        "ur10_shoulder_lift_joint": -1.83,
        "ur10_shoulder_pan_joint": 0.0,
        "ur10_wrist_1_joint": -3.58,
        "ur10_wrist_2_joint": -1.57,
        "ur10_wrist_3_joint": 0.0,
    },

    "right_low": {
        "ur10_elbow_joint": 2.50,
        "ur10_shoulder_lift_joint": -2.23,
        "ur10_shoulder_pan_joint": -1.57,
        "ur10_wrist_1_joint": -3.42,
        "ur10_wrist_2_joint": -1.57,
        "ur10_wrist_3_joint": 0.0,
    },

    "rear_low": {
        "ur10_elbow_joint": 2.37,
        "ur10_shoulder_lift_joint": -1.95,
        "ur10_shoulder_pan_joint": -3.14,
        "ur10_wrist_1_joint": -3.58,
        "ur10_wrist_2_joint": -1.57,
        "ur10_wrist_3_joint": 0.0,
    },

    "left_low": {
        "ur10_elbow_joint": 2.46,
        "ur10_shoulder_lift_joint": -2.13,
        "ur10_shoulder_pan_joint": -4.71,
        "ur10_wrist_1_joint": -3.49,
        "ur10_wrist_2_joint": -1.57,
        "ur10_wrist_3_joint": 0.0,
    },

    "front_high": {
        "ur10_elbow_joint": 1.40,
        "ur10_shoulder_lift_joint": -1.62,
        "ur10_shoulder_pan_joint": 1.54,
        "ur10_wrist_1_joint": -1.20,
        "ur10_wrist_2_joint": -1.60,
        "ur10_wrist_3_joint": -0.11,
    },

    "right_high": {
        "ur10_elbow_joint": 1.40,
        "ur10_shoulder_lift_joint": -1.62,
        "ur10_shoulder_pan_joint": 1.54,
        "ur10_wrist_1_joint": -1.20,
        "ur10_wrist_2_joint": -1.60,
        "ur10_wrist_3_joint": -0.11,
    },

    "rear_high": {
        "ur10_elbow_joint": 1.40,
        "ur10_shoulder_lift_joint": -1.62,
        "ur10_shoulder_pan_joint": 1.54,
        "ur10_wrist_1_joint": -1.20,
        "ur10_wrist_2_joint": -1.60,
        "ur10_wrist_3_joint": -0.11,
    },

    "left_high": {
        "ur10_elbow_joint": 1.40,
        "ur10_shoulder_lift_joint": -1.62,
        "ur10_shoulder_pan_joint": 1.54,
        "ur10_wrist_1_joint": -1.20,
        "ur10_wrist_2_joint": -1.60,
        "ur10_wrist_3_joint": -0.11,
    },
}


class PresetViewsNode(Node):

    def __init__(self):

        super().__init__(
            "preset_views",
            parameter_overrides=[
                rclpy.parameter.Parameter(
                    "use_sim_time",
                    rclpy.parameter.Parameter.Type.BOOL,
                    True,
                )
            ],
        )

        self._ac = ActionClient(
            self,
            MoveGroupAction,
            "/move_action"
        )

    def move_to(self, label, joint_values):

        done = Event()
        success = [False]

        def result_cb(future):

            code = future.result().result.error_code.val

            success[0] = (
                code == MoveItErrorCodes.SUCCESS
            )

            if success[0]:
                self.get_logger().info(
                    f"Reached {label}"
                )
            else:
                self.get_logger().error(
                    f"Move failed for {label} (error={code})"
                )

            done.set()

        def goal_cb(future):

            goal_handle = future.result()

            if goal_handle is None or not goal_handle.accepted:

                self.get_logger().error(
                    f"Goal rejected: {label}"
                )

                done.set()
                return

            goal_handle.get_result_async().add_done_callback(
                result_cb
            )

        constraints = []

        for joint_name, position in joint_values.items():

            jc = JointConstraint()

            jc.joint_name = joint_name
            jc.position = position
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0

            constraints.append(jc)

        request = MotionPlanRequest()

        request.group_name = PLANNING_GROUP
        request.num_planning_attempts = 10
        request.allowed_planning_time = ALLOWED_PLANNING_TIME
        request.max_velocity_scaling_factor = 1.0
        request.max_acceleration_scaling_factor = 1.0

        request.goal_constraints = [
            Constraints(
                joint_constraints=constraints
            )
        ]

        planning_options = PlanningOptions()
        planning_options.plan_only = False
        planning_options.replan = True
        planning_options.replan_attempts = 3

        goal = MoveGroupAction.Goal()
        goal.request = request
        goal.planning_options = planning_options

        self._ac.send_goal_async(goal).add_done_callback(
            goal_cb
        )

        done.wait(MOVE_TIMEOUT)

        return success[0]


def main():

    rclpy.init()

    node = PresetViewsNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    spin_thread = Thread(
        target=executor.spin
    )
    spin_thread.start()

    print("Waiting for MoveIt action server...")

    if not node._ac.wait_for_server(timeout_sec=30.0):

        print("ERROR: /move_action not available")

        executor.shutdown()
        spin_thread.join()

        node.destroy_node()
        rclpy.shutdown()

        return

    for position in POSITIONS:

        print(f"\nMoving to {position}")

        success = node.move_to(
            position,
            NAMED_STATES[position]
        )

        if success:
            time.sleep(1.0)

    print("\nFinished")

    executor.shutdown()
    spin_thread.join()

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
