#!/usr/bin/env python3
"""Test whether IK can be solved for a UR10 end-effector pose expressed in a
robot-fixed frame (e.g. base_link, base_footprint) via MoveIt's /compute_ik
service, and optionally execute it via /move_action.

Requires move_group to be running (e.g. via husky_ur10_moveit_office_large.launch.py).

Manual orientation (roll/pitch/yaw of --tip-link in --frame):
    ros2 run husky_ur10_cam test_ee_pose_ik.py \
        --frame base_link --xyz 0.6 0.0 0.9 --rpy 0 1.57 0

Camera-safe usage: instead of guessing an rpy that happens to keep the
camera level, solve for camera_optical_link directly and give a gaze
direction with --look-dir - the orientation is then computed so the image
is never rolled/tilted/upside-down (REP-103: optical Z = forward,
Y = down, X = right), only the gaze direction changes:
    ros2 run husky_ur10_cam test_ee_pose_ik.py \
        --tip-link camera_optical_link --frame base_link \
        --xyz 0.6 0.0 1.2 --look-dir 1 0 0 --execute
"""
import argparse
import math
import sys

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup as MoveGroupAction
from moveit_msgs.msg import (
    PositionIKRequest,
    MoveItErrorCodes,
    MotionPlanRequest,
    Constraints,
    JointConstraint,
    PlanningOptions,
)
from moveit_msgs.srv import GetPositionIK

PLANNING_GROUP = "ur10_ur_manipulator"
IK_TIP_LINK = "ur10_tool0"
MOVE_TIMEOUT = 60.0
ALLOWED_PLANNING_TIME = 15.0

ERROR_CODE_NAMES = {
    getattr(MoveItErrorCodes, name): name
    for name in dir(MoveItErrorCodes)
    if name.isupper() and isinstance(getattr(MoveItErrorCodes, name), int)
}


def quat_from_rpy(roll, pitch, yaw):
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _normalize(v):
    n = math.sqrt(sum(c * c for c in v))
    if n < 1e-9:
        raise ValueError("zero-length vector")
    return tuple(c / n for c in v)


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _mat3_to_quat(r):
    (m00, m01, m02), (m10, m11, m12), (m20, m21, m22) = r
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        return ((m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s, 0.25 / s)
    if m00 > m11 and m00 > m22:
        s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        return (0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s)
    if m11 > m22:
        s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        return ((m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s)
    s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
    return ((m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s)


def look_at_quaternion(forward, up=(0.0, 0.0, 1.0)):
    """Orientation for a REP-103 optical frame (X=right, Y=down, Z=forward)
    that gazes along `forward` with zero roll relative to `up` - i.e. the
    image horizon stays level, never tilted or upside-down, no matter which
    direction it looks."""
    f = _normalize(forward)
    u = _normalize(up)
    if abs(_dot(f, u)) > 0.999:
        # forward (nearly) parallel to the up reference: avoid singularity
        u = (1.0, 0.0, 0.0) if abs(f[0]) < 0.9 else (0.0, 1.0, 0.0)
    right = _normalize(_cross(f, u))
    down = _cross(f, right)
    r = (
        (right[0], down[0], f[0]),
        (right[1], down[1], f[1]),
        (right[2], down[2], f[2]),
    )
    return _mat3_to_quat(r)


class EEPoseIKTester(Node):
    def __init__(self):
        super().__init__("test_ee_pose_ik")
        self.cli = self.create_client(GetPositionIK, "/compute_ik")
        self._move_ac = ActionClient(self, MoveGroupAction, "/move_action")

    def check(self, frame_id, xyz, quat, group, tip_link, timeout_sec, attempts):
        if not self.cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().error(
                "/compute_ik service not available - is move_group running?"
            )
            return False, None

        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = xyz
        qx, qy, qz, qw = quat
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        for attempt in range(1, attempts + 1):
            req = GetPositionIK.Request()
            req.ik_request = PositionIKRequest()
            req.ik_request.group_name = group
            req.ik_request.ik_link_name = tip_link
            req.ik_request.pose_stamped = pose
            req.ik_request.avoid_collisions = True
            req.ik_request.timeout.sec = int(timeout_sec)
            req.ik_request.timeout.nanosec = int((timeout_sec % 1) * 1e9)

            future = self.cli.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            result = future.result()

            if result is None:
                self.get_logger().error(f"Attempt {attempt}: service call failed")
                continue

            code = result.error_code.val
            name = ERROR_CODE_NAMES.get(code, str(code))

            if code == MoveItErrorCodes.SUCCESS:
                self.get_logger().info(
                    f"Attempt {attempt}: IK SUCCESS in frame '{frame_id}' for "
                    f"pose xyz={xyz} quat={tuple(round(c, 4) for c in quat)}"
                )
                all_joints = dict(
                    zip(result.solution.joint_state.name,
                        result.solution.joint_state.position)
                )
                arm_joints = {
                    j: v for j, v in all_joints.items() if j.startswith("ur10_")
                }
                for j, v in arm_joints.items():
                    self.get_logger().info(f"    {j}: {v:.4f}")
                return True, arm_joints

            self.get_logger().warn(f"Attempt {attempt}: IK FAILED - {name} ({code})")

        return False, None

    def execute(self, joint_positions, group, vel_scale, accel_scale):
        if not self._move_ac.wait_for_server(timeout_sec=30.0):
            self.get_logger().error("/move_action not available")
            return False

        constraints = []
        for joint_name, position in joint_positions.items():
            jc = JointConstraint()
            jc.joint_name = joint_name
            jc.position = position
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.append(jc)

        request = MotionPlanRequest()
        request.group_name = group
        request.num_planning_attempts = 10
        request.allowed_planning_time = ALLOWED_PLANNING_TIME
        request.max_velocity_scaling_factor = vel_scale
        request.max_acceleration_scaling_factor = accel_scale
        request.goal_constraints = [Constraints(joint_constraints=constraints)]

        planning_options = PlanningOptions()
        planning_options.plan_only = False
        planning_options.replan = True
        planning_options.replan_attempts = 3

        goal = MoveGroupAction.Goal()
        goal.request = request
        goal.planning_options = planning_options

        send_future = self._move_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=MOVE_TIMEOUT)
        goal_handle = send_future.result()

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Move goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=MOVE_TIMEOUT)
        result = result_future.result()

        if result is None:
            self.get_logger().error("Move goal timed out")
            return False

        code = result.result.error_code.val
        if code == MoveItErrorCodes.SUCCESS:
            self.get_logger().info("Execution SUCCESS - robot moved to target pose")
            return True

        self.get_logger().error(f"Execution FAILED (error={code})")
        return False


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frame", default="base_link",
                    help="Frame the target pose is expressed in "
                         "(e.g. base_link, base_footprint)")
    p.add_argument("--xyz", nargs=3, type=float, default=[0.6, 0.0, 0.9],
                    metavar=("X", "Y", "Z"))
    p.add_argument("--rpy", nargs=3, type=float, default=None,
                    metavar=("R", "P", "Y"),
                    help="Manual roll/pitch/yaw of --tip-link. Ignored if "
                         "--look-dir is given.")
    p.add_argument("--look-dir", nargs=3, type=float, default=None,
                    metavar=("DX", "DY", "DZ"),
                    help="Gaze direction (in --frame) for --tip-link, "
                         "computed level/roll-free (see module docstring). "
                         "Recommended over --rpy when --tip-link is "
                         "camera_optical_link. Overrides --rpy.")
    p.add_argument("--up", nargs=3, type=float, default=[0.0, 0.0, 1.0],
                    metavar=("UX", "UY", "UZ"),
                    help="World 'up' reference used by --look-dir (default: +Z)")
    p.add_argument("--group", default=PLANNING_GROUP)
    p.add_argument("--tip-link", default=IK_TIP_LINK)
    p.add_argument("--timeout", type=float, default=0.1)
    p.add_argument("--attempts", type=int, default=5)
    p.add_argument("--execute", action="store_true",
                    help="Plan and execute the IK solution via /move_action "
                         "(actually moves the robot)")
    p.add_argument("--vel-scale", type=float, default=0.5)
    p.add_argument("--accel-scale", type=float, default=0.5)
    return p.parse_args(argv)


def main(args=None):
    rclpy.init(args=args)
    argv = rclpy.utilities.remove_ros_args(args=sys.argv)[1:]
    opts = parse_args(argv)

    if opts.look_dir is not None:
        try:
            quat = look_at_quaternion(tuple(opts.look_dir), tuple(opts.up))
        except ValueError:
            print(f"ERROR: --look-dir {opts.look_dir} is a zero vector - "
                  f"give an actual gaze direction, e.g. 1 0 0")
            rclpy.shutdown()
            sys.exit(2)
    else:
        quat = quat_from_rpy(*(opts.rpy if opts.rpy is not None else [0.0, 1.57, 0.0]))

    node = EEPoseIKTester()
    ok, joints = node.check(
        opts.frame, tuple(opts.xyz), quat,
        opts.group, opts.tip_link, opts.timeout, opts.attempts,
    )

    if ok and opts.execute:
        ok = node.execute(joints, opts.group, opts.vel_scale, opts.accel_scale)

    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
