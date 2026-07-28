#!/usr/bin/env python3
"""ROS 2 action server wrapping the two-phase object search
(ee_camera_environment_scan.py's run_full_search) as
custom_interfaces/action/FindObject.

Intended caller: the STC coverage node (stc_cpp/stc_cpp/stc.py), which
sends a goal once the mobile base arrives at a new major cell. Also
works standalone for manual testing.

Goal:     target_object (YOLO/COCO class name), max_object_distance
          (optional override; <= 0 uses the script's default).
Feedback: phase ("phase1"/"phase2"), status string, step count -
          passed through from run_phase1_helix/run_phase2_vlm.
Result:   success, found_objects[] (DetectedObject), message.

execute_callback is a long-running blocking call that internally spins
the node via nested rclpy.spin_once()/spin_until_future_complete() (IK,
MoveGroup, image/points capture). It runs in its own callback group,
separate from the default group EnvironmentScanner's subscriptions/TF/
clients use, so those nested waits can still be dispatched against
without a MultiThreadedExecutor. One goal at a time; no cancellation.
"""
import os
import sys
import time

import rclpy
from rclpy.action import ActionServer
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException

# ee_camera_environment_scan.py is a sibling script, not an installed
# package - add its directory to sys.path so it can be imported directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ee_camera_environment_scan import (  # noqa: E402
    DEFAULT_DETECTIONS_DIR,
    EnvironmentScanner,
    install_shutdown_handler,
    parse_args,
    run_full_search,
)

from custom_interfaces.action import FindObject  # noqa: E402
from custom_interfaces.msg import DetectedObject  # noqa: E402

# Every search-tuning knob, overridable via ROS parameters (see
# husky_ur10_cam/config/all_params.yaml, the single place to edit them -
# main() loads it automatically). Not included: image_topic/points_topic
# (fixed at construction, before parameters can be read) and
# yolo_weights/save_detections_dir/qwen_model (machine-specific paths,
# left to their code-level defaults unless explicitly added to the yaml).
TUNABLE_PARAMS = [
    "helix_radius", "helix_height", "tilt_up_deg", "tilt_down_deg",
    "num_sectors", "tilts_per_sector", "helix_center", "objects",
    "max_tries", "revisit_attempts", "phase2_max_steps",
    "max_object_distance", "approach_standoff", "approach_max_step",
    "skip_phase1", "skip_phase2", "skip_start_home", "skip_return_home",
    "dwell", "group", "vel_scale", "accel_scale",
]


class ObjectSearchActionServer(EnvironmentScanner):
    def __init__(self):
        super().__init__(image_topic="/ee_camera/image_raw",
                          node_name="object_search_action_server")
        # parse_args([]) gives the script's own code-level defaults - used
        # here only as the ROS parameter declaration defaults, so a fresh
        # checkout with no yaml/override still behaves exactly like the CLI.
        defaults = parse_args([])
        for name in TUNABLE_PARAMS:
            self.declare_parameter(name, getattr(defaults, name))

        self._action_group = MutuallyExclusiveCallbackGroup()
        self._action_server = ActionServer(
            self, FindObject, "find_object",
            execute_callback=self.execute_callback,
            callback_group=self._action_group,
        )

    def execute_callback(self, goal_handle):
        goal = goal_handle.request
        opts = parse_args([])  # fills in the fields TUNABLE_PARAMS doesn't cover
        for name in TUNABLE_PARAMS:
            setattr(opts, name, self.get_parameter(name).value)
        opts.objects = list(goal.target_objects) if goal.target_objects else list(opts.objects)
        if goal.max_object_distance > 0.0:
            opts.max_object_distance = goal.max_object_distance

        if opts.save_detections_dir:
            run_id = time.strftime("%Y%m%d_%H%M%S")
            opts.save_detections_dir = os.path.join(
                opts.save_detections_dir, f"{run_id}_{'+'.join(opts.objects)}")

        self.get_logger().info(
            f"FindObject goal accepted: objects={opts.objects!r}, "
            f"max_object_distance={opts.max_object_distance}m"
        )

        def feedback_cb(phase, message, step):
            fb = FindObject.Feedback()
            fb.phase = phase
            fb.status = message
            fb.step = int(step)
            goal_handle.publish_feedback(fb)

        found = run_full_search(self, opts, feedback_cb=feedback_cb)

        goal_handle.succeed()

        result = FindObject.Result()
        result.success = bool(found)
        for f in found:
            obj = DetectedObject()
            obj.label = f.get("label", "?")
            obj.confidence = float(f.get("confidence", 0.0))
            x, y, z = f["position_base_link"]
            obj.position.x, obj.position.y, obj.position.z = float(x), float(y), float(z)
            obj.description = f.get("description", "")
            result.found_objects.append(obj)

        if found:
            result.message = f"{len(found)} object(s) found"
            self.get_logger().info(f"FindObject result: {result.message}")
        else:
            result.message = f"nothing found within {opts.max_object_distance}m"
            self.get_logger().info(f"FindObject result: {result.message}")

        return result


def _default_params_file():
    """husky_ur10_cam/config/all_params.yaml, if installed - the single
    file all TUNABLE_PARAMS (and stc_cpp's coverage params) are meant to
    be edited in, auto-loaded below so `ros2 run` alone picks it up."""
    try:
        from ament_index_python.packages import get_package_share_directory
        path = os.path.join(
            get_package_share_directory("husky_ur10_cam"), "config", "all_params.yaml")
        return path if os.path.isfile(path) else None
    except Exception:
        return None


def main(args=None):
    install_shutdown_handler()
    argv = list(sys.argv if args is None else args)
    if "--params-file" not in argv:
        default_params = _default_params_file()
        if default_params:
            argv += ["--ros-args", "--params-file", default_params]

    rclpy.init(args=argv)
    node = ObjectSearchActionServer()

    node.get_logger().info("Waiting for /compute_ik and /move_action...")
    if not node.wait_for_servers(30.0):
        node.get_logger().error("MoveIt servers not available - exiting")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    node.get_logger().info(
        "FindObject action server ready (default objects: "
        f"{node.get_parameter('objects').value}, "
        f"detections dir: {DEFAULT_DETECTIONS_DIR})")
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
