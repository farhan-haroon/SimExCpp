#!/usr/bin/env python3
"""Active perception: periodically checks the 4 radially-outward chassis
cameras (front_left/rear_left/rear_right/front_right - see combined.urdf.
xacro's chassis_corner_camera macro) PLUS the arm-mounted end-effector
camera, and asks the shared VLM (qwen_vlm_server.py) whether any of them
looks like a plausible place to find the configured target object(s) -
e.g. a bookshelf or a table with book-like items on it, for target "book"
- as opposed to, say, an empty corridor.

The EE camera's view is anchored at the arm's "home" pose (HOME_JOINTS in
ee_camera_environment_scan.py, matching the SRDF's "home" group_state):
this node doesn't command the arm itself, it just trusts that's where the
arm already is between searches - run_full_search() always returns it
there at the end of a search (skip_return_home), and stays there for as
long as the base is driving/being monitored.

A VLM-flagged direction only becomes a real hint if it's also within
max_check_distance_m (default 3m) of that camera, measured via its depth
point cloud at the frame center (the closest available proxy to "whatever
it was looking at" - there's no bbox here, just a direction) - see
_check_distance. Flagged-but-too-far or flagged-but-no-depth-reading both
suppress the hint rather than guess.

Publishes the latest verdict on /active_perception/hint
(custom_interfaces/msg/ActivePerceptionHint). stc_cpp's coverage node
(stc.py) is the consumer: it only calls FindObject at a new major cell if
the latest hint names a direction, and restricts that search to the
matching 90deg quadrant (or, for a "home" hint, does a full 360deg sweep
from home - same as no direction at all, just VLM-triggered instead of
always-on) instead of always running the full sweep.

Ticks on a fixed timer (--tick-interval-sec / tick_interval_sec ROS param)
while enabled. stc.py disables this node (via /active_perception/enable,
std_msgs/Bool) around each FindObject call so its periodic ticks don't
queue behind phase 2's own VLM calls on the shared qwen_vlm_server.

Usage:
    ros2 run husky_ur10_cam active_perception_node.py
"""
import math
import os
import sys

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Bool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ee_camera_environment_scan import (  # noqa: E402
    DIRECTION_ORDER, get_point_from_cloud, install_shutdown_handler,
)

from custom_interfaces.msg import ActivePerceptionHint  # noqa: E402
from custom_interfaces.srv import VlmQuery  # noqa: E402

DEFAULT_OBJECTS = ["cup", "book"]
DEFAULT_TICK_INTERVAL_SEC = 2.0
DEFAULT_MAX_CHECK_DISTANCE_M = 3.0
EE_CAMERA_TOPIC = "/ee_camera/image_raw"
EE_POINTS_TOPIC = "/ee_camera/points"

# Prompt-only vocabulary the VLM is asked to answer in - the 4 chassis
# directions (matches DIRECTION_ORDER exactly, upper-cased) plus HOME for
# the EE camera's view, plus NONE for "nothing promising here". Same "VLM
# only ever picks one word, fixed code does the rest" philosophy as
# ee_camera_environment_scan.py's CLASSIFICATION_WORDS.
HINT_WORDS = tuple(d.upper() for d in DIRECTION_ORDER) + ("HOME", "NONE")


class ActivePerceptionNode(Node):
    def __init__(self):
        super().__init__("active_perception_node")
        # Shared with object_search_action_server's "target_objects" and
        # kruskal_stc_node's "target_objects" via all_params.yaml's "/**"
        # block - see its comment there.
        self.declare_parameter("target_objects", list(DEFAULT_OBJECTS))
        self.declare_parameter("tick_interval_sec", DEFAULT_TICK_INTERVAL_SEC)
        # Chassis/EE cameras are depth cameras (combined.urdf.xacro's
        # chassis_corner_camera macro, and the pre-existing ee_camera) -
        # a flagged direction only becomes a real hint if something's
        # actually within this range at its frame center (see _check_
        # distance). No bbox is available here (the VLM only ever names a
        # direction, not a pixel) - frame center is the best available
        # proxy for "whatever it was looking at".
        self.declare_parameter("max_check_distance_m", DEFAULT_MAX_CHECK_DISTANCE_M)

        self._latest_images = {name: None for name in DIRECTION_ORDER}
        self._latest_points = {name: None for name in DIRECTION_ORDER}
        for name in DIRECTION_ORDER:
            self.create_subscription(
                Image, f"/chassis_cam_{name}/image_raw",
                lambda msg, name=name: self._image_cb(name, msg), 1)
            self.create_subscription(
                PointCloud2, f"/chassis_cam_{name}/points",
                lambda msg, name=name: self._points_cb(name, msg), 1)

        self._latest_ee_image = None
        self._latest_ee_points = None
        self.create_subscription(Image, EE_CAMERA_TOPIC, self._ee_image_cb, 1)
        self.create_subscription(PointCloud2, EE_POINTS_TOPIC, self._ee_points_cb, 1)

        self._enabled = True
        self.create_subscription(Bool, "/active_perception/enable", self._enable_cb, 10)
        self.hint_pub = self.create_publisher(ActivePerceptionHint, "/active_perception/hint", 1)

        # vlm_cli stays in the node's default callback group; _tick gets
        # its own so the nested rclpy.spin_until_future_complete() in
        # _query_vlm can actually dispatch vlm_cli's response callback -
        # a MutuallyExclusiveCallbackGroup (the default) never runs two
        # callbacks from the SAME group concurrently, even via a nested
        # spin, so without this _tick would be waiting on a response
        # callback that can never be scheduled while _tick itself is still
        # running - a guaranteed timeout, every single tick. Same reason
        # object_search_action_server.py's execute_callback gets its own
        # group (see its docstring).
        self.vlm_cli = self.create_client(VlmQuery, "vlm_query")
        self._tick_group = MutuallyExclusiveCallbackGroup()

        tick_interval = self.get_parameter("tick_interval_sec").value
        self.create_timer(tick_interval, self._tick, callback_group=self._tick_group)

    def _image_cb(self, name, msg):
        self._latest_images[name] = msg

    def _points_cb(self, name, msg):
        self._latest_points[name] = msg

    def _ee_image_cb(self, msg):
        self._latest_ee_image = msg

    def _ee_points_cb(self, msg):
        self._latest_ee_points = msg

    def _enable_cb(self, msg):
        self._enabled = msg.data
        if not self._enabled:
            self.get_logger().debug("Paused (a search is in progress)")
        else:
            self.get_logger().debug("Resumed")

    def _tick(self):
        if not self._enabled:
            return
        images = [self._latest_images[name] for name in DIRECTION_ORDER]
        images.append(self._latest_ee_image)
        points = [self._latest_points[name] for name in DIRECTION_ORDER]
        points.append(self._latest_ee_points)
        if any(v is None for v in images + points):
            self.get_logger().debug(
                "Waiting for all 4 chassis camera + EE camera image and "
                "depth frames before first check"
            )
            return

        objects = self.get_parameter("target_objects").value
        targets = ", ".join(objects)
        prompt = (
            f"You are shown 5 camera views from a mobile search robot, in "
            f"this fixed order: 1) front-left chassis camera, 2) rear-left "
            f"chassis camera, 3) rear-right chassis camera, 4) front-right "
            f"chassis camera, 5) the robot's arm-mounted camera, currently "
            f"parked facing forward at its home position. The robot is "
            f"searching for: {targets}.\n\n"
            f"Look for CONTEXT that suggests one of these objects is nearby "
            f"even if not directly visible - e.g. a bookshelf or a table "
            f"with book-like items on it is worth investigating for a "
            f"'book', but an empty corridor or bare wall is not. Do not "
            f"flag a view just because it contains furniture or "
            f"clutter in general - only if it plausibly relates to the "
            f"target object(s).\n\n"
            f"Which ONE of the 5 views (if any) looks most worth stopping "
            f"to investigate? Respond in EXACTLY this two-line format:\n"
            f"REASON: <one short sentence explaining your choice>\n"
            f"ANSWER: <one of FRONT_LEFT, REAR_LEFT, REAR_RIGHT, FRONT_RIGHT, "
            f"HOME, or NONE if none of them look promising>"
        )

        response = self._query_vlm(images, prompt, max_new_tokens=80)
        direction = self._parse_hint(response)
        reason = self._parse_reason(response)

        if direction:
            max_distance = self.get_parameter("max_check_distance_m").value
            distance = self._check_distance(direction)
            if distance is None:
                self.get_logger().info(
                    f"Active perception: {reason} -> {direction} flagged, but no "
                    f"valid depth reading at its frame center - suppressing "
                    f"(can't confirm it's within {max_distance}m)"
                )
                direction = ""
            elif distance > max_distance:
                self.get_logger().info(
                    f"Active perception: {reason} -> {direction} flagged, but "
                    f"~{distance:.1f}m away (> {max_distance}m limit) - suppressing"
                )
                direction = ""
            else:
                self.get_logger().info(
                    f"Active perception: {reason} -> {direction} (~{distance:.1f}m away)"
                )
        else:
            self.get_logger().info(f"Active perception: {reason} -> not stopping")

        self.get_logger().debug(f"Active perception raw response: {response!r}")
        self.hint_pub.publish(ActivePerceptionHint(direction=direction))

    def _check_distance(self, direction):
        """Straight-line distance (m) from the flagged camera to whatever's
        at its frame center, via its depth point cloud - the best available
        proxy without a specific pixel location (the VLM only ever names a
        direction, not a bbox). Returns None if unavailable/invalid there."""
        if direction in self._latest_images:
            image, points = self._latest_images[direction], self._latest_points[direction]
        else:  # "home"
            image, points = self._latest_ee_image, self._latest_ee_points
        point = get_point_from_cloud(points, image.width // 2, image.height // 2)
        if point is None:
            return None
        return math.sqrt(sum(c * c for c in point))

    def _query_vlm(self, images, prompt, max_new_tokens=10, timeout=60.0):
        req = VlmQuery.Request()
        req.images = images
        req.prompt = prompt
        req.max_new_tokens = max_new_tokens
        future = self.vlm_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        result = future.result()
        if result is None:
            self.get_logger().warn(
                f"vlm_query timed out after {timeout}s or failed - treating this "
                f"tick as no hint. If this keeps happening, check qwen_vlm_server's "
                f"own log (still loading the model? GPU busy/OOM?)."
            )
            return ""
        return result.response

    @staticmethod
    def _parse_hint(response):
        """First HINT_WORDS match after the response's last "ANSWER:"
        marker -> its matching direction (lowercased "home" included), or
        "" for NONE/unrecognized. Only looks after "ANSWER:" (falling back
        to the whole response if that marker's missing) rather than
        scanning the full response, since the REASON sentence can
        legitimately mention OTHER directions while explaining why they
        were rejected - a naive whole-response scan could match one of
        those instead of the actual answer. An unrecognized response
        defaults to "" (no hint) rather than guessing - the safer failure
        mode here is under- not over-triggering searches."""
        upper = response.upper()
        idx = upper.rfind("ANSWER:")
        tail = upper[idx + len("ANSWER:"):] if idx != -1 else upper
        for word in HINT_WORDS:
            if word in tail:
                return "" if word == "NONE" else word.lower()
        return ""

    @staticmethod
    def _parse_reason(response):
        """Text between "REASON:" and "ANSWER:" (for logging only - not
        safety-critical, so this is best-effort). Falls back to the whole
        response if the model didn't follow the requested format."""
        upper = response.upper()
        idx_r = upper.find("REASON:")
        idx_a = upper.find("ANSWER:")
        if idx_r == -1:
            return response.strip()
        end = idx_a if idx_a > idx_r else len(response)
        return response[idx_r + len("REASON:"):end].strip()


def _default_params_file():
    """husky_ur10_cam/config/all_params.yaml, if installed - see its
    header comment. Auto-loaded below so `ros2 run` alone picks it up."""
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
    node = ActivePerceptionNode()

    node.get_logger().info("Waiting for vlm_query...")
    if not node.vlm_cli.wait_for_service(timeout_sec=30.0):
        node.get_logger().error("vlm_query service not available - exiting")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    node.get_logger().info(
        f"active_perception_node ready (objects: {node.get_parameter('target_objects').value}, "
        f"tick interval: {node.get_parameter('tick_interval_sec').value}s, "
        f"max check distance: {node.get_parameter('max_check_distance_m').value}m)"
    )
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
