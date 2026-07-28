#!/usr/bin/env python3
"""Two-phase environment scan with the UR10-mounted end-effector camera.

Phase 1 - fixed pan/tilt sweep, YOLO only:
    The surroundings are divided into --num-sectors azimuths (default 16,
    spaced for ~60% frame overlap given the ee camera's ~57 deg horizontal
    FOV). At each azimuth the camera holds position and steps through
    --tilts-per-sector pitch angles between --tilt-up-deg and
    --tilt-down-deg. The arm returns to a fixed HOME_JOINTS pose before
    and after (--skip-start-home / --skip-return-home). YOLO
    (detect_objects) runs at every waypoint: detections >=
    DETECTION_CONFIDENCE_THRESHOLD are recorded as found; weaker ones are
    saved as candidates for phase 2. The sweep always runs to completion,
    so multiple objects can be found in one pass. Target-label detections
    are saved to --save-detections-dir with boxes drawn.

Phase 2 - revisit phase 1 candidates only, YOLO confirms:
    For each candidate, up to --max-tries attempts: YOLO re-checks the
    current view; a hit confirms the candidate and moves on to the next
    one. Otherwise the bbox center is back-projected through the depth
    cloud (/ee_camera/points) to the object's 3D position and the camera
    moves to --approach-standoff from it (compute_3d_approach_pose). If
    depth is invalid there or the move fails, vlm_guide() (Qwen2.5-VL)
    picks one qualitative pan/tilt/forward move instead. No open-ended
    search - every move only refines a candidate phase 1 already flagged.

Models: YOLOv8 (--yolo-weights) and Qwen2.5-VL-7B-Instruct (--qwen-model),
loaded lazily and cached for the process lifetime.

Usage:
    ros2 run husky_ur10_cam ee_camera_environment_scan.py
    ros2 run husky_ur10_cam ee_camera_environment_scan.py --objects chair book laptop
    ros2 run husky_ur10_cam ee_camera_environment_scan.py --dry-run
    ros2 run husky_ur10_cam ee_camera_environment_scan.py --skip-phase1
"""
import argparse
import math
import os
import struct
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from sensor_msgs.msg import Image, PointCloud2
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
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
CAMERA_FRAME = "camera_optical_link"
MAP_FRAME = "map"
MOVE_TIMEOUT = 30.0

DEFAULT_YOLO_WEIGHTS = os.path.expanduser("~/husky_ws/yolov8n.pt")
DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_OBJECT = "chair"
DEFAULT_DETECTIONS_DIR = os.path.expanduser("~/husky_ws/detections")

VLM_ROTATION_ACTIONS = ("pan_left", "pan_right", "tilt_up", "tilt_down")
# No lateral translation here - sideways moves shift the image via
# parallax in a direction a small VLM doesn't reliably predict. Pan/tilt
# is the intuitive tool for centering; "forward" closes distance.
VLM_APPROACH_ACTIONS = ("pan_left", "pan_right", "tilt_up", "tilt_down", "forward")
# Clamp the VLM's proposed step size so a single move can't be a huge,
# failure-prone jump.
MAX_TRANSLATION_STEP = 0.5   # meters
MAX_ROTATION_STEP = 0.17     # radians (~20 deg)
ALLOWED_PLANNING_TIME = 10.0

# Below this, a phase 1 detection is saved as a candidate for phase 2
# rather than counted as an immediate find. Fixed, not CLI-configurable.
DETECTION_CONFIDENCE_THRESHOLD = 0.7

# Fixed joint-space target rather than an IK solve at call time, so the
# arm returns to the exact same configuration every time (IK's
# elbow-up/down branch depends on the arm's current state, not just the
# target pose). Solved once via test_ee_pose_ik.py for camera_optical_link
# at xyz=(0.5, 0.0, 1.0) in base_link, facing +X, level (0 roll, 0 pitch).
HOME_JOINTS = {
    "ur10_shoulder_pan_joint": 5.8584,
    "ur10_shoulder_lift_joint": -1.9169,
    "ur10_elbow_joint": 1.8289,
    "ur10_wrist_1_joint": -3.0536,
    "ur10_wrist_2_joint": -1.1460,
    "ur10_wrist_3_joint": 0.0,
}

# Fallback when go_home() can't reach HOME_JOINTS directly - the UR10's
# tucked-in pose, reachable from almost any starting configuration.
RECOVERY_JOINTS = {
    "ur10_shoulder_pan_joint": 0.0,
    "ur10_shoulder_lift_joint": -1.57,
    "ur10_elbow_joint": 0.0,
    "ur10_wrist_1_joint": -1.57,
    "ur10_wrist_2_joint": 0.0,
    "ur10_wrist_3_joint": 0.0,
}


def _points_topic_for(image_topic):
    """Default organized-point-cloud topic for the depth_camera plugin
    that publishes `image_topic` - e.g. /ee_camera/image_raw ->
    /ee_camera/points. Override with --points-topic if it doesn't fit."""
    ns = image_topic.rsplit("/", 1)[0]
    return f"{ns}/points"


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


def look_at_quaternion(forward, up=(0.0, 0.0, 1.0)):
    """Orientation for a REP-103 optical frame (X=right, Y=down, Z=forward)
    that gazes along `forward` with zero roll relative to `up`."""
    f = _normalize(forward)
    u = _normalize(up)
    if abs(_dot(f, u)) > 0.999:
        u = (1.0, 0.0, 0.0) if abs(f[0]) < 0.9 else (0.0, 1.0, 0.0)
    right = _normalize(_cross(f, u))
    down = _cross(f, right)
    r = (
        (right[0], down[0], f[0]),
        (right[1], down[1], f[1]),
        (right[2], down[2], f[2]),
    )
    return _mat3_to_quat(r)


def get_point_from_cloud(points_msg, u, v):
    """Return (x, y, z) in the cloud's own frame at pixel (u, v) of an
    organized PointCloud2 (same width/height grid as the paired image),
    or None if out of bounds or the depth there is invalid (NaN/inf)."""
    if points_msg is None:
        return None
    if not (0 <= v < points_msg.height and 0 <= u < points_msg.width):
        return None
    offsets = {f.name: f.offset for f in points_msg.fields}
    if not {"x", "y", "z"} <= offsets.keys():
        return None
    base = v * points_msg.row_step + u * points_msg.point_step
    x, y, z = (
        struct.unpack_from("f", points_msg.data, base + offsets[axis])[0]
        for axis in "xyz"
    )
    if any(math.isnan(c) or math.isinf(c) for c in (x, y, z)):
        return None
    return (x, y, z)


def _rotate_vector_by_quat(v, quat):
    """Rotate vector `v` by quaternion `quat` (x,y,z,w)."""
    qx, qy, qz, qw = quat
    ux, uy, uz = qx, qy, qz
    dot_uv = ux * v[0] + uy * v[1] + uz * v[2]
    cross = (uy * v[2] - uz * v[1], uz * v[0] - ux * v[2], ux * v[1] - uy * v[0])
    u_len_sq = ux * ux + uy * uy + uz * uz
    return tuple(
        2 * dot_uv * u_i + (qw * qw - u_len_sq) * v_i + 2 * qw * c_i
        for u_i, v_i, c_i in zip((ux, uy, uz), v, cross)
    )


def transform_point_to_base_link(point_camera_frame, camera_xyz, camera_quat):
    """A point given in the camera's own frame -> base_link, given the
    camera's current (xyz, quat) pose in base_link."""
    rotated = _rotate_vector_by_quat(point_camera_frame, camera_quat)
    return tuple(r + c for r, c in zip(rotated, camera_xyz))


def select_tracked_detection(detections, last_bbox):
    """Pick the detection to keep pursuing this try. With multiple
    instances of the target class in frame, track by proximity to
    `last_bbox` so investigation stays locked onto the same object across
    tries, rather than jumping to whichever is highest-confidence.
    Falls back to highest confidence on the first try (no prior bbox)."""
    if not detections:
        return None
    if last_bbox is None:
        return max(detections, key=lambda d: d["confidence"])

    def center(bbox):
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    lcx, lcy = center(last_bbox)

    def dist_sq(det):
        cx, cy = center(det["bbox"])
        return (cx - lcx) ** 2 + (cy - lcy) ** 2

    return min(detections, key=dist_sq)


def measure_bbox_distance(points_msg, bbox):
    """Straight-line distance (meters) from the camera to whatever's at
    bbox's center pixel, via the depth point cloud - frame-independent
    (just the vector magnitude from the camera's own origin), so no pose
    transform needed. Returns None if depth is unavailable/invalid there."""
    cx = int((bbox[0] + bbox[2]) / 2)
    cy = int((bbox[1] + bbox[3]) / 2)
    point = get_point_from_cloud(points_msg, cx, cy)
    if point is None:
        return None
    return math.sqrt(sum(c * c for c in point))


def back_project_object_position(node, bbox, points_msg=None):
    """Best-effort 3D position (base_link) of whatever's at bbox's center
    pixel, via depth back-projection through the point cloud. This is the
    OBJECT's position, not the camera/waypoint's - a phase 1 sweep
    waypoint sits on a fixed orbit around the robot and can be well short
    of the object along the line of sight.

    Pass an already-captured points_msg to avoid a redundant capture;
    otherwise one is captured here. Returns None if depth or the camera's
    TF pose isn't available."""
    if points_msg is None:
        points_msg = node.capture_points(timeout=2.0)
    cx = int((bbox[0] + bbox[2]) / 2)
    cy = int((bbox[1] + bbox[3]) / 2)
    point_cam = get_point_from_cloud(points_msg, cx, cy)
    if point_cam is None:
        return None
    camera_xyz, camera_quat = node.get_camera_pose(timeout=2.0)
    if camera_xyz is None:
        return None
    return transform_point_to_base_link(point_cam, camera_xyz, camera_quat)


def compute_3d_approach_pose(object_xyz, camera_xyz, standoff, max_step=None):
    """New camera (xyz, quat), both in base_link, that moves toward
    `object_xyz` along the current camera->object line, ending up
    `standoff` meters short of it.

    If `max_step` is given and the distance to standoff exceeds it, only
    moves max_step closer this call instead of jumping the whole way -
    a single depth reading can be noisy, and a full-distance target can
    land outside the arm's reach. Bounded steps converge over several
    tries instead. If already closer than standoff, travel is negative
    (backs off)."""
    direction = tuple(o - c for o, c in zip(object_xyz, camera_xyz))
    dist = math.sqrt(sum(d * d for d in direction))
    if dist < 1e-6:
        return camera_xyz, look_at_quaternion((1.0, 0.0, 0.0))
    unit_dir = tuple(d / dist for d in direction)
    travel = dist - standoff
    if max_step is not None:
        travel = max(-max_step, min(travel, max_step))
    new_xyz = tuple(c + d * travel for c, d in zip(camera_xyz, unit_dir))
    return new_xyz, look_at_quaternion(unit_dir)


REVERSE_ACTION = {
    "forward": "backward", "backward": "forward",
    "pan_left": "pan_right", "pan_right": "pan_left",
    "tilt_up": "tilt_down", "tilt_down": "tilt_up",
}


def command_to_offset(action, amount):
    """Relative move vocabulary as (xyz, rpy) offsets in the camera's own
    current frame (REP-103 optical convention: X=right, Y=down, Z=forward).
    Signs verified against look_at_quaternion in test_ee_pose_ik.py."""
    table = {
        "forward": ((0.0, 0.0, amount), (0.0, 0.0, 0.0)),
        "backward": ((0.0, 0.0, -amount), (0.0, 0.0, 0.0)),
        "right": ((amount, 0.0, 0.0), (0.0, 0.0, 0.0)),
        "left": ((-amount, 0.0, 0.0), (0.0, 0.0, 0.0)),
        "up": ((0.0, -amount, 0.0), (0.0, 0.0, 0.0)),
        "down": ((0.0, amount, 0.0), (0.0, 0.0, 0.0)),
        "pan_right": ((0.0, 0.0, 0.0), (0.0, amount, 0.0)),
        "pan_left": ((0.0, 0.0, 0.0), (0.0, -amount, 0.0)),
        "tilt_up": ((0.0, 0.0, 0.0), (amount, 0.0, 0.0)),
        "tilt_down": ((0.0, 0.0, 0.0), (-amount, 0.0, 0.0)),
    }
    if action not in table:
        raise ValueError(f"Unknown action '{action}'")
    return table[action]


class EnvironmentScanner(Node):
    def __init__(self, image_topic, points_topic=None, node_name="ee_camera_environment_scan"):
        super().__init__(node_name)
        self.ik_cli = self.create_client(GetPositionIK, "/compute_ik")
        self.move_ac = ActionClient(self, MoveGroupAction, "/move_action")
        self._latest_image = None
        self.create_subscription(Image, image_topic, self._image_cb, 1)

        self._latest_points = None
        self.create_subscription(
            PointCloud2, points_topic or _points_topic_for(image_topic),
            self._points_cb, 1,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Transient-local so RViz picks up the last-published markers even
        # if it's (re)started after this node already found something.
        marker_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.found_marker_pub = self.create_publisher(
            MarkerArray, "found_objects_markers", marker_qos)
        # RViz keys markers by (namespace, id); a never-reused counter
        # across calls to publish_found_markers keeps each search run's
        # markers from overwriting the previous run's.
        self._next_marker_id = 0

    def _image_cb(self, msg):
        self._latest_image = msg

    def _points_cb(self, msg):
        self._latest_points = msg

    def wait_for_servers(self, timeout=30.0):
        return (self.ik_cli.wait_for_service(timeout_sec=timeout)
                and self.move_ac.wait_for_server(timeout_sec=timeout))

    def capture_points(self, timeout=5.0):
        """Block briefly for a fresh organized point cloud after this
        call; return the latest one received, or None on timeout."""
        self._latest_points = None
        deadline = time.time() + timeout
        while self._latest_points is None and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self._latest_points

    def get_camera_pose(self, timeout=2.0):
        """Current (xyz, quat) of CAMERA_FRAME in base_link, via TF.
        Returns (None, None) if the transform isn't available in time."""
        from rclpy.time import Time
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                t = self.tf_buffer.lookup_transform(
                    "base_link", CAMERA_FRAME, Time())
                xyz = (t.transform.translation.x, t.transform.translation.y,
                       t.transform.translation.z)
                quat = (t.transform.rotation.x, t.transform.rotation.y,
                        t.transform.rotation.z, t.transform.rotation.w)
                return xyz, quat
            except Exception:
                rclpy.spin_once(self, timeout_sec=0.1)
        return None, None

    def publish_found_markers(self, found, timeout=2.0):
        """Publish an RViz MarkerArray (a labeled sphere per object) in
        MAP_FRAME for every confirmed find. Each found dict's
        "position_base_link" is transformed via the map -> base_link TF
        (the base doesn't move during a search, so this is exact). Does
        nothing if there's nothing found or the map transform isn't
        available (e.g. no localization running)."""
        if not found:
            return
        from rclpy.time import Time
        deadline = time.time() + timeout
        map_xyz = map_quat = None
        while time.time() < deadline:
            try:
                t = self.tf_buffer.lookup_transform(MAP_FRAME, "base_link", Time())
                map_xyz = (t.transform.translation.x, t.transform.translation.y,
                           t.transform.translation.z)
                map_quat = (t.transform.rotation.x, t.transform.rotation.y,
                            t.transform.rotation.z, t.transform.rotation.w)
                break
            except Exception:
                rclpy.spin_once(self, timeout_sec=0.1)
        if map_xyz is None:
            self.get_logger().warn(
                f"publish_found_markers: no {MAP_FRAME}->base_link transform "
                "available, skipping markers"
            )
            return

        stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()
        for f in found:
            marker_id = self._next_marker_id
            self._next_marker_id += 1
            p_map = tuple(m + r for m, r in
                           zip(map_xyz, _rotate_vector_by_quat(f["position_base_link"], map_quat)))

            sphere = Marker()
            sphere.header.frame_id = MAP_FRAME
            sphere.header.stamp = stamp
            sphere.ns = "found_objects"
            sphere.id = marker_id
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x, sphere.pose.position.y, sphere.pose.position.z = p_map
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.2
            sphere.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.9)
            markers.markers.append(sphere)

            label = Marker()
            label.header.frame_id = MAP_FRAME
            label.header.stamp = stamp
            label.ns = "found_objects_labels"
            label.id = marker_id
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x, label.pose.position.y = p_map[0], p_map[1]
            label.pose.position.z = p_map[2] + 0.25
            label.pose.orientation.w = 1.0
            label.scale.z = 0.15
            label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.9)
            label.text = f"{f.get('label', '?')} ({f.get('confidence', 0.0):.2f})"
            markers.markers.append(label)

        self.found_marker_pub.publish(markers)
        self.get_logger().info(
            f"Published {len(found)} found-object marker(s) in {MAP_FRAME} frame"
        )

    def capture_image(self, timeout=5.0):
        """Block briefly for a fresh frame after this call; return the
        latest one received, or None on timeout."""
        self._latest_image = None
        deadline = time.time() + timeout
        while self._latest_image is None and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self._latest_image

    def solve_ik(self, xyz, quat, group, frame_id=CAMERA_FRAME, attempts=5, timeout=0.1):
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = xyz
        qx, qy, qz, qw = quat
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        for _ in range(attempts):
            req = GetPositionIK.Request()
            req.ik_request = PositionIKRequest()
            req.ik_request.group_name = group
            req.ik_request.ik_link_name = CAMERA_FRAME
            req.ik_request.pose_stamped = pose
            req.ik_request.avoid_collisions = True
            req.ik_request.timeout.sec = int(timeout)
            req.ik_request.timeout.nanosec = int((timeout % 1) * 1e9)

            future = self.ik_cli.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            result = future.result()

            if result is not None and result.error_code.val == MoveItErrorCodes.SUCCESS:
                return dict(zip(
                    result.solution.joint_state.name,
                    result.solution.joint_state.position,
                ))
        return None

    def execute_joints(self, joint_positions, group, vel_scale, accel_scale):
        constraints = []
        for joint_name, position in joint_positions.items():
            if not joint_name.startswith("ur10_"):
                continue
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

        send_future = self.move_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=MOVE_TIMEOUT)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=MOVE_TIMEOUT)
        result = result_future.result()
        if result is None:
            return False

        return result.result.error_code.val == MoveItErrorCodes.SUCCESS

    def move(self, action, amount, group, vel_scale, accel_scale):
        """Relative move, expressed in the camera's own current frame."""
        xyz, rpy = command_to_offset(action, amount)
        quat = quat_from_rpy(*rpy)
        joints = self.solve_ik(xyz, quat, group, frame_id=CAMERA_FRAME)
        if joints is None:
            self.get_logger().warn(f"IK failed for {action}({amount:.3f})")
            return False
        if not self.execute_joints(joints, group, vel_scale, accel_scale):
            self.get_logger().warn(f"Execution failed for {action}({amount:.3f})")
            return False
        return True

    def move_to_pose(self, xyz, quat, group, vel_scale, accel_scale,
                      frame_id="base_link", attempts=1):
        """Absolute move to a fixed waypoint.

        With the default attempts=1, a single planning failure just fails
        (cheap to skip a phase 1 waypoint). Pass attempts>1 for a
        stronger guarantee (phase 2 revisiting a saved candidate): retries
        planning up to `attempts` times (OMPL's planner isn't
        deterministic, so a retry can succeed where the last one didn't),
        then falls back through RECOVERY_JOINTS and makes one final
        attempt at the real target from there."""
        joints = self.solve_ik(xyz, quat, group, frame_id=frame_id)
        if joints is None:
            return False

        for _ in range(attempts):
            if self.execute_joints(joints, group, vel_scale, accel_scale):
                return True

        if attempts > 1:
            self.get_logger().warn(
                "move_to_pose: falling back through RECOVERY_JOINTS to "
                "retry the target"
            )
            if (self.execute_joints(RECOVERY_JOINTS, group, vel_scale, accel_scale)
                    and self.execute_joints(joints, group, vel_scale, accel_scale)):
                return True

        return False

    def go_home(self, group, vel_scale, accel_scale, attempts=3):
        """Move to the fixed HOME_JOINTS target (see HOME_JOINTS comment).

        Retries planning up to `attempts` times - reaching a fixed joint
        target still needs a collision-free path from wherever the arm
        currently is, and OMPL's planner isn't deterministic, so a retry
        can succeed where the last one didn't.

        If all attempts fail, falls back to RECOVERY_JOINTS then makes
        one more try at HOME_JOINTS from there. Returns True either way
        the arm ends up in a known-good state, logging which one."""
        for attempt in range(1, attempts + 1):
            if self.execute_joints(HOME_JOINTS, group, vel_scale, accel_scale):
                return True
            self.get_logger().warn(f"go_home attempt {attempt}/{attempts} failed")

        self.get_logger().warn(
            "go_home: falling back to RECOVERY_JOINTS (not the exact home config)"
        )
        if not self.execute_joints(RECOVERY_JOINTS, group, vel_scale, accel_scale):
            self.get_logger().error("go_home: recovery pose also failed")
            return False

        self.get_logger().debug(
            "Reached recovery pose - retrying exact HOME_JOINTS from there"
        )
        if self.execute_joints(HOME_JOINTS, group, vel_scale, accel_scale):
            return True

        self.get_logger().warn(
            "go_home: still couldn't reach exact HOME_JOINTS, staying at recovery pose"
        )
        return True


def generate_helix_plan(radius, height, tilt_up_deg, tilt_down_deg,
                         num_sectors, tilts_per_sector, center_xy=(0.0, 0.0)):
    """Sparse pan/tilt sweep around the robot's own vertical axis.

    `num_sectors` evenly-spaced azimuths around a fixed `radius`/`height`
    orbit; at each azimuth, `tilts_per_sector` waypoints span elevation
    angles between +tilt_up_deg and -tilt_down_deg. Default 16 sectors x
    2 tilts = 32 waypoints, spaced (22.5 deg) for ~60% frame overlap given
    the ee camera's ~57 deg horizontal FOV.

    Returns a list of (xyz, quat) absolute waypoints in base_link."""
    cx, cy = center_xy
    sector_width = 2.0 * math.pi / num_sectors

    if tilts_per_sector <= 1:
        tilt_offsets_deg = [0.0]
    else:
        span = tilt_up_deg + tilt_down_deg
        tilt_offsets_deg = [
            tilt_up_deg - i * span / (tilts_per_sector - 1)
            for i in range(tilts_per_sector)
        ]

    plan = []
    for sector in range(num_sectors):
        theta = sector * sector_width
        x = cx + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)
        for tilt_off_deg in tilt_offsets_deg:
            tilt = math.radians(tilt_off_deg)
            forward = (math.cos(theta) * math.cos(tilt),
                       math.sin(theta) * math.cos(tilt),
                       math.sin(tilt))
            quat = look_at_quaternion(forward)
            plan.append(((x, y, height), quat))
    return plan


_yolo_model = None
_qwen_model = None
_qwen_processor = None
_cv_bridge = None


def _get_cv_bridge():
    global _cv_bridge
    if _cv_bridge is None:
        from cv_bridge import CvBridge
        _cv_bridge = CvBridge()
    return _cv_bridge


def _image_msg_to_pil(image_msg):
    from PIL import Image as PILImage
    cv_img = _get_cv_bridge().imgmsg_to_cv2(image_msg, desired_encoding="rgb8")
    return PILImage.fromarray(cv_img)


def _get_yolo(weights_path):
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO(weights_path)
    return _yolo_model


def _get_qwen(model_id):
    global _qwen_model, _qwen_processor
    if _qwen_model is None:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        # device_map="auto" (not a single fixed cuda:N) - the 7B checkpoint
        # is ~16.6GB in bf16, which doesn't reliably fit on one 16GB card
        # alongside whatever else is resident (Gazebo/rviz, etc). "auto"
        # lets accelerate shard it across both GPUs as needed.
        _qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
            local_files_only=True,
        )
        _qwen_processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    return _qwen_model, _qwen_processor


def _save_detection_image(cv_img, detections, step, save_dir, label_hint=DEFAULT_OBJECT):
    """Draw a box + label/confidence for every detection onto a copy of
    the frame and save it to `save_dir` for later review. Works with an
    empty `detections` list too (just saves the plain frame). Returns
    the saved path."""
    import cv2
    os.makedirs(save_dir, exist_ok=True)

    annotated = cv_img.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{det['label']} {det['confidence']:.2f}"
        cv2.putText(annotated, text, (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if detections:
        best = max(detections, key=lambda d: d["confidence"])
        label_str = f"{best['label']}_{best['confidence']:.2f}"
    else:
        label_str = f"{label_hint}_none"

    step_str = f"step{step:03d}" if isinstance(step, int) else str(step)
    filename = f"{step_str}_{label_str}.jpg"
    path = os.path.join(save_dir, filename)
    cv2.imwrite(path, annotated)
    return path


def detect_objects(image_msg, context):
    """YOLOv8 detector for context["target_labels"] (a list of YOLO/COCO
    class names; a bare context["target_label"] string is also accepted
    and normalized to a one-element list). Detects any of them in one
    pass - phase 1 passes the full search list, phase 2 passes just the
    one class it's currently investigating. Confidence thresholding
    (found vs. save-as-candidate) is decided by the caller, not here.

    If context["save_detections_dir"] is set, saves one image for the
    frame (target-label boxes drawn, or the plain frame if none) and
    records its path on every detection dict - on a hit, or always if
    context["save_always"] is set (phase 2 keeps every try's frame).

    Returns a list of zero or more:
        {"confidence": float, "label": str, "description": str,
         "bbox": [x1, y1, x2, y2], "image_path": str or None}
    """
    weights_path = context.get("yolo_weights", DEFAULT_YOLO_WEIGHTS)
    target_labels = context.get("target_labels")
    if target_labels is None:
        target_labels = [context.get("target_label", DEFAULT_OBJECT)]
    save_dir = context.get("save_detections_dir")
    model = _get_yolo(weights_path)

    cv_img = _get_cv_bridge().imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
    results = model(cv_img, verbose=False)[0]

    detections = []
    for box in results.boxes:
        label = model.names[int(box.cls[0])]
        if label not in target_labels:
            continue
        conf = float(box.conf[0])
        xyxy = [round(v, 1) for v in box.xyxy[0].tolist()]
        detections.append({
            "confidence": conf,
            "label": label,
            "description": f"YOLO detected '{label}' (conf={conf:.2f}, bbox={xyxy})",
            "bbox": xyxy,
            "image_path": None,
        })

    save_always = context.get("save_always", False)
    if save_dir and (detections or save_always):
        path = _save_detection_image(cv_img, detections, context.get("step", 0),
                                      save_dir, "+".join(target_labels))
        for det in detections:
            det["image_path"] = path

    return detections


# Single-word classification -> fixed (action, amount). The VLM only
# ever has to pick a word from CLASSIFICATION_WORDS - the actual move
# magnitude is fixed/deterministic code, not generated text (a generated
# "amount" number was found empirically unreliable too).
CLASSIFICATION_WORDS = ("LEFT", "RIGHT", "ABOVE", "BELOW", "CENTERED", "UNSEEN")
_CLASSIFICATION_TO_MOVE = {
    "LEFT": ("pan_left", 0.2),
    "RIGHT": ("pan_right", 0.2),
    "ABOVE": ("tilt_up", 0.2),
    "BELOW": ("tilt_down", 0.2),
    "CENTERED": ("forward", 0.1),
    "UNSEEN": ("pan_left", 0.3),  # arbitrary sweep direction to try to relocate it
}


def _classify_to_move(text):
    """Pick out the first CLASSIFICATION_WORDS match in the VLM's
    response and map it to its fixed move. Falls back to a small pan if
    the response doesn't contain any recognizable classification word."""
    upper = text.upper()
    for word in CLASSIFICATION_WORDS:
        if word in upper:
            action, amount = _CLASSIFICATION_TO_MOVE[word]
            return {
                "action": action,
                "amount": amount,
                "description": f"VLM classified object as {word} of frame center",
            }
    action, amount = _CLASSIFICATION_TO_MOVE["UNSEEN"]
    return {
        "action": action,
        "amount": amount,
        "description": f"(unrecognized VLM response: {text[:200]!r})",
    }


def vlm_guide(image_msg, context):
    """Qwen2.5-VL-7B-Instruct move-guidance call for phase 2.

    The VLM never decides whether the object is present - YOLO does
    that. This only ever picks ONE move to get a better look at a
    candidate phase 1 already flagged (context["investigating_candidate"],
    required), called up to --max-tries times between YOLO re-checks, and
    only when context["detector_bbox"] is available (the caller handles
    the no-detection case itself with deterministic backoff, since the
    VLM has nothing to ground on there).

    Restricted to VLM_APPROACH_ACTIONS (pan/tilt + forward, no lateral
    translation - translation shifts the image via parallax, which reads
    as unintuitive drift rather than a direct fix).

    Asks for a single-word position classification (LEFT/RIGHT/ABOVE/
    BELOW/CENTERED) grounded with context["detector_bbox"] as explicit
    numeric frame-position text; the move and step amount are then
    decided by fixed code (_CLASSIFICATION_TO_MOVE), not generated by the
    model - free-form judgment and full JSON move output were both tried
    and were unreliable at tracking the bbox correctly.

    Returns:
        {"action": str, "amount": float, "description": str}
    """
    import torch

    model_id = context.get("qwen_model", DEFAULT_QWEN_MODEL)
    candidate = context["investigating_candidate"]
    target_label = candidate["label"]
    bbox = context["detector_bbox"]

    model, processor = _get_qwen(model_id)
    pil_image = _image_msg_to_pil(image_msg)
    img_w, img_h = pil_image.size

    x1, y1, x2, y2 = bbox
    cx_pct = ((x1 + x2) / 2.0) / img_w * 100.0
    cy_pct = ((y1 + y2) / 2.0) / img_h * 100.0
    w_pct = (x2 - x1) / img_w * 100.0
    grounding = (
        f"\n\nThe object detector's current bounding box for the "
        f"'{candidate['label']}' is centered at {cx_pct:.0f}% across "
        f"and {cy_pct:.0f}% down the frame (0,0 = top-left corner, "
        f"50,50 = dead center, 100,100 = bottom-right corner), "
        f"covering about {w_pct:.0f}% of the frame's width. TRUST "
        f"THESE NUMBERS over your own visual impression, which can be "
        f"unreliable."
    )

    prompt = (
        f"You are guiding a robot arm-mounted camera to get a clear, "
        f"close, centered view of a possible '{target_label}'."
        f"{grounding}\n\n"
        f"Classify where the '{target_label}' is relative to the CENTER "
        f"of the frame. Answer with EXACTLY ONE WORD and nothing else:\n"
        f"LEFT - it is left of center\n"
        f"RIGHT - it is right of center\n"
        f"ABOVE - it is above center\n"
        f"BELOW - it is below center\n"
        f"CENTERED - it is already roughly centered (both close to 50%)"
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[pil_image], return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(gen, skip_special_tokens=True)[0]

    return _classify_to_move(response)


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--helix-radius", type=float, default=0.6,
                    help="Phase 1 sweep orbit radius around base_link (m)")
    p.add_argument("--helix-height", type=float, default=1.0,
                    help="Phase 1 sweep orbit height in base_link (m)")
    p.add_argument("--tilt-up-deg", type=float, default=5.0,
                    help="Camera pitch swept up this many degrees per sector")
    p.add_argument("--tilt-down-deg", type=float, default=30.0,
                    help="Camera pitch swept down this many degrees per sector")
    p.add_argument("--num-sectors", type=int, default=16,
                    help="Number of azimuths around the robot (360/N deg apart, "
                         "N/2 to each side). Default matches the ee camera's "
                         "current ~57 deg horizontal FOV for ~60% frame-to-frame "
                         "overlap (22.5 deg spacing) - raise --num-sectors "
                         "further if the FOV is narrowed more")
    p.add_argument("--tilts-per-sector", type=int, default=2,
                    help="Tilt waypoints per azimuth (2 = up then down)")
    p.add_argument("--helix-center", nargs=2, type=float, default=[0.0, 0.0],
                    metavar=("CX", "CY"),
                    help="Helix center (x, y) in base_link")
    p.add_argument("--objects", nargs="+", default=[DEFAULT_OBJECT],
                    help="One or more objects to search for - each must be "
                         "a YOLO/COCO class name (e.g. --objects chair book "
                         "bottle). Phase 1 sweeps once, checking every "
                         "waypoint's frame for ANY of them; phase 2 then "
                         "investigates each candidate as its own specific "
                         "class, so a chair candidate is never confused "
                         "with a book detection in the same frame.")
    p.add_argument("--yolo-weights", default=DEFAULT_YOLO_WEIGHTS,
                    help="Path to YOLO weights file for phase 1 detection")
    p.add_argument("--save-detections-dir", default=DEFAULT_DETECTIONS_DIR,
                    help="Directory to save bounding-box-overlaid images for "
                         "every target-label detection. Empty string disables saving.")
    p.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL,
                    help="HF model id/path for phase 2 VLM reasoning")
    p.add_argument("--max-tries", type=int, default=5,
                    help="Per-candidate budget: VLM-guided moves + YOLO "
                         "re-checks before giving up on a candidate and "
                         "moving to the next one")
    p.add_argument("--revisit-attempts", type=int, default=3,
                    help="Retries (+ RECOVERY_JOINTS fallback) when phase 2 "
                         "tries to move back to a saved candidate pose, "
                         "before giving up on that candidate entirely")
    p.add_argument("--phase2-max-steps", type=int, default=20,
                    help="Overall cap on total phase 2 steps across all candidates")
    p.add_argument("--skip-phase1", action="store_true")
    p.add_argument("--skip-phase2", action="store_true")
    p.add_argument("--skip-start-home", action="store_true",
                    help="Don't move to HOME_JOINTS before the scan starts")
    p.add_argument("--skip-return-home", action="store_true",
                    help="Don't move back to HOME_JOINTS after the scan ends")
    p.add_argument("--dwell", type=float, default=0.5,
                    help="Seconds to settle before capturing each image")
    p.add_argument("--group", default=PLANNING_GROUP)
    p.add_argument("--vel-scale", type=float, default=1.0)
    p.add_argument("--accel-scale", type=float, default=1.0)
    p.add_argument("--image-topic", default="/ee_camera/image_raw")
    p.add_argument("--points-topic", default=None,
                    help="Organized point cloud topic for depth-based 3D "
                         "approach moves in phase 2 (default: derived from "
                         "--image-topic, e.g. .../points)")
    p.add_argument("--approach-standoff", type=float, default=0.15,
                    help="Phase 2 3D approach: stop this many meters short "
                         "of the object's depth-measured position")
    p.add_argument("--approach-max-step", type=float, default=0.3,
                    help="Phase 2 3D approach: cap a single move to this many "
                         "meters closer, converging over --max-tries rather "
                         "than jumping straight to the standoff distance")
    p.add_argument("--max-object-distance", type=float, default=3.0,
                    help="Phase 1 candidates measured (via depth) further than "
                         "this from the camera are not handed to phase 2. "
                         "Candidates with no valid depth reading are kept "
                         "(can't rule them in or out)")
    p.add_argument("--dry-run", action="store_true",
                    help="Print the phase 1 helix waypoints without commanding the robot")
    return p.parse_args(argv)


def run_phase1_helix(node, plan, opts, feedback_cb=None):
    """Deterministic sweep of the easily-visible region, with YOLO
    (detect_objects) checked at every waypoint.

    Does not stop early on a high-confidence detection - always covers
    every waypoint, so multiple objects can be found in one run.

    Logs each found location at INFO; per-waypoint progress is DEBUG.

    feedback_cb, if given, is called as feedback_cb(phase, message, step)
    once per waypoint (ROS action feedback); a no-op for plain CLI use.

    Returns (found, candidates):
      - found: list of detection dicts (confidence >=
        DETECTION_CONFIDENCE_THRESHOLD), each with a "position_base_link".
      - candidates: list of {"position_base_link", "gaze_quat",
        "confidence", "label", "description", "step"} for every
        below-threshold-but-plausible (confidence > 0) detection within
        --max-object-distance (measured via the depth point cloud at the
        bbox center; kept if depth is unavailable there), for phase 2 to
        go re-investigate."""
    node.get_logger().debug(f"Phase 1: helix sweep, {len(plan)} waypoints")
    failures = 0
    candidates = []
    found = []

    for i, (xyz, quat) in enumerate(plan):
        ok = node.move_to_pose(xyz, quat, opts.group, opts.vel_scale,
                                opts.accel_scale)
        if not ok:
            failures += 1
            node.get_logger().debug(f"Phase 1 waypoint {i + 1}/{len(plan)} failed, skipping")
            if feedback_cb:
                feedback_cb("phase1", f"waypoint {i + 1}/{len(plan)}: move failed", i + 1)
            continue

        image = node.capture_image(timeout=5.0)
        if image is None:
            node.get_logger().debug(f"Phase 1 waypoint {i + 1}: no image received")
            if feedback_cb:
                feedback_cb("phase1", f"waypoint {i + 1}/{len(plan)}: no image", i + 1)
            time.sleep(opts.dwell)
            continue

        detections = detect_objects(image, {
            "step": i, "waypoint_xyz": xyz, "waypoint_quat": quat,
            "yolo_weights": opts.yolo_weights, "target_labels": opts.objects,
            "save_detections_dir": os.path.join(opts.save_detections_dir, "phase1")
                                   if opts.save_detections_dir else None,
        })
        for det in detections:
            conf = det.get("confidence", 0.0)
            if conf <= 0.0:
                continue
            loc = tuple(round(v, 3) for v in xyz)
            if conf >= DETECTION_CONFIDENCE_THRESHOLD:
                obj_xyz = (back_project_object_position(node, det["bbox"])
                           if det.get("bbox") else None)
                position = obj_xyz if obj_xyz is not None else xyz
                loc = tuple(round(v, 3) for v in position)
                node.get_logger().info(
                    f"Phase 1 FOUND: {det.get('label', '?')} "
                    f"(conf={conf:.2f}) at {loc}"
                )
                found.append({
                    "found": True, "confidence": conf,
                    "description": det.get("description", ""),
                    "label": det.get("label", "?"),
                    "position_base_link": position,
                })
                continue

            dist = None
            if det.get("bbox"):
                points_msg = node.capture_points(timeout=2.0)
                dist = measure_bbox_distance(points_msg, det["bbox"])

            if dist is not None and dist > opts.max_object_distance:
                node.get_logger().debug(
                    f"Phase 1 candidate skipped: {det.get('label', '?')} at {loc} "
                    f"is ~{dist:.1f}m away, beyond --max-object-distance "
                    f"({opts.max_object_distance}m)"
                )
                continue

            dist_str = f", ~{dist:.1f}m" if dist is not None else ""
            node.get_logger().info(
                f"Phase 1 candidate: {det.get('label', '?')} "
                f"(conf={conf:.2f}{dist_str}) at {loc} -> phase 2"
            )
            candidates.append({
                "position_base_link": xyz,
                "gaze_quat": quat,
                "confidence": conf,
                "label": det.get("label", "?"),
                "description": det.get("description", ""),
                "bbox": det.get("bbox"),
                "distance_m": dist,
                "step": i,
            })

        if feedback_cb:
            feedback_cb("phase1", f"waypoint {i + 1}/{len(plan)}: "
                        f"{len(found)} found, {len(candidates)} candidate(s) so far",
                        i + 1)
        time.sleep(opts.dwell)

    node.get_logger().debug(
        f"Phase 1 complete: {len(plan) - failures}/{len(plan)} waypoints reached, "
        f"{len(found)} found, {len(candidates)} candidate(s) for phase 2"
    )
    return found, candidates


def run_phase2_vlm(node, opts, candidates=None, feedback_cb=None):
    """Investigate phase 1 candidates only - no open-ended exploration.
    YOLO decides found, the VLM only decides how to move.

    feedback_cb, if given, is called as feedback_cb(phase, message, step)
    at each candidate/try boundary; a no-op for plain CLI use.

    For each candidate (in order): move to its saved position, then for
    up to opts.max_tries attempts, alternate YOLO re-detection and a move
    to get a closer/better-centered look (a precise depth-based 3D move
    when possible, else vlm_guide()'s qualitative pan/tilt/forward, else
    deterministic backoff - see the branches below). A confirmed
    candidate doesn't stop the scan - investigation continues to the
    next one, so multiple objects can be confirmed in one run. Also
    respects an overall opts.phase2_max_steps cap across all candidates.

    Logs each move's reasoning, one line per try (INFO); routine
    step/progress bookkeeping is DEBUG.

    Returns a list of found detection dicts (each with a
    "position_base_link"), possibly empty."""
    candidates = candidates or []
    node.get_logger().debug(
        f"Phase 2: investigating {len(candidates)} phase 1 candidate(s), "
        f"up to {opts.max_tries} tries each (max {opts.phase2_max_steps} "
        f"total steps), no open-ended exploration"
    )

    step = 0
    found = []
    for cand_idx, cand in enumerate(candidates):
        if step >= opts.phase2_max_steps:
            break

        ok = node.move_to_pose(cand["position_base_link"], cand["gaze_quat"],
                                opts.group, opts.vel_scale, opts.accel_scale,
                                attempts=opts.revisit_attempts)
        if not ok:
            node.get_logger().warn(
                f"Phase 2 candidate {cand_idx + 1}/{len(candidates)}: couldn't "
                f"revisit candidate from phase 1 step {cand['step'] + 1}, skipping"
            )
            continue

        if feedback_cb:
            feedback_cb("phase2",
                        f"investigating candidate {cand_idx + 1}/{len(candidates)}: "
                        f"{cand.get('label', '?')}", step)

        # Last successfully executed move, so a "lost it" try can back
        # off rather than guess a fresh direction (see no-bbox branch).
        last_action, last_amount = None, None
        last_bbox = cand.get("bbox")
        confirmed = False

        for try_idx in range(opts.max_tries):
            if step >= opts.phase2_max_steps:
                break
            step += 1

            image = node.capture_image(timeout=5.0)
            if image is None:
                node.get_logger().warn(f"Phase 2 step {step}: no image received")
                continue

            detections = detect_objects(image, {
                "step": f"p2_c{cand_idx}_t{try_idx}",
                "yolo_weights": opts.yolo_weights, "target_labels": [cand["label"]],
                "save_detections_dir": os.path.join(opts.save_detections_dir, "phase2")
                                       if opts.save_detections_dir else None,
                "save_always": True,
            })
            best = select_tracked_detection(detections, last_bbox)
            if best is not None:
                last_bbox = best["bbox"]
            best_conf = best["confidence"] if best else 0.0

            node.get_logger().debug(
                f"Phase 2 candidate {cand_idx + 1}/{len(candidates)}, try "
                f"{try_idx + 1}/{opts.max_tries} (step {step}): YOLO best "
                f"confidence={best_conf:.2f}"
            )
            if feedback_cb:
                feedback_cb("phase2",
                            f"candidate {cand_idx + 1}/{len(candidates)}, try "
                            f"{try_idx + 1}/{opts.max_tries}: confidence={best_conf:.2f}",
                            step)

            if best_conf >= DETECTION_CONFIDENCE_THRESHOLD:
                obj_xyz = back_project_object_position(node, best["bbox"])
                if obj_xyz is None:
                    obj_xyz, _ = node.get_camera_pose(timeout=2.0)
                position = obj_xyz if obj_xyz is not None else cand["position_base_link"]
                loc = tuple(round(v, 3) for v in position)
                node.get_logger().info(
                    f"Phase 2 FOUND: {best.get('label', '?')} "
                    f"(conf={best_conf:.2f}) at {loc}"
                )
                if feedback_cb:
                    feedback_cb("phase2",
                                f"FOUND {best.get('label', '?')} "
                                f"(conf={best_conf:.2f}) at {loc}", step)
                found.append({
                    "found": True, "confidence": best_conf,
                    "description": best.get("description", ""),
                    "label": best.get("label", "?"),
                    "position_base_link": position,
                })
                confirmed = True
                break

            if try_idx < opts.max_tries - 1:
                # Not the last try - get a better look before the next
                # YOLO re-check.
                depth_move_done = False
                if best is not None:
                    # Back-project the bbox center through the depth
                    # cloud for a precise 3D move to a fixed standoff,
                    # instead of a qualitative pan/tilt/forward guess.
                    cx = int((best["bbox"][0] + best["bbox"][2]) / 2)
                    cy = int((best["bbox"][1] + best["bbox"][3]) / 2)
                    points_msg = node.capture_points(timeout=2.0)
                    point_cam = get_point_from_cloud(points_msg, cx, cy)
                    camera_xyz, camera_quat = (
                        node.get_camera_pose(timeout=2.0) if point_cam else (None, None)
                    )
                    if point_cam is not None and camera_xyz is not None:
                        object_xyz = transform_point_to_base_link(
                            point_cam, camera_xyz, camera_quat)
                        new_xyz, new_quat = compute_3d_approach_pose(
                            object_xyz, camera_xyz, opts.approach_standoff,
                            max_step=opts.approach_max_step)
                        dist = math.sqrt(sum((o - c) ** 2 for o, c in
                                              zip(object_xyz, camera_xyz)))
                        node.get_logger().info(
                            f"Phase 2 candidate {cand_idx + 1}, try {try_idx + 1}: "
                            f"3D approach - object at "
                            f"{tuple(round(v, 3) for v in object_xyz)} in base_link, "
                            f"~{dist:.2f}m away"
                        )
                        depth_move_done = node.move_to_pose(
                            new_xyz, new_quat, opts.group, opts.vel_scale, opts.accel_scale)
                        if depth_move_done:
                            # Absolute move, no simple reverse for a
                            # later backoff try - don't track it as one.
                            last_action, last_amount = None, None
                        else:
                            node.get_logger().warn(
                                f"Phase 2 candidate {cand_idx + 1}: 3D approach move "
                                f"failed, falling back to VLM"
                            )

                if depth_move_done:
                    pass
                elif best is not None:
                    # No usable depth this try - fall back to the VLM,
                    # which can still reason about the bbox qualitatively.
                    decision = vlm_guide(image, {
                        "step": step,
                        "investigating_candidate": cand,
                        "qwen_model": opts.qwen_model,
                        "detector_bbox": best["bbox"],
                    })
                    action, amount = decision["action"], decision["amount"]
                    reason = f"VLM move - {decision['description']}"
                else:
                    # Nothing detected - deterministic recovery instead
                    # of asking the VLM, which has nothing to ground on.
                    # Back off by reversing the last move at half
                    # magnitude; with no prior move, nudge forward.
                    if last_action is not None:
                        action = REVERSE_ACTION[last_action]
                        amount = last_amount * 0.5
                        reason = (f"deterministic backoff (reverse of "
                                  f"{last_action}) - lost sight of it")
                    else:
                        action, amount = "forward", 0.1
                        reason = "deterministic default - no detection yet, nudging closer"

                if not depth_move_done:
                    node.get_logger().info(
                        f"Phase 2 candidate {cand_idx + 1}, try {try_idx + 1}: "
                        f"{action}({amount:.3f}) - {reason}"
                    )
                    moved = node.move(action, amount, opts.group,
                                       opts.vel_scale, opts.accel_scale)
                    if moved:
                        last_action, last_amount = action, amount
                    else:
                        node.get_logger().warn(
                            f"Phase 2 candidate {cand_idx + 1}: move "
                            f"{action} failed, retrying YOLO from same pose"
                        )
            time.sleep(opts.dwell)
        else:
            if not confirmed:
                node.get_logger().debug(
                    f"Phase 2 candidate {cand_idx + 1}/{len(candidates)}: gave up "
                    f"after {opts.max_tries} tries, moving to next candidate"
                )

    node.get_logger().debug(
        f"Phase 2 complete: {step} step(s) used across {len(candidates)} "
        f"candidate(s), {len(found)} confirmed"
    )
    return found


def run_full_search(node, opts, feedback_cb=None):
    """End-to-end two-phase search: home -> phase 1 sweep -> phase 2
    investigation -> home. Shared by the CLI entry point (main(), below)
    and the FindObject action server (object_search_action_server.py) so
    both drive the exact same search behavior. feedback_cb, if given, is
    threaded through to run_phase1_helix/run_phase2_vlm - see their
    docstrings; it's a no-op for plain CLI use.

    Returns the list of found detection dicts (each with a
    "position_base_link"), possibly empty. Does not init/shutdown rclpy
    or touch the node's lifecycle - callers own that."""
    helix_plan = generate_helix_plan(
        opts.helix_radius, opts.helix_height,
        opts.tilt_up_deg, opts.tilt_down_deg,
        opts.num_sectors, opts.tilts_per_sector,
        center_xy=tuple(opts.helix_center),
    )

    if not opts.skip_start_home:
        node.get_logger().debug("Moving to home position before starting")
        if not node.go_home(opts.group, opts.vel_scale, opts.accel_scale):
            node.get_logger().warn("Failed to reach home position before starting")

    found = []
    candidates = []
    if not opts.skip_phase1:
        found, candidates = run_phase1_helix(node, helix_plan, opts, feedback_cb=feedback_cb)

    if not opts.skip_phase2:
        found += run_phase2_vlm(node, opts, candidates, feedback_cb=feedback_cb)

    node.publish_found_markers(found)

    if not opts.skip_return_home:
        node.get_logger().debug("Returning to home position")
        if not node.go_home(opts.group, opts.vel_scale, opts.accel_scale):
            node.get_logger().warn("Failed to return to home position")

    return found


def main(args=None):
    rclpy.init(args=args)
    argv = rclpy.utilities.remove_ros_args(args=sys.argv)[1:]
    opts = parse_args(argv)

    if opts.save_detections_dir:
        # Scope saved images to this run - a fresh timestamped subfolder
        # each invocation, so runs don't mix in one flat directory.
        run_id = time.strftime("%Y%m%d_%H%M%S")
        opts.save_detections_dir = os.path.join(opts.save_detections_dir, run_id)
        print(f"Saving detection images to {opts.save_detections_dir}")

    helix_plan = generate_helix_plan(
        opts.helix_radius, opts.helix_height,
        opts.tilt_up_deg, opts.tilt_down_deg,
        opts.num_sectors, opts.tilts_per_sector,
        center_xy=tuple(opts.helix_center),
    )
    print(f"Phase 1 sweep plan: {len(helix_plan)} waypoints "
          f"({opts.num_sectors} sectors x {opts.tilts_per_sector} tilts, "
          f"radius {opts.helix_radius}m, height {opts.helix_height}m, "
          f"tilt up {opts.tilt_up_deg}deg / down {opts.tilt_down_deg}deg)")

    if opts.dry_run:
        for i, (xyz, quat) in enumerate(helix_plan):
            print(f"  {i + 1:3d}. xyz=({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}) "
                  f"quat=({quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f})")
        print(f"Phase 2 (VLM-guided): up to {opts.phase2_max_steps} steps, not enumerable")
        rclpy.shutdown()
        return

    node = EnvironmentScanner(opts.image_topic, opts.points_topic)

    print("Waiting for /compute_ik and /move_action...")
    if not node.wait_for_servers(30.0):
        node.get_logger().error("MoveIt servers not available")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    found = run_full_search(node, opts)

    # Search always completes (no early stop on a find) - report every
    # confirmed object found, not just the first.
    if found:
        node.get_logger().info(f"Scan complete: {len(found)} object(s) found")
        for f in found:
            loc = tuple(round(v, 3) for v in f["position_base_link"])
            node.get_logger().info(
                f"  {f['label']} (conf={f['confidence']:.2f}) at {loc}"
            )
    else:
        node.get_logger().info(
            f"Scan complete, nothing found within {opts.max_object_distance}m"
        )

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
