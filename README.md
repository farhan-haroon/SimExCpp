# SimExCpp — VLM+YOLO-Guided Object Search for a Husky+UR10 Mobile Manipulator

A ROS 2 workspace simulating a **Clearpath Husky** carrying a **Universal
Robots UR10** arm with a wrist-mounted RGB-D camera. The arm sweeps its
surroundings looking for target objects (YOLOv8 + Qwen2.5-VL), and a
spanning-tree coverage planner drives the base around the map, triggering a
search each time it enters a new area. Every confirmed find gets an RViz
marker at its real 3D position.

This README is a step-by-step path to reproducing that from a clean machine.

---

## Contents

- [Quickstart](#quickstart)
- [CLI reference](#cli-reference)
- [How it works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [Repository layout](#repository-layout)

---

## Quickstart

### 1. Prerequisites

- Ubuntu 22.04, **ROS 2 Humble**, **Gazebo Classic 11**, **MoveIt 2**, **Nav2**
- An NVIDIA GPU for the VLM (Qwen2.5-VL-7B in bf16 is ~16.6 GB; a single
  24 GB+ GPU works, or it shards across multiple smaller ones). YOLOv8n runs
  fine on CPU or GPU.
- Python 3.10 (Humble's default)

### 2. Clone and install ROS dependencies

```bash
mkdir -p ~/husky_ws/src && cd ~/husky_ws/src
git clone https://github.com/farhan-haroon/SimExCpp.git .

cd ~/husky_ws
rosdep install --from-paths src --ignore-src -r -y
```

`rosdep` pulls in `ur_description`, `moveit_kinematics`, `nav2_bringup`, etc.
— make sure `ros-humble-ur`, `ros-humble-moveit`, `ros-humble-navigation2`
apt sources are enabled first.

### 3. Install Python/ML dependencies

The search scripts run as plain `ros2 run` executables, so their
dependencies are ordinary `pip` installs into whatever interpreter `ros2 run`
uses:

```bash
pip install ultralytics torch transformers accelerate opencv-python cv_bridge
```

(`torch` — pick the build matching your CUDA version from pytorch.org.
`cv_bridge` usually already comes from `ros-humble-cv-bridge`.)

### 4. Get the models

```bash
# YOLOv8n — auto-downloads, then move it to where the script expects it
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
mv yolov8n.pt ~/husky_ws/yolov8n.pt

# Qwen2.5-VL-7B-Instruct — must be pre-fetched, the script loads it offline
pip install -U "huggingface_hub[cli]"
hf download Qwen/Qwen2.5-VL-7B-Instruct
```

### 5. Build

```bash
cd ~/husky_ws
colcon build --symlink-install
source install/setup.bash
```

### 6. Launch the simulation

```bash
ros2 launch husky_ur10_cam husky_ur10_moveit_office_large.launch.py
```

Wait ~10–15 seconds for Gazebo and all controllers to come up. (Two other
worlds are available: `husky_ur10_moveit_small_house.launch.py`,
`husky_ur10_moveit_agriculture.launch.py` — the office world is the one
tuned/tested against.)

### 7. Run a search

In a new terminal (sourced):

```bash
ros2 run husky_ur10_cam ee_camera_environment_scan.py --objects chair
```

The arm sweeps its surroundings, investigates anything promising, and logs
each confirmed find:

```
[INFO] Phase 1 FOUND: chair (conf=0.91) at (0.82, 1.14, 0.61)
[INFO] Scan complete. Found 1 object(s):
[INFO]   - chair (conf=0.91) at (0.82, 1.14, 0.61)
[INFO] Published 1 found-object marker(s) in map frame
```

### 8. (Optional) Full map coverage + search

Needs localization/navigation running too:

```bash
ros2 launch nav2_bringup localization_launch.py map:=/path/to/map.yaml
ros2 launch nav2_bringup navigation_launch.py use_sim_time:=true

# Action server the coverage node calls into
ros2 run husky_ur10_cam object_search_action_server.py

# Drives the base around the whole map, searching as it goes
ros2 launch stc_cpp stc.launch.py search_objects:="chair,book,laptop" subcell_per_cell:=4
```

Add a `MarkerArray` RViz display on `/found_objects_markers` (Fixed Frame =
`map`) to watch finds accumulate as the robot covers the map.

---

## CLI reference

`ee_camera_environment_scan.py` (also usable as the `find_object` action's
goal parameters):

| Flag | Default | Description |
|---|---|---|
| `--objects` | `chair` | Target class(es), space-separated, e.g. `--objects chair book laptop`. Must be YOLO/COCO classes. |
| `--helix-radius` / `--helix-height` | `0.6` / `1.0` | Sweep orbit radius/height around `base_link` (m). |
| `--helix-center` | `0.0 0.0` | Orbit center `(x, y)` in `base_link`. |
| `--tilt-up-deg` / `--tilt-down-deg` | `5.0` / `30.0` | Pitch sweep range per azimuth. |
| `--num-sectors` | `16` | Azimuths around the robot. Raise if you widen the camera FOV. |
| `--tilts-per-sector` | `2` | Tilt waypoints per azimuth. |
| `--yolo-weights` | `~/husky_ws/yolov8n.pt` | Path to YOLO weights. |
| `--qwen-model` | `Qwen/Qwen2.5-VL-7B-Instruct` | HF model id/path for phase 2 guidance. |
| `--max-tries` | `5` | Per-candidate attempt budget before giving up on it. |
| `--max-object-distance` | `3.0` | Candidates farther than this (depth-measured, m) are skipped. |
| `--approach-standoff` / `--approach-max-step` | `0.15` / `0.3` | 3D approach standoff distance / max step per move (m). |
| `--save-detections-dir` | `~/husky_ws/detections` | Where annotated detection images are saved (empty disables). |
| `--skip-phase1` / `--skip-phase2` | off | Skip a phase entirely. |
| `--skip-start-home` / `--skip-return-home` | off | Skip the home move before/after the scan. |
| `--dry-run` | off | Print the phase 1 waypoints without moving the robot. |

Full list: `ros2 run husky_ur10_cam ee_camera_environment_scan.py --help`

`stc.launch.py` args:

| Arg | Default | Description |
|---|---|---|
| `search_objects` | `chair` | Comma-separated target class(es). |
| `search_max_object_distance` | `0.0` | `FindObject` goal's `max_object_distance`; `<= 0.0` uses the server default. |
| `subcell_per_cell` | `2` | Major-cell size as a multiple of the robot footprint. Raise for fewer, coarser search stops. |

---

## How it works

**Phase 1 — fixed sweep.** The arm holds a fixed orbit position and steps
through `--num-sectors` azimuths x `--tilts-per-sector` tilt angles, running
YOLO at each stop. A confident hit is recorded as found; a weaker one is
saved as a candidate for phase 2. The sweep always runs to completion, so
multiple objects can be found in one pass.

**Phase 2 — investigate candidates.** For each saved candidate, YOLO
re-checks the view; if still unconfirmed, the camera either moves precisely
to a fixed standoff from the object (via depth back-projection) or, if depth
isn't available, Qwen2.5-VL picks a qualitative pan/tilt/forward move. No
open-ended search — only ever refines a candidate phase 1 already flagged.

**STC coverage integration.** `stc_cpp`'s coverage node acts as a
`FindObject` action client: each time it enters a new major cell, it pauses,
sends a search goal, and resumes coverage once the search returns (or logs a
warning and continues if the action server isn't reachable).

**Markers.** Every confirmed find publishes a labeled sphere on
`/found_objects_markers` in the `map` frame (needs a live `map -> base_link`
TF, i.e. localization running).

---

## Troubleshooting

- **Nothing found within `--max-object-distance`:** try raising it, or
  `--tilt-down-deg`/`--num-sectors` for wider coverage, or check the saved
  images in `--save-detections-dir` to see what the sweep actually captured.
- **YOLO never detects your target:** it must be one of YOLO's 80 COCO
  classes (`chair`, `book`, `bottle`, `laptop`, `keyboard`, `person`, ...).
  Classes outside COCO (e.g. "camera") can't be detected without custom
  training.
- **Qwen fails to load / OOM:** confirm it was pre-downloaded (`hf download
  ...`) and `accelerate` is installed for `device_map="auto"` sharding.
- **`/compute_ik` or `/move_action` never come up:** give `move_group` a few
  seconds after launch; if it hangs, check `ros2 node list` for a failed
  Gazebo spawn or controller further up the launch chain.
- **STC coverage never pauses to search:** make sure
  `object_search_action_server.py` is running — if `find_object` isn't
  reachable within 5s, `stc.py` logs a warning and continues without
  searching rather than blocking forever.
- **No markers in RViz:** needs a live `map -> base_link` TF (localization
  running). Check the `MarkerArray` display's topic (`/found_objects_markers`)
  and Fixed Frame (`map`).

---

## Repository layout

```
husky_ws/src/
├── husky_ur10_cam/            # Robot description, worlds, launch files, search scripts
│   ├── urdf/                  #   Husky+UR10+camera URDF/xacro
│   ├── worlds/                #   Gazebo worlds (office, small house, agriculture, empty)
│   ├── launch/                #   Sim launch files
│   ├── config/                #   ros2_control / controller yaml
│   ├── models/                #   Gazebo models used to populate the worlds
│   └── scripts/                #   Object search (CLI + action server) + IK tools
├── husky_ur10_moveit_config/  # MoveIt2 config for the UR10
├── pointcloud_concatenate_ros2/  # Merges point clouds (feeds laserscan generation)
├── pointcloud_to_laserscan/   # Point cloud -> 2D LaserScan for Husky navigation
├── custom_interfaces/         # FindObject action, DetectedObject message
└── stc_cpp/                   # Spanning-tree-coverage planner
```

| Package | Role |
|---|---|
| `husky_ur10_cam` | Robot description, Gazebo worlds, sim launch files, object-search scripts. Start here. |
| `husky_ur10_moveit_config` | MoveIt2 configuration for the UR10 arm. |
| `custom_interfaces` | `FindObject` action and `DetectedObject` message. |
| `stc_cpp` | Coverage path planner; drives the base and calls `FindObject` per cell. |
| `pointcloud_concatenate_ros2` | Merges point cloud sources into one fused cloud. |
| `pointcloud_to_laserscan` | Converts the fused cloud into a 2D `/scan` for navigation. |

| Script (`husky_ur10_cam/scripts/`) | Purpose |
|---|---|
| `ee_camera_environment_scan.py` | Two-phase search logic + standalone CLI. |
| `object_search_action_server.py` | Same logic as a `FindObject` action server. |
| `test_ee_pose_ik.py` | Command the camera to an arbitrary pose via `/compute_ik` (used to derive `HOME_JOINTS`). |
| `preset_views.py` | Cycles the arm through a fixed list of preset joint poses. |
