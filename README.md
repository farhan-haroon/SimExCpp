# SimExCpp — VLM+YOLO-Guided Object Search for a Husky+UR10 Mobile Manipulator

A ROS 2 workspace that simulates a **Clearpath Husky** carrying a **Universal
Robots UR10** arm, fitted with a wrist-mounted RGB-D camera. It combines two
pieces:

1. **Object search** — a two-phase YOLO+VLM-guided arm routine that looks
   for one or more target objects from wherever the base currently is, and
   is exposed as both a standalone CLI script and a ROS 2 action.
2. **STC coverage** — a spanning-tree-coverage planner that drives the base
   around the whole map, and calls the object search action every time it
   enters a new coverage cell, so the robot searches as it explores.

Every confirmed find gets an RViz marker dropped at its real (depth
back-projected) position in the `map` frame.

This README documents the whole workspace and gives a step-by-step path to
reproducing it from a clean machine.

---

## Table of contents

- [Repository layout](#repository-layout)
- [How the object search works](#how-the-object-search-works)
- [The FindObject action](#the-findobject-action)
- [STC coverage + object search integration](#stc-coverage--object-search-integration)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
  - [1. ROS 2 workspace](#1-ros-2-workspace)
  - [2. Python/ML dependencies](#2-pythonml-dependencies)
  - [3. Models](#3-models)
- [Building](#building)
- [Running the simulation](#running-the-simulation)
- [Running the object search](#running-the-object-search)
  - [Standalone CLI](#standalone-cli)
  - [CLI reference](#cli-reference)
  - [As a ROS 2 action](#as-a-ros-2-action)
  - [Reading the output](#reading-the-output)
- [Running full coverage + search](#running-full-coverage--search)
- [Other useful scripts](#other-useful-scripts)
- [Camera configuration notes](#camera-configuration-notes)
- [Troubleshooting](#troubleshooting)
- [Package overview](#package-overview)

---

## Repository layout

```
husky_ws/src/
├── husky_ur10_cam/            # Robot description, worlds, launch files, search scripts
│   ├── urdf/                  #   Husky+UR10+camera URDF/xacro (incl. Gazebo sensor plugins)
│   ├── worlds/                #   Gazebo Classic .world files (office, small house, agriculture, empty)
│   ├── launch/                #   Top-level sim launch files (spawn robot, controllers, MoveIt, etc.)
│   ├── config/                #   ros2_control / controller yaml, initial joint positions
│   ├── models/                #   Gazebo models used to populate the worlds
│   └── scripts/                #   Python: IK testing + object search (CLI + action server)
├── husky_ur10_moveit_config/  # MoveIt2 config (SRDF, kinematics, OMPL planning, controllers) for the UR10
├── pointcloud_concatenate_ros2/  # Merges multiple point clouds into one (used for laserscan generation)
├── pointcloud_to_laserscan/   # Point cloud → 2D LaserScan conversion (feeds Husky navigation)
├── custom_interfaces/         # Custom message + action definitions (DetectedObject, FindObject)
└── stc_cpp/                   # Spanning-tree-coverage planner, calls FindObject per coverage cell
    └── launch/                #   stc.launch.py (search_objects, subcell_per_cell, ... launch args)
```

The object-search work lives in **`husky_ur10_cam/scripts/`**:

| File | Purpose |
|---|---|
| `ee_camera_environment_scan.py` | The two-phase search logic (`run_full_search`), plus a standalone CLI entry point. |
| `object_search_action_server.py` | Wraps the same search logic as a `custom_interfaces/action/FindObject` action server. |
| `test_ee_pose_ik.py` | Standalone IK/pose-commanding tool used to test and solve fixed end-effector poses (e.g. how `HOME_JOINTS` was derived). |
| `preset_views.py` | Drives the arm through a list of fixed preset joint poses. |

---

## How the object search works

### Phase 1 — fixed pan/tilt sweep

- The 360° around the robot is divided into `--num-sectors` azimuths
  (default **16**, i.e. every 22.5°/±8° step), tuned to give ~60%
  frame-to-frame overlap given the camera's ~57° horizontal FOV.
- At each azimuth the arm holds a fixed orbit position (`--helix-radius`,
  `--helix-height` around `--helix-center`) and tilts through
  `--tilts-per-sector` pitch angles between `--tilt-up-deg` and
  `--tilt-down-deg`. The camera always faces radially outward and stays
  perfectly level (0 roll/pitch) via a `look_at_quaternion()` helper — this
  matters for image quality fed to both YOLO and the VLM.
- **YOLOv8** runs once per waypoint, checking for *any* of `--objects` (one
  or more target classes) in a single pass:
  - confidence ≥ 80% (fixed threshold, not configurable) → logged as
    **FOUND** immediately, but the sweep *keeps going* through every
    remaining waypoint.
  - confidence > 0 but below threshold, and within `--max-object-distance`
    (measured via the depth camera) → saved as a **candidate** for phase 2.
- Before the sweep starts (and after it ends), the arm moves to a fixed,
  hardcoded joint-space `HOME_JOINTS` pose — not re-solved via IK each
  time, so it lands in the exact same configuration every run.

### Phase 2 — investigate phase 1 candidates only

For each saved candidate, in order, up to `--max-tries` (default 5) times,
checked against **that candidate's own class only** (so a chair candidate
is never confused by a book detection sharing the frame):

1. **YOLO re-checks** the current view. Confidence ≥ 80% → candidate
   confirmed, recorded, move on to the next candidate.
2. Otherwise, get a better look:
   - **Depth-based 3D approach (preferred):** back-project the detected
     bounding box's center pixel through the depth camera's point cloud to
     get the object's real 3D position, transform it to `base_link` via
     the camera's live TF pose, and move directly to a fixed
     `--approach-standoff` distance from it along the line of sight — a
     single precise move rather than a qualitative guess (step size capped
     by `--approach-max-step` so it converges gradually rather than
     attempting one large possibly-unreachable jump).
   - **VLM fallback:** if depth is invalid at that pixel (common at object
     edges/thin objects) or the move fails, Qwen2.5-VL-7B-Instruct looks at
     the bounding box and picks one qualitative move
     (`pan_left`/`pan_right`/`tilt_up`/`tilt_down`/`forward`).
   - **Deterministic backoff:** if YOLO loses the object entirely (no
     detection that frame), the script reverses the last move at half
     magnitude rather than asking the VLM to guess blind — this avoids a
     monotonic drift bug where a small VLM's fixed-direction guess would
     compound across tries and sweep the camera into a different room.
   - Within a frame with **multiple detections of the same class**, the
     detection nearest (by bbox-center proximity) to the candidate's
     last-known bbox is tracked — not simply the highest-confidence one —
     so investigation doesn't jump to a different, already-confirmed
     instance of the same class.
3. If `--max-tries` is exhausted without confirming, the candidate is given
   up on and the next one is tried. There is no open-ended exploration —
   every phase 2 move only ever gets the camera closer to a *known*
   candidate.

### Reporting a find

The scan **never stops early**: it completes the full sweep and
investigates every candidate, then reports every confirmed object at the
end. A found object's position is **not** the camera's pose — it's the
object's own 3D position, back-projected through the depth cloud at the
confirming detection's bounding box (falling back to the camera pose only
if depth is unavailable right at that pixel). Each one also gets an RViz
marker (see below), then the arm returns home.

---

## The FindObject action

`custom_interfaces/action/FindObject` wraps the same search logic
(`run_full_search`) as a ROS 2 action, served by
`object_search_action_server.py`:

```
# Goal
string[] target_objects                # one or more YOLO/COCO class names, e.g. ["chair", "book"]
float32 max_object_distance 0.0        # meters; <= 0.0 means "use server default"
---
# Result
bool success                           # true if at least one object was confirmed
DetectedObject[] found_objects         # label, confidence, position (base_link), description
string message
---
# Feedback
string phase                           # "phase1" | "phase2"
string status
int32 step
```

The action server runs the goal's `execute_callback` in its own dedicated
callback group, separate from the inherited image/points/IK/MoveGroup
callbacks, so the search's internal blocking waits keep working correctly
without needing a `MultiThreadedExecutor`. One goal at a time; cancellation
isn't supported (the search isn't structured to check for it mid-step).

### RViz markers

Every confirmed find (from either the CLI or the action server, since both
go through `run_full_search`) gets published as a labeled sphere +
text-label `Marker` on `/found_objects_markers` (`map` frame, transient-local
QoS so RViz picks them up even if the display is added after the fact). The
object's `base_link` position is transformed to `map` via TF (the
`map → odom → base_link` chain AMCL/odometry already broadcast) — this only
works if localization is actually running; otherwise it logs a warning and
skips publishing rather than dropping a wrong marker.

Add a `MarkerArray` display in RViz subscribed to `/found_objects_markers`
with Fixed Frame = `map` to see them.

---

## STC coverage + object search integration

`stc_cpp`'s `kruskal_stc_node` (`stc.py`) drives the base around a
Kruskal-MST spanning-tree coverage path over the map's free space. It now
also acts as a `FindObject` action **client**: every time the coverage path
crosses into a new major cell (different from the last one it searched),
it pauses forward progress, sends a `FindObject` goal for
`search_objects`, and only advances to the next waypoint once that search
finishes (or the action server isn't reachable — see below).

- If the `object_search_action_server.py` isn't running, it logs a warning
  and continues coverage without pausing, rather than blocking the robot
  forever.
- The major-cell grid size is `subcell_per_cell × subcell_size`
  (`subcell_size` is fixed to the robot's footprint for fine-grained
  coverage resolution); `subcell_per_cell` (default `2`) is a launch
  parameter, so you can make the search-trigger grid coarser or finer
  without touching the robot's actual step size.

---

## Prerequisites

- **Ubuntu 22.04** with **ROS 2 Humble**
- **Gazebo Classic 11** (`gazebo_ros_pkgs`)
- **MoveIt 2** (Humble binaries)
- **Nav2** (`amcl` + `map_server`/localization, for the `map` frame TF and
  for STC coverage's `NavigateToPose` calls)
- An **NVIDIA GPU** for the VLM. Qwen2.5-VL-7B-Instruct in bf16 is ~16.6 GB
  of weights, which does *not* fit comfortably on a single 16 GB GPU — the
  script loads it with `device_map="auto"` so it will shard across multiple
  GPUs if available. A single 24 GB+ GPU also works. YOLOv8n is tiny and
  runs fine on CPU or GPU.
- Python 3.10 (matches Humble's default interpreter)

## Setup

### 1. ROS 2 workspace

```bash
# Install ROS 2 Humble desktop + Gazebo Classic + MoveIt2 + Nav2 first (apt), then:
mkdir -p ~/husky_ws/src
cd ~/husky_ws/src
git clone https://github.com/farhan-haroon/SimExCpp.git .

# ROS package dependencies (rosdep)
cd ~/husky_ws
rosdep install --from-paths src --ignore-src -r -y
```

You'll also need the upstream Husky and UR description/driver packages that
this repo's URDFs and MoveIt config build on top of (`ur_description`,
`ur_msgs`, `moveit_kinematics`, `moveit_planners_ompl`, `moveit_servo`,
`moveit_simple_controller_manager`, `warehouse_ros_sqlite`, `nav2_bringup`,
etc. — all pulled in by `rosdep` above, provided the corresponding apt
sources are enabled: `ros-humble-ur`, `ros-humble-moveit`,
`ros-humble-navigation2`, `ros-humble-clearpath-*` or equivalent).

### 2. Python/ML dependencies

The object-search scripts run as plain `ros2 run` executables (not Python
package nodes), so their dependencies are ordinary `pip` installs into
whatever interpreter `ros2 run` uses (the system Python 3.10, unless you're
using a venv that ROS 2 has been sourced into):

```bash
pip install ultralytics          # YOLOv8
pip install torch                # match your CUDA version, see pytorch.org
pip install transformers         # Qwen2.5-VL support
pip install accelerate           # needed for device_map="auto" sharding
pip install opencv-python
pip install cv_bridge            # usually already provided by ros-humble-cv-bridge (apt)
```

Versions this was developed/tested against: `ultralytics==8.4.96`,
`transformers==5.14.1`, `torch==2.13.0` (CUDA-enabled build).

### 3. Models

**YOLOv8n weights** — the script defaults to `~/husky_ws/yolov8n.pt`:

```bash
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"  # auto-downloads
mv yolov8n.pt ~/husky_ws/yolov8n.pt
```

(Or point `--yolo-weights` at wherever you keep it. Detection is limited to
YOLO's 80 COCO classes — e.g. `chair`, `book`, `bottle`, `laptop`, `keyboard`,
`person`; it **cannot** detect classes outside COCO, such as "camera".)

**Qwen2.5-VL-7B-Instruct** — downloaded from Hugging Face and loaded with
`local_files_only=True`, so it must be pre-fetched into the HF cache before
running:

```bash
pip install -U "huggingface_hub[cli]"
hf download Qwen/Qwen2.5-VL-7B-Instruct
```

This pulls ~16.6 GB into `~/.cache/huggingface/hub/`. A smaller model can be
substituted via `--qwen-model` (e.g. `Qwen/Qwen2.5-VL-3B-Instruct`), but note
that in testing the 3B model's spatial reasoning (particularly vertical /
centered / unseen judgments) was unreliable — 7B is recommended.

---

## Building

```bash
cd ~/husky_ws
colcon build --symlink-install
source install/setup.bash
```

`--symlink-install` matters here: the Python scripts in
`husky_ur10_cam/scripts/` are installed as symlinks, so they can be edited
in place and re-run with `ros2 run` without a rebuild. Only new files or
`CMakeLists.txt`/`setup.py` changes require a `colcon build`.

---

## Running the simulation

Three pre-built worlds are provided; the office world is the one the search
scripts have been tuned/tested against:

```bash
# Large office (has chairs, books, keyboards, monitors, etc. — recommended)
ros2 launch husky_ur10_cam husky_ur10_moveit_office_large.launch.py

# Small house
ros2 launch husky_ur10_cam husky_ur10_moveit_small_house.launch.py

# Agriculture field
ros2 launch husky_ur10_cam husky_ur10_moveit_agriculture.launch.py
```

Each of these brings up, in order: `gzserver`/`gzclient` with the chosen
world, robot state publisher, the Husky spawn (`ros2_control` diff-drive +
joint-state-broadcaster + UR10 joint-trajectory controller), point cloud
concatenation + laserscan conversion, and MoveIt2 (`move_group` +
RViz) — with staggered `TimerAction` delays so Gazebo has finished spawning
the robot before controllers/MoveIt come up.

Give it **10–15 seconds** after launch for every controller to come up
before commanding the arm — the search script waits for `/compute_ik` and
the `/move_action` action server internally, but Gazebo/gzserver itself can
take a few seconds longer to settle.

If you want `map`-frame markers or STC coverage, also bring up localization
(e.g. `ros2 launch nav2_bringup localization_launch.py map:=/path/to/map.yaml`)
and navigation (`ros2 launch nav2_bringup navigation_launch.py use_sim_time:=true`).

---

## Running the object search

### Standalone CLI

With the sim up and workspace sourced in another terminal:

```bash
ros2 run husky_ur10_cam ee_camera_environment_scan.py --objects chair
```

### CLI reference

| Flag | Default | Description |
|---|---|---|
| `--objects` | `chair` | One or more target classes (space-separated), e.g. `--objects chair book laptop`. Each must be a YOLO/COCO class. Phase 1 sweeps once checking for all of them; phase 2 investigates each candidate as its own specific class. |
| `--helix-radius` | `0.6` | Phase 1 sweep orbit radius around `base_link` (m). |
| `--helix-height` | `1.0` | Phase 1 sweep orbit height (m). |
| `--helix-center` | `0.0 0.0` | Orbit center `(x, y)` in `base_link`. |
| `--tilt-up-deg` / `--tilt-down-deg` | `5.0` / `30.0` | Pitch sweep range per azimuth. |
| `--num-sectors` | `16` | Azimuths around the robot. Raise this if you narrow the camera FOV further, to keep frame overlap. |
| `--tilts-per-sector` | `2` | Tilt waypoints per azimuth. |
| `--yolo-weights` | `~/husky_ws/yolov8n.pt` | Path to YOLO weights. |
| `--qwen-model` | `Qwen/Qwen2.5-VL-7B-Instruct` | HF model id/path for phase 2 move guidance. |
| `--max-tries` | `5` | Per-candidate budget (YOLO re-checks + moves) before giving up on it. |
| `--revisit-attempts` | `3` | Retries (+ recovery-pose fallback) when moving back to a saved candidate pose. |
| `--phase2-max-steps` | `20` | Overall cap on total phase 2 steps across all candidates. |
| `--max-object-distance` | `3.0` | Candidates farther than this (depth-measured, meters) are skipped. Unmeasurable-depth candidates are kept. |
| `--approach-standoff` | `0.15` | Stop this many meters short of the object during the 3D depth approach. |
| `--approach-max-step` | `0.3` | Cap on a single 3D approach move (m), so it converges gradually instead of jumping. |
| `--save-detections-dir` | `~/husky_ws/detections` | Where bounding-box-overlaid images are saved (empty string disables). Each run gets its own timestamped subfolder, split into `phase1/`/`phase2/`. |
| `--skip-phase1` / `--skip-phase2` | off | Skip a phase entirely. |
| `--skip-start-home` / `--skip-return-home` | off | Skip the home move before/after the scan. |
| `--dwell` | `0.5` | Seconds to settle before each image capture. |
| `--group` | `ur10_ur_manipulator` | MoveIt planning group. |
| `--vel-scale` / `--accel-scale` | `1.0` / `1.0` | Velocity/acceleration scaling for all moves. |
| `--image-topic` | `/ee_camera/image_raw` | RGB source topic. |
| `--points-topic` | derived from `--image-topic` | Organized point cloud topic (depth). |
| `--dry-run` | off | Print the phase 1 waypoints without commanding the robot. |

Example runs:

```bash
# Sanity-check the sweep plan without moving the robot
ros2 run husky_ur10_cam ee_camera_environment_scan.py --dry-run

# Search for a book and a laptop, save every frame for review
ros2 run husky_ur10_cam ee_camera_environment_scan.py --objects book laptop \
    --save-detections-dir ~/husky_ws/detections

# Only run phase 2 investigation moves logic in isolation / skip the sweep
ros2 run husky_ur10_cam ee_camera_environment_scan.py --objects bottle --skip-phase1

# Tighter search radius, closer object distance cap
ros2 run husky_ur10_cam ee_camera_environment_scan.py --objects chair \
    --helix-radius 0.5 --max-object-distance 2.0
```

### As a ROS 2 action

```bash
ros2 run husky_ur10_cam object_search_action_server.py
```

Then send a goal, e.g. from another terminal:

```bash
ros2 action send_goal /find_object custom_interfaces/action/FindObject \
    "{target_objects: ['chair', 'book'], max_object_distance: 2.5}" --feedback
```

This is what `stc_cpp`'s coverage node calls automatically — see
[STC coverage + object search integration](#stc-coverage--object-search-integration).

### Reading the output

Terminal logging is intentionally minimal (INFO level shows only phase 1
detections and phase 2 reasoning/moves; everything else is DEBUG):

```
[INFO] Phase 1 FOUND: chair (conf=0.91) at (0.82, 1.14, 0.61)
[INFO] Phase 1 candidate: book (conf=0.42) at (1.05, -0.30, 0.55) -> phase 2
[INFO] Phase 2 candidate 1, try 1: 3D approach - object at (1.02, -0.28, 0.52) in base_link, ~1.10m away
[INFO] Phase 2 FOUND: book (conf=0.86) at (0.94, -0.25, 0.50)
[INFO] Scan complete. Found 2 object(s):
[INFO]   - chair (conf=0.91) at (0.82, 1.14, 0.61)
[INFO]   - book (conf=0.86) at (0.94, -0.25, 0.50)
[INFO] Published 2 found-object marker(s) in map frame
```

If `--save-detections-dir` is set, every target-label detection (phase 1)
and every phase 2 frame gets a bounding-box-overlaid JPEG saved under a
per-run timestamped folder, e.g.:

```
~/husky_ws/detections/20260726_141619/
├── phase1/
│   ├── step003.jpg
│   └── ...
└── phase2/
    ├── p2_c0_t0.jpg   # candidate 0, try 0
    └── ...
```

---

## Running full coverage + search

With the sim (+ localization/navigation) up:

```bash
# Terminal 2: the action server the coverage node will call into
ros2 run husky_ur10_cam object_search_action_server.py

# Terminal 3: coverage, searching for chair/book/laptop, bigger 4-subcell major cells
ros2 launch stc_cpp stc.launch.py search_objects:="chair,book,laptop" subcell_per_cell:=4
```

| Launch arg | Default | Description |
|---|---|---|
| `search_objects` | `chair` | Comma-separated target class(es), e.g. `"chair,book,laptop"`. |
| `search_max_object_distance` | `0.0` | Forwarded as `FindObject` goal's `max_object_distance`; `<= 0.0` leaves it to the action server's own default. |
| `subcell_per_cell` | `2` | Major-cell size as a multiple of the robot-footprint-sized subcell. Raise it for bigger cells (fewer, coarser search stops). |

The robot covers the map via the spanning-tree path and pauses to run
`FindObject` each time it crosses into a new major cell (see
[integration details](#stc-coverage--object-search-integration) above).

---

## Other useful scripts

**`test_ee_pose_ik.py`** — command the end-effector camera to an arbitrary
pose directly via `/compute_ik`, useful for exploring reachability or
deriving new fixed joint targets (this is how `HOME_JOINTS` in the main
script was originally solved):

```bash
# Check (don't execute) IK for a pose 0.6m in front, 0.9m up, in base_link
ros2 run husky_ur10_cam test_ee_pose_ik.py --xyz 0.6 0.0 0.9

# Point the camera at a specific direction and execute the move
ros2 run husky_ur10_cam test_ee_pose_ik.py --xyz 0.6 0.0 0.9 \
    --look-dir 1 0 0 --execute
```

**`preset_views.py`** — cycles the arm through a fixed list of preset joint
poses (useful for quick visual sanity checks of reachable views).

---

## Camera configuration notes

The wrist camera is defined in `husky_ur10_cam/urdf/ur10_cam.urdf.xacro`
(and mirrored in the pre-expanded `combined.urdf`) as a Gazebo `depth`-type
`<sensor>` publishing both RGB (`/ee_camera/image_raw`) and an organized
point cloud (`/ee_camera/points`), in `camera_optical_link` (REP-103 optical
convention: X=right, Y=down, Z=forward).

Current settings: **640×480**, **1.0 rad (~57.3°) horizontal FOV** — chosen
to reduce lens distortion and improve YOLO detection quality vs. the
original 114.6° FOV. If you widen the FOV again, increase `--num-sectors`
to compensate for the reduced frame-to-frame overlap.

> **Gotcha if you ever edit this sensor block:** Gazebo Classic's SDF spec
> requires the element to be named `<camera>` — **not** `<depth_camera>` —
> even for `type="depth"` sensors. A `<depth_camera>` tag is silently
> ignored (no error), and Gazebo falls back to a hardcoded 320×240 default
> regardless of what you put in `<image>`. If a resolution/FOV change isn't
> taking effect after a sim restart, check the tag name first.

---

## Troubleshooting

- **Nothing found / candidates never confirm within `--max-object-distance`
  (default 3m):** the object may genuinely be farther than the cap, or
  outside every phase 1 waypoint's view. Try increasing
  `--max-object-distance`, `--tilt-down-deg`, or `--num-sectors`, or check
  the saved phase 1 images to see what the sweep actually captured.
- **YOLO never detects your target object:** confirm it's one of YOLO's 80
  COCO classes. Common misses: "camera" isn't a COCO class at all — no
  weights will detect it without custom training.
  <details><summary>COCO classes reference</summary>
  person, bicycle, car, motorcycle, airplane, bus, train, truck, boat,
  traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat,
  dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack,
  umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball,
  kite, baseball bat, baseball glove, skateboard, surfboard, tennis
  racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana,
  apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake,
  chair, couch, potted plant, bed, dining table, toilet, tv, laptop,
  mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink,
  refrigerator, book, clock, vase, scissors, teddy bear, hair drier,
  toothbrush.
  </details>
- **Qwen model fails to load / OOM:** confirm it was pre-downloaded with
  `local_files_only=True` in mind (`hf download ...` beforehand — the
  script does not fetch it on demand), and that `accelerate` is installed
  so `device_map="auto"` can shard the model across GPUs.
- **`/compute_ik` or `/move_action` never come up:** MoveIt2's `move_group`
  node takes a few seconds after the launch file starts; the search script
  waits for both, but check `ros2 node list` / `ros2 topic list` if it
  hangs indefinitely — most often this means the Gazebo spawn or
  controller spawners further up the launch chain failed.
- **STC coverage never pauses to search:** check
  `object_search_action_server.py` is actually running — if the
  `find_object` action server isn't reachable within 5s, `stc.py` logs a
  warning and continues coverage without searching, by design (fail-open
  rather than blocking forever).
- **No markers in RViz:** `publish_found_markers` needs a live
  `map -> base_link` TF (i.e. localization actually running); without it,
  it logs a warning and skips publishing rather than dropping a wrong
  marker. Also double check the RViz `MarkerArray` display's topic
  (`/found_objects_markers`) and Fixed Frame (`map`).
- **RViz interactive-marker log spam:** already suppressed in
  `husky_ur10_moveit_config/launch/ur_moveit.launch.py` via
  `--ros-args --log-level warn` on the RViz node.
- **Gazebo crashes on launch with "non-unique link names":** already fixed
  in `office_env_large.world` (duplicate `window`/`door_3d` model
  instances used to share link names). If you add more model instances of
  the same type to a world file, give each a unique model name.

---

## Package overview

| Package | Role |
|---|---|
| `husky_ur10_cam` | Robot description (Husky + UR10 + wrist camera), Gazebo worlds, sim launch files, and the object-search scripts (CLI + action server). Start here. |
| `husky_ur10_moveit_config` | MoveIt2 configuration for the UR10 arm (SRDF, kinematics solver, OMPL planning pipeline, controller bridge, RViz MoveIt plugin). |
| `custom_interfaces` | `FindObject` action and `DetectedObject` message shared between the object search and STC coverage. |
| `stc_cpp` | Spanning-tree-coverage path planner; drives the base and calls `FindObject` per coverage cell. |
| `pointcloud_concatenate_ros2` | Merges multiple point cloud sources into one fused cloud (`/fusion`). |
| `pointcloud_to_laserscan` | Converts the fused point cloud into a 2D `/scan` LaserScan for Husky ground navigation. |
