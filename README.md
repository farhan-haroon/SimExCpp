# SimExCpp — VLM+YOLO-Guided Object Search for a Husky+UR10 Mobile Manipulator

A ROS 2 workspace that simulates a **Clearpath Husky** carrying a **Universal
Robots UR10** arm, fitted with a wrist-mounted RGB-D camera, and drives it
through a **two-phase autonomous object search**:

1. **Phase 1 — deterministic sweep.** The arm sweeps the camera through a
   fixed pan/tilt pattern around the robot while a YOLOv8 detector scans
   every frame for a target object class (e.g. `chair`, `book`, `bottle`).
2. **Phase 2 — VLM-guided investigation.** Every plausible-but-unconfirmed
   detection from phase 1 is revisited. The arm closes in on it using a
   combination of **depth-camera 3D back-projection** (precise, when
   available) and a **vision-language model** (Qwen2.5-VL-7B-Instruct, as a
   qualitative fallback) until YOLO confirms it above a fixed confidence
   threshold — or the search gives up on that candidate and moves to the
   next one.

The scan **never stops early**: it completes the full sweep and investigates
every candidate, then reports every object it found (with 3D position in
`base_link`) at the end.

This README documents the whole workspace and gives a step-by-step path to
reproducing the search demo from a clean machine.

---

## Table of contents

- [Repository layout](#repository-layout)
- [How the search works](#how-the-search-works)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
  - [1. ROS 2 workspace](#1-ros-2-workspace)
  - [2. Python/ML dependencies](#2-pythonml-dependencies)
  - [3. Models](#3-models)
- [Building](#building)
- [Running the simulation](#running-the-simulation)
- [Running the object search](#running-the-object-search)
  - [CLI reference](#cli-reference)
  - [Example runs](#example-runs)
  - [Reading the output](#reading-the-output)
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
│   └── scripts/                #   Python: IK testing + the object-search node (see below)
├── husky_ur10_moveit_config/  # MoveIt2 config (SRDF, kinematics, OMPL planning, controllers) for the UR10
├── pointcloud_concatenate_ros2/  # Merges multiple point clouds into one (used for laserscan generation)
├── pointcloud_to_laserscan/   # Point cloud → 2D LaserScan conversion (feeds Husky navigation)
├── custom_interfaces/         # Custom message definitions
└── stc_cpp/                   # Spanning-tree-coverage planner (Python)
```

The object-search work lives almost entirely in **`husky_ur10_cam/scripts/`**:

| File | Purpose |
|---|---|
| `ee_camera_environment_scan.py` | **Main entry point.** The two-phase search node described above. |
| `test_ee_pose_ik.py` | Standalone IK/pose-commanding tool used to test and solve fixed end-effector poses (e.g. how `HOME_JOINTS` in the main script was derived). |
| `preset_views.py` | Drives the arm through a list of fixed preset joint poses. |

---

## How the search works

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
- **YOLOv8** runs on every waypoint's frame, looking for `--object`:
  - confidence ≥ 80% (fixed threshold, not configurable) → logged as
    **FOUND** immediately, but the sweep *keeps going* through every
    remaining waypoint.
  - confidence > 0 but below threshold, and within `--max-object-distance`
    (measured via the depth camera) → saved as a **candidate** for phase 2.
- Before the sweep starts (and after it ends), the arm moves to a fixed,
  hardcoded joint-space `HOME_JOINTS` pose — not re-solved via IK each
  time, so it lands in the exact same configuration every run.

### Phase 2 — investigate phase 1 candidates only

For each saved candidate, in order, up to `--max-tries` (default 5) times:

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
     so investigation doesn't jump to a different, already-confirmed, or
     irrelevant nearby object of the same class.
3. If `--max-tries` is exhausted without confirming, the candidate is given
   up on and the next one is tried. There is no open-ended exploration —
   every phase 2 move only ever gets the camera closer to a *known*
   candidate.

At the end, the script reports **every** confirmed object (label,
confidence, 3D position in `base_link`) found across both phases, then
returns the arm home.

---

## Prerequisites

- **Ubuntu 22.04** with **ROS 2 Humble**
- **Gazebo Classic 11** (`gazebo_ros_pkgs`)
- **MoveIt 2** (Humble binaries)
- An **NVIDIA GPU** for the VLM. Qwen2.5-VL-7B-Instruct in bf16 is ~16.6 GB
  of weights, which does *not* fit comfortably on a single 16 GB GPU — the
  script loads it with `device_map="auto"` so it will shard across multiple
  GPUs if available. A single 24 GB+ GPU also works. YOLOv8n is tiny and
  runs fine on CPU or GPU.
- Python 3.10 (matches Humble's default interpreter)

## Setup

### 1. ROS 2 workspace

```bash
# Install ROS 2 Humble desktop + Gazebo Classic + MoveIt2 first (apt), then:
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
`moveit_simple_controller_manager`, `warehouse_ros_sqlite`, etc. — all
pulled in by `rosdep` above, provided the corresponding apt sources are
enabled: `ros-humble-ur`, `ros-humble-moveit`, `ros-humble-clearpath-*` or
equivalent).

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
`CMakeLists.txt` changes require a `colcon build`.

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

---

## Running the object search

With the sim up and workspace sourced in another terminal:

```bash
ros2 run husky_ur10_cam ee_camera_environment_scan.py --object chair
```

### CLI reference

| Flag | Default | Description |
|---|---|---|
| `--object` | `chair` | Target class name — must be a YOLO/COCO class; also used in the VLM's prompt. |
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

### Example runs

```bash
# Sanity-check the sweep plan without moving the robot
ros2 run husky_ur10_cam ee_camera_environment_scan.py --dry-run

# Search for a book, save every frame for review
ros2 run husky_ur10_cam ee_camera_environment_scan.py --object book \
    --save-detections-dir ~/husky_ws/detections

# Only run phase 2 investigation moves logic in isolation / skip the sweep
ros2 run husky_ur10_cam ee_camera_environment_scan.py --object bottle --skip-phase1

# Tighter search radius, closer object distance cap
ros2 run husky_ur10_cam ee_camera_environment_scan.py --object chair \
    --helix-radius 0.5 --max-object-distance 2.0
```

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
| `husky_ur10_cam` | Robot description (Husky + UR10 + wrist camera), Gazebo worlds, sim launch files, and the object-search scripts. Start here. |
| `husky_ur10_moveit_config` | MoveIt2 configuration for the UR10 arm (SRDF, kinematics solver, OMPL planning pipeline, controller bridge, RViz MoveIt plugin). |
| `pointcloud_concatenate_ros2` | Merges multiple point cloud sources into one fused cloud (`/fusion`). |
| `pointcloud_to_laserscan` | Converts the fused point cloud into a 2D `/scan` LaserScan for Husky ground navigation. |
| `custom_interfaces` | Custom ROS 2 message definitions used elsewhere in the workspace. |
| `stc_cpp` | Spanning-tree-coverage path planner (Python), independent of the object-search work. |
