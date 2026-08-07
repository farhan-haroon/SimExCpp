# SimExCpp — An Agentic Object-Search System for a Husky+UR10 Mobile Manipulator

## VLM guided object detection
https://github.com/user-attachments/assets/527f9ec9-837a-4a45-9f2b-567cee19195d

## Spanning Tree based 2D coverage
https://github.com/user-attachments/assets/ee48e1a7-ab16-4f15-8393-08e0941419d7

A ROS 2 workspace simulating a **Clearpath Husky** carrying a **Universal
Robots UR10** arm with a wrist-mounted RGB-D camera, built around a
**search agent** that runs its own perceive → reason → act loop rather
than following a scripted path: it sweeps the scene with YOLOv8, reasons
about ambiguous detections with a vision-language model (Qwen2.5-VL) when
geometry alone isn't enough, and directs the arm to close in and confirm.

The agent is exposed as a ROS 2 action (`FindObject`) — a tool any
orchestrator can call. Here, a spanning-tree coverage planner plays that
role: it drives the base around the map and invokes the search agent when
**active perception** — 4 chassis-mounted cameras and a periodic VLM scene
check — judges an area worth investigating, restricting the search to just
that direction rather than sweeping everywhere; it never blocks
indefinitely if the agent isn't reachable. Every confirmed find gets an
RViz marker at its real 3D position.

This README is a step-by-step path to reproducing that from a clean machine.

---

## Contents

- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [CLI reference](#cli-reference)
- [How the agent works](#how-the-agent-works)
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

### 7. Run the search agent

In a new terminal (sourced):

```bash
ros2 run husky_ur10_cam ee_camera_environment_scan.py   # default targets: cup, book
```

The agent sweeps its surroundings, reasons about anything promising, and
logs each confirmed find:

```
[INFO] Phase 1 FOUND: book (conf=0.91) at (0.82, 1.14, 0.61)
[INFO] Scan complete. Found 1 object(s):
[INFO]   - book (conf=0.91) at (0.82, 1.14, 0.61)
[INFO] Published 1 found-object marker(s) in map frame
```

### 8. (Optional) Autonomous coverage + agent-driven search

The coverage planner takes over as orchestrator, invoking the search agent
on its own as it explores. Needs localization/navigation running too. A
pre-built map of the office world ships in `husky_ur10_cam/maps/`
(`my_map.yaml` + `my_map.pgm`):

```bash
ros2 launch nav2_bringup localization_launch.py use_sim_time:=true \
    map:="$(ros2 pkg prefix husky_ur10_cam)/share/husky_ur10_cam/maps/my_map.yaml"

# Wraps nav2_bringup's navigation_launch.py, rerouting its final velocity
# command (published on /cmd_vel by default) to
# /diff_drive_controller/cmd_vel_unstamped
# - the topic ros2_control's diff_drive_controller actually listens on.
ros2 launch husky_ur10_cam navigation.launch.py use_sim_time:=true

# Everything else in one shot: the shared Qwen2.5-VL server, active
# perception (chassis-camera scene check), the search agent (FindObject
# action server), and the coverage orchestrator - see husky_ur10_cam/
# launch/object_search.launch.py for what this bundles.
ros2 launch husky_ur10_cam object_search.launch.py
```

Whether the orchestrator ever calls the agent is controlled by
`enable_object_search` in `all_params.yaml` (see
[Configuration](#configuration)) - `false` there runs plain coverage with
no `FindObject` calls at all. When `true`, coverage doesn't search every
cell unconditionally anymore: it only calls `FindObject` at a new major
cell if active perception's latest chassis-camera check flagged a
direction there (and then only sweeps that 90° quadrant, not the full
360°) - see [How the agent works](#how-the-agent-works). Point
`object_search.launch.py` at your own copy of the file with
`params_file:=/path/to/override.yaml` to change it without touching the
checked-in one.

Prefer separate terminals instead (e.g. to watch one node's log on its
own, or restart just one without the rest)? Each piece still runs
standalone:

```bash
ros2 run husky_ur10_cam qwen_vlm_server.py            # must be up before the next two
ros2 run husky_ur10_cam active_perception_node.py
ros2 run husky_ur10_cam object_search_action_server.py
ros2 launch stc_cpp stc.launch.py
```

Add a `MarkerArray` RViz display on `/found_objects_markers` (Fixed Frame =
`map`) to watch finds accumulate as the robot covers the map.

---

## Configuration

`husky_ur10_cam/config/all_params.yaml` is the single file every search and
coverage tunable lives in - edit it, no code or launch-arg changes needed:

- **`object_search.launch.py`** (the launch file from step 8) passes it to
  every node it starts - the single argument to change for a whole run.
- Run a node standalone instead (`ros2 run ...`) and it loads this file
  automatically too (each falls back to its own code defaults if the
  file's missing); `stc_cpp`'s `kruskal_stc_node` needs it explicitly via
  `stc.launch.py`'s `params_file` argument (defaults to this file).

**What to search for is set in exactly one place**, `target_objects` under
the top-level `/**:` block - every node below reads that same value (ROS
2's yaml parser doesn't support referencing one block's value from
another, so a shared `/**:` block, applied to every node in the file, is
the native way to do this instead of three separate lists):

```yaml
/**:
  ros__parameters:
    target_objects: ["cup", "book"]   # <- change what's searched for here, once

object_search_action_server:
  ros__parameters:
    helix_radius: 0.6
    max_tries: 5
    # ... every phase 1/phase 2 tuning knob - see the file for the full list

kruskal_stc_node:
  ros__parameters:
    enable_object_search: true   # false = never search; true = gated by
                                  # active perception's chassis-camera hint
    subcell_per_cell: 2

active_perception_node:
  ros__parameters:
    tick_interval_sec: 2.0        # how often to sample the cameras
    max_check_distance_m: 3.0     # flagged direction must be within this range to count

qwen_vlm_server:
  ros__parameters:
    qwen_model: "Qwen/Qwen2.5-VL-7B-Instruct"   # shared by every VLM caller
```

To run with a different config without touching the checked-in file, point
`object_search.launch.py` at your own copy - it passes `params_file`
through to `stc.launch.py` too, so this one line covers all 4 nodes:

```bash
ros2 launch husky_ur10_cam object_search.launch.py params_file:=/path/to/override.yaml
```

(Running `stc.launch.py` standalone instead? It takes the same
`params_file` argument on its own - see step 8's per-node alternative.)

---

## CLI reference

`ee_camera_environment_scan.py` (also usable as the `find_object` action's
goal parameters):

| Flag | Default | Description |
|---|---|---|
| `--objects` | `cup book` | Target class(es), space-separated, e.g. `--objects chair book laptop`. Must be YOLO/COCO classes. |
| `--helix-radius` / `--helix-height` | `0.6` / `1.0` | Sweep orbit radius/height around `base_link` (m). |
| `--helix-center` | `0.0 0.0` | Orbit center `(x, y)` in `base_link`. |
| `--tilt-up-deg` / `--tilt-down-deg` | `5.0` / `30.0` | Pitch sweep range per azimuth. |
| `--num-sectors` | `16` | Azimuths around the robot. Raise if you widen the camera FOV. |
| `--tilts-per-sector` | `2` | Tilt waypoints per azimuth. |
| `--yolo-weights` | `~/husky_ws/yolov8n.pt` | Path to YOLO weights. |
| `--max-tries` | `5` | Per-candidate attempt budget before giving up on it. |
| `--max-object-distance` | `3.0` | Candidates farther than this (depth-measured, m) are skipped. |
| `--approach-standoff` / `--approach-max-step` | `0.15` / `0.3` | 3D approach standoff distance / max step per move (m). |
| `--save-detections-dir` | `~/husky_ws/detections` | Where annotated detection images are saved (empty disables). |
| `--skip-phase1` / `--skip-phase2` | off | Skip a phase entirely. |
| `--skip-start-home` / `--skip-return-home` | off | Skip the home move before/after the scan. |
| `--direction` | (none) | Restrict phase 1 to one chassis-camera quadrant (`front_left`/`front_right`/`rear_right`/`rear_left`), starting from that direction's anchor pose instead of home. This is how active-perception-triggered `FindObject` goals search; unset = full 360° sweep. |
| `--dry-run` | off | Print the phase 1 waypoints without moving the robot. |

Full list: `ros2 run husky_ur10_cam ee_camera_environment_scan.py --help`

These defaults (plus `stc.launch.py`'s coverage-search params) are edited in
`all_params.yaml` when running as a ROS node - see
[Configuration](#configuration). `--flag` CLI args above are only for direct
`ros2 run ee_camera_environment_scan.py` invocations, and always take
precedence there.

---

## How the agent works

The search agent (`ee_camera_environment_scan.py`, also exposed as the
`object_search_action_server.py` action server) runs a two-phase
perceive → reason → act loop per invocation. Neither phase follows a fixed
script beyond its stopping condition — each step's action depends on what
the previous one perceived.

**Phase 1 — perceive: broad sweep.** The arm holds a fixed orbit position
and steps through `--num-sectors` azimuths x `--tilts-per-sector` tilt
angles, running YOLO at each stop. This is the agent's coverage pass over
its own perception: a confident hit is recorded as found outright; a
weaker one is kept as a candidate worth a closer look. The sweep always
runs to completion, so multiple objects can surface in one pass.

**Phase 2 — reason + act: close the loop on each candidate.** For every
candidate, the agent re-perceives (YOLO re-check) then decides how to act:

- Confirmed → recorded as found, move on to the next candidate.
- Not yet confirmed, but depth is available at the detection → **act
  geometrically**: back-project the bounding box through the depth cloud
  and drive the arm to a precise 3D standoff from the object. No guessing.
- Depth unavailable or unreliable there → **act on VLM reasoning**: hand
  the frame and the detector's bounding box to Qwen2.5-VL, which classifies
  the object's position (LEFT/RIGHT/ABOVE/BELOW/CENTERED) and the agent
  converts that into a pan/tilt/forward move. The VLM never decides *what*
  was found — YOLO owns that — only *where to look next* when geometry
  can't answer it.

This repeats up to `--max-tries` per candidate, self-correcting each step
against fresh perception rather than committing to an open-loop plan. No
open-ended exploration — phase 2 only ever refines a candidate phase 1
already flagged.

**The agent as a tool.** `FindObject` is the agent's externally callable
interface: any caller — the coverage planner, or you by hand via `ros2
action send_goal` — sends a goal (target objects, optional max distance,
optional `direction` to restrict phase 1 to one quadrant - see below) and
gets back streaming feedback per step plus a structured result (confirmed
objects with 3D positions). The agent itself doesn't know or care who
called it.

**Active perception — deciding whether/where to look.** 4 depth cameras are
mounted radially on the chassis, at the same 4 diagonal directions
(`front_left`/`front_right`/`rear_right`/`rear_left`) as 4 preset arm poses
already defined in the MoveIt SRDF. `active_perception_node.py` samples all
4, plus the arm-mounted EE camera (anchored at the arm's `home` pose, where
it already sits between searches), every `tick_interval_sec` (default 2s)
while the base drives, and asks the shared VLM a single question: does any
of these 5 views give *context* suggesting the target might be nearby
(e.g. a bookshelf or a table with book-like items on it, for
`target_objects: ["book"]`) - as opposed to, say, a bare corridor? Its
answer (one of the 4 chassis directions, `home` for the EE camera, or
"nothing promising") only becomes a real hint if it's also within
`max_check_distance_m` (default 3m) of that camera - measured via its
depth point cloud at the frame center, the closest available proxy to
"whatever it was looking at" since the VLM only ever names a direction,
not a pixel. That hint is published for `stc_cpp` to read at each new
major cell. A chassis-direction hit sends the arm straight to that
direction's anchor pose and restricts phase 1 to the matching 90° quadrant;
a `home` hit runs a full 360° sweep from home instead (same shape as no
hint at all, just VLM-triggered rather than automatic); no hit (including
"promising but too far") means no search at all at that cell. YOLO still
owns confirming *what* was found -
the chassis/EE cameras and this VLM check only ever narrow down *whether
it's worth stopping to look*, and in which direction.

**Orchestration.** `stc_cpp`'s coverage node is the orchestrator: it drives
the base around the map and, each time it enters a new major cell with an
active-perception hint pending, pauses and calls the search agent as a
tool with that hint's direction. If the agent isn't reachable within 5s it
logs a warning and resumes coverage rather than blocking forever —
coverage always makes progress even if the agent is down.

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
- **Qwen fails to load / OOM:** it's loaded once by `qwen_vlm_server.py`
  (shared by phase 2 guidance and active perception's scene check, not
  loaded separately by either) - confirm it was pre-downloaded (`hf
  download ...`) and `accelerate` is installed for `device_map="auto"`
  sharding.
- **`/compute_ik` or `/move_action` never come up:** give `move_group` a few
  seconds after launch; if it hangs, check `ros2 node list` for a failed
  Gazebo spawn or controller further up the launch chain.
- **STC coverage never calls the search agent:** check `enable_object_search`
  in `all_params.yaml` is `true` first (`false` means never search, by
  design). If it is, this is most likely active perception never flagging
  a direction - by design, `true` no longer means "search every cell":
  `ros2 topic echo /active_perception/hint` while driving near the target
  object(s) to confirm `active_perception_node.py` and `qwen_vlm_server.py`
  are both running and producing non-empty `direction` hits. Only once a
  hint fires does `stc.py` call `find_object` - if that isn't reachable
  within 5s it logs a warning and continues without searching rather than
  blocking forever.
- **No markers in RViz:** needs a live `map -> base_link` TF (localization
  running). Check the `MarkerArray` display's topic (`/found_objects_markers`)
  and Fixed Frame (`map`).
- **Robot doesn't move under Nav2:** make sure you launched
  `husky_ur10_cam navigation.launch.py`, not `nav2_bringup`'s directly -
  it reroutes the final velocity command to
  `/diff_drive_controller/cmd_vel_unstamped`, which is what this robot's
  `diff_drive_controller` actually subscribes to (`use_stamped_vel: false`
  in `merged.yaml`; Nav2's `velocity_smoother` here only ever publishes
  plain `Twist`, not `TwistStamped`). Verify with `ros2 topic info
  /diff_drive_controller/cmd_vel_unstamped -v` - you should see
  `velocity_smoother` publishing and `diff_drive_controller` subscribed.
  Controller parameters are only read at spawn time, so a config change
  here needs the whole sim relaunched, not just Nav2.

---

## Repository layout

```
husky_ws/src/
├── husky_ur10_cam/            # Robot description, worlds, launch files, the search agent
│   ├── urdf/                  #   Husky+UR10+camera URDF/xacro
│   ├── worlds/                #   Gazebo worlds (office, small house, agriculture, empty)
│   ├── launch/                #   Sim launch files
│   ├── config/                #   ros2_control / controller yaml, all_params.yaml (agent + orchestrator tuning)
│   ├── models/                #   Gazebo models used to populate the worlds
│   ├── maps/                  #   Pre-built occupancy grid map (my_map.yaml/.pgm) for the office world
│   └── scripts/                #   The search agent, active perception, VLM server + IK tools
├── husky_ur10_moveit_config/  # MoveIt2 config for the UR10
├── pointcloud_concatenate_ros2/  # Merges point clouds (feeds laserscan generation)
├── pointcloud_to_laserscan/   # Point cloud -> 2D LaserScan for Husky navigation
├── custom_interfaces/         # FindObject action, DetectedObject/ActivePerceptionHint messages, VlmQuery service
└── stc_cpp/                   # Spanning-tree-coverage planner, the agent's orchestrator
```

| Package | Role |
|---|---|
| `husky_ur10_cam` | Robot description, Gazebo worlds, sim launch files, the object-search agent, active perception, the shared VLM server. Start here. |
| `husky_ur10_moveit_config` | MoveIt2 configuration for the UR10 arm. |
| `custom_interfaces` | `FindObject` action, `DetectedObject`/`ActivePerceptionHint` messages, `VlmQuery` service - the agent's tool-call contract. |
| `stc_cpp` | Coverage path planner; orchestrates the search agent via `FindObject` when active perception flags a cell. |
| `pointcloud_concatenate_ros2` | Merges point cloud sources into one fused cloud. |
| `pointcloud_to_laserscan` | Converts the fused cloud into a 2D `/scan` for navigation. |

| Script (`husky_ur10_cam/scripts/`) | Purpose |
|---|---|
| `ee_camera_environment_scan.py` | The search agent's perceive/reason/act logic + standalone CLI. |
| `object_search_action_server.py` | Same agent, exposed as a `FindObject` action server (tool interface). |
| `active_perception_node.py` | Periodically checks the 4 chassis cameras; publishes a direction hint (or none) for `stc_cpp` to act on. |
| `qwen_vlm_server.py` | Loads Qwen2.5-VL once, shared by phase 2 guidance and active perception via the `VlmQuery` service. |
| `test_ee_pose_ik.py` | Command the camera to an arbitrary pose via `/compute_ik` (used to derive `HOME_JOINTS`). |
| `preset_views.py` | Cycles the arm through a fixed list of preset joint poses. |
