# MuJoCo Simulation

FR3 Husky controllers can run the same `ros2_control` pipeline inside MuJoCo without real hardware. Simply add `use_mujoco:=true` to any controller launch file.

---

## Table of Contents
- [Requirements](#requirements)
- [Supported Combinations](#supported-combinations)
- [Launch Commands](#launch-commands)
- [Launch Arguments](#launch-arguments)
- [Scene Files](#scene-files)
- [Reach v2 Torque-Response Comparison](#reach-v2-torque-response-comparison)
- [How It Works](#how-it-works)
- [Differences from Real Hardware](#differences-from-real-hardware)

---

## Requirements

- [mujoco_ros_hardware](https://github.com/JunHeonYoon/mujoco_ros_hardware) built and sourced
- MuJoCo >= 3.x

Build:
```bash
colcon build --packages-select \
  mujoco_ros_hardware \
  fr3_husky_description \
  fr3_husky_controller
```

---

## Supported Combinations

### FR3 only (`fr3_controller.launch.py`, `fr3_action_controller.launch.py`)

| `robot_side` | `load_gripper` | `load_mobile` |
|---|---|---|
| `left` / `right` / `dual` | `true` / `false` | `true` / `false` |

### FR3 + Husky (`fr3_husky_controller.launch.py`, `fr3_husky_action_controller.launch.py`)

| `robot_side` | `load_gripper` |
|---|---|
| `left` / `right` / `dual` | `true` / `false` |

---

## Launch Commands

```bash
# FR3 test controller
ros2 launch fr3_husky_controller fr3_controller.launch.py \
  robot_side:=left load_gripper:=true load_mobile:=false use_mujoco:=true

# FR3 test controller (with mobile base)
ros2 launch fr3_husky_controller fr3_controller.launch.py \
  robot_side:=left load_gripper:=true load_mobile:=true use_mujoco:=true

# FR3 + Husky test controller
ros2 launch fr3_husky_controller fr3_husky_controller.launch.py \
  robot_side:=left load_gripper:=true use_mujoco:=true

# FR3 action controller
ros2 launch fr3_husky_controller fr3_action_controller.launch.py \
  robot_side:=left load_gripper:=true load_mobile:=false use_mujoco:=true

# FR3 + Husky action controller
ros2 launch fr3_husky_controller fr3_husky_action_controller.launch.py \
  robot_side:=left load_gripper:=true use_mujoco:=true
```

---

## Launch Arguments

Behaviors enabled when `use_mujoco:=true`:

| Behavior | Description |
|---|---|
| `joint_state_publisher` disabled | MuJoCo publishes joint states directly; merging is not needed |
| `robot_state_publisher` remapped | `joint_states` → `{robot}/joint_states` (subscribes to arm topic directly) |
| `mujoco_scene_xacro_path` passed | MJCF xacro path forwarded as a `controller_manager` parameter |
| `mujoco_scene_xacro_args` passed | Xacro args such as `side`, `hand`, `mobile` forwarded at runtime |
| Gripper launch disabled | Real gripper node not needed in simulation |
| `franka_robot_state_broadcaster` disabled | Real-hardware-only broadcaster not needed |
| `husky_control` (robot_localization) disabled | Real-hardware-only; FR3+Husky launch files only |

---

## Scene Files

MJCF xacro scene files are located in `fr3_husky_description/mjcf/`:

| File | Description |
|---|---|
| `single_fr3.xml.xacro` | Single FR3 arm scene |
| `dual_fr3.xml.xacro` | Dual FR3 arm scene |
| `single_fr3_husky.xml.xacro` | Single FR3 arm + Husky scene |
| `dual_fr3_husky.xml.xacro` | Dual FR3 arm + Husky scene |

### Key xacro arguments (`single_fr3.xml.xacro`)

| Argument | Default | Description |
|---|---|---|
| `side` | `left` | Arm side (`left` \| `right`) |
| `hand` | `false` | Include gripper |
| `mobile` | `false` | Include mobile base |
| `arm_id` | `fr3` | Arm model identifier |
| `control_mode` | `effort` | Control mode (`position` \| `velocity` \| `effort`) |

> `control_mode` is not set directly at launch. `FrankaSubHandler` in `mujoco_ros_hardware` determines it automatically from the activated controller's command interfaces.

---

## Reach v2 Torque-Response Comparison

`scripts/tools/record_reach_v2_torque_response_mujoco.py` reproduces the
open-loop experiment from
`dual_fr3_lab/scripts/tools/record_reach_v2_torque_response.py` directly in
MuJoCo. It bypasses ROS 2 controllers and the policy so the recorded motion is
the rigid-body response to joint torque alone.

Requirements:

- MuJoCo 3.x Python bindings (`python3 -m pip install mujoco`)
- A sourced ROS 2 workspace containing `franka_description`,
  `husky_description`, and `fr3_husky_description`, so `xacro` can expand the
  dual-FR3 MJCF

Run it from the `fr3_husky_etri` repository:

```bash
source /opt/ros/humble/setup.bash
cd /home/dyros/etri_ws
colcon build --symlink-install --packages-select fr3_husky_description
source /home/dyros/etri_ws/install/setup.bash
cd /home/dyros/etri_ws/src/fr3_husky_etri

python3 scripts/tools/record_reach_v2_torque_response_mujoco.py \
  --reference_metadata ../dual_fr3_lab/logs/torque_step/physx_reach_v2_torque_step.json \
  --torque_rate_limit 1000 \
  --viz none
```

The reference JSON is authoritative: the script reuses its 1 kHz timestep,
14-DoF initial pose, selected joints, per-joint torque amplitudes, positive and
negative trials, settle/pulse/recovery durations, torque-rate limit, armature,
friction, and displacement safety threshold. Command-line overrides are
available, but should not be used for a direct PhysX–MuJoCo comparison.

The default output is:

```text
logs/torque_step/mujoco_reach_v2_torque_step.csv
logs/torque_step/mujoco_reach_v2_torque_step.json
```

The MuJoCo CSV uses the same column names and trial ordering as the PhysX CSV.
Every trial resets both arms to the reference ready pose before applying
`settle -> pulse -> recovery`. During this test, the normal MJCF armature,
damping, and friction are replaced by the values in the reference metadata;
gravity and contacts are disabled to match Isaac's per-robot gravity disable
and no-ground-plane setup. These overrides affect only this standalone test,
not normal MuJoCo controller launches.

Use `--viz passive` to observe the trials. To test only selected action-order
joints, pass indices with `--joint_ids`; left joints 1–7 are `0..6` and right
joints 1–7 are `7..13`.

---

## How It Works

1. `ros2_control_node` starts with `mujoco_scene_xacro_path` and `mujoco_scene_xacro_args` as node parameters.
2. `MujocoHardwareInterface::on_init()` creates a sub-handler based on `robot_type` and registers it with `MujocoWorldSingleton`.
3. When a controller activates, `perform_command_mode_switch()` lets `FrankaSubHandler` determine `control_mode` from the activated interfaces.
4. The xacro is processed to produce an MJCF XML string. `MujocoWorldSingleton` loads the scene, opens the viewer, and starts the simulation thread.
5. `read()` / `write()` access `mjData` each control cycle to exchange joint states and commands.

For internal architecture details, see [mujoco_ros_hardware/ARCHITECTURE.md](../../mujoco_ros_hardware/ARCHITECTURE.md).

---

## Differences from Real Hardware

| Item | Real hardware | MuJoCo |
|---|---|---|
| `joint_state_publisher` | Active (merges arm + gripper) | Disabled |
| Gripper node | Running | Disabled |
| `franka_robot_state_broadcaster` | Running | Disabled |
| `robot_model` dynamics | libfranka | `MujocoFrankaModel` (mass matrix, gravity from `qfrc_bias`) |
| `coriolis()` | Computed by libfranka | Returns zeros (already included in `gravity()`) |
| `pose()` / Jacobians | Computed by libfranka | Returns zeros (stub) |
| Control rate | 1000 Hz (real arm) | Based on MJCF `opt.timestep` |
