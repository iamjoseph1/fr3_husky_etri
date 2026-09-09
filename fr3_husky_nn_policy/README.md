# FR3 Husky NN policy

This ROS 2 package deploys trained LiftCube and Reach PPO actors on FR3 robots.
Both actors are exported as pickle-free NPZ files and run with NumPy, so
PyTorch and a GPU are not required at deployment time.

## Supported tasks

| Task | Controlled joints | Policy | External task input |
| --- | --- | --- | --- |
| LiftCube | Right arm and gripper | 36-D observation → 8-D action | Object pose and lift target |
| Reach | Left and right arms | 58-D observation → 14-D action | User-entered target center |

Both policies run at 20 Hz through the same policy control interface. Launch
files default to shadow mode, which performs inference without sending commands
to the robot. Their actuator-side execution modes are deliberately task-specific:

| Task | Streamed arm command | Controller execution |
| --- | --- | --- |
| LiftCube | Absolute `q + 0.1 * action` target | Existing 1 kHz velocity/acceleration/step-limited trajectory |
| Reach | Relative `0.1 * action` offset | Hold the offset for 50 ms; recompute `q_target = measured_q + offset` and Isaac-style PD torque every 1 kHz controller step |

Reach uses the nominal training actuator values `Kp=80`, `Kd=4`, with effort
limits of 87 Nm for joints 1-4 and 12 Nm for joints 5-7. This path bypasses the
legacy trajectory limiter so that Isaac Lab's `RelativeJointPositionAction`
substep behavior is preserved. Timeout, finite-value, joint-limit, and maximum
policy-offset validation remain active.

## Build

PolicyJointCommand supports either 7 single-arm targets or 14 dual-arm targets,
so rebuild the messages, controller, and policy package together:

~~~bash
cd /home/dyros/etri_ws
colcon build --packages-up-to fr3_husky_nn_policy fr3_husky_controller
source install/setup.bash
~~~

## Common interfaces

- Joint states: /joint_states (sensor_msgs/JointState)
- Policy command: /policy_joint_command
- Controller action: /fr3_policy_control
- Command frame: base
- NPZ loader: NumpyMLPActor with allow_pickle=False

Policy target validation uses `max_policy_target_delta_rad`. For LiftCube,
actuator command smoothing is configured separately with
`max_actuator_step_rad`, `joint_velocity_scale`, and
`joint_acceleration_scale`. Reach does not use those three smoothing values;
its legacy `relative_target_refresh_hz` parameter is retained for action API
compatibility and is set to 1000 Hz. The controller no longer uses it as a gate.

### Shadow mode

To run either task without sending commands to the robot, add
`shadow_mode:=true` to its launch command. Use `shadow_mode:=false` only after
checking the robot state, initial pose, coordinate frame, and safety limits.

## LiftCube

### Behavior and model

LiftCube controls the right FR3 arm and gripper to lift an observed object
toward a configured target. It requires an object pose, typically supplied by
the ArUco perception pipeline.

- Model: models/dual_fr3_lift_v3_actor.npz
- Network: 36 → 256 → 128 → 64 → 8
- Launch: dual_fr3_lift_policy.launch.py
- Node: ppo_liftcube_policy_node

### Inputs and services

- Object pose: /object_pose (geometry_msgs/PoseStamped, base frame)
- Optional target pose: configured target_pose_topic
- Start: /ppo_liftcube_policy_node/start_policy
- Stop: /ppo_liftcube_policy_node/stop_policy

### Execution order

Move the right arm to the training ready pose and open its gripper. Then launch
without auto-start and explicitly start the policy:

~~~bash
ros2 launch fr3_husky_nn_policy dual_fr3_lift_policy.launch.py shadow_mode:=false auto_start:=false

ros2 service call /ppo_liftcube_policy_node/start_policy std_srvs/srv/Trigger {}
~~~

Stop the policy with:

~~~bash
ros2 service call /ppo_liftcube_policy_node/stop_policy std_srvs/srv/Trigger {}
~~~

## Reach

### Behavior and model

Reach controls all 14 arm joints. It does not require an object pose, camera,
ArUco marker, or gripper. The user supplies a target center in the robot base
frame, and the policy moves both end effectors toward the corresponding points
0.20 m to either side of that center.

- Model: models/dual_fr3_reach_actor.npz
- Network: 58 → 256 → 128 → 64 → 14
- Deterministic output: tanh(last linear layer), float32 range `[-1, 1]`
- Launch: dual_fr3_reach_policy.launch.py
- Node: ppo_reach_policy_node

The deployed model is exported from
`dual_fr3_reach_sim2real_v1/2026-09-03_18-57-37/model_4999.pt`.
The NPZ stores `output_activation=tanh`, and the Reach node rejects legacy
identity-output artifacts so an unsquashed policy cannot command the robot.
The node-side `action_clip=1.0` remains a numerical safety guard.

At each policy step the node publishes the 14 relative joint offsets rather
than an absolute target. The controller holds that offset for the 50 ms policy
interval. In reach v2, every 1 kHz controller cycle recomputes the target as
the latest measured joint position plus the held offset, matching Isaac Lab's
`RelativeJointPositionAction`, and then evaluates the PD torque and 1 Nm/update
torque-rate limit. There is no separate 100 Hz torque gate or torque-hold stage.

### Inputs and services

- Target center: /reach_target_pose (geometry_msgs/PoseStamped, base frame)
- Interactive target input: reach_target_cli
- Start: /ppo_reach_policy_node/start_policy
- Stop: /ppo_reach_policy_node/stop_policy

The target must remain inside the training range:

| Axis | Minimum | Maximum |
| --- | ---: | ---: |
| x | 0.40 m | 0.60 m |
| y | -0.10 m | 0.10 m |
| z | 0.10 m | 0.35 m |

### Execution order

Run the following steps in order. Do not start the policy before both arms have
reached the training ready pose and a Reach target has been published.

1. In terminal 1, launch the controller and Reach policy node without
   auto-start:

~~~bash
ros2 launch fr3_husky_nn_policy dual_fr3_reach_policy.launch.py shadow_mode:=false auto_start:=false
~~~

2. In terminal 2, move both arms to the training ready pose. Wait until the
   command finishes successfully:

~~~bash
ros2 run fr3_husky_task_manager move_to_joint
~~~

3. In terminal 3, start the target CLI and enter the desired center position
   as `x y z` in meters. The example below publishes `(0.50, 0.00, 0.20)` in
   the robot base frame:

~~~bash
source /home/dyros/etri_ws/install/setup.bash
ros2 run fr3_husky_nn_policy reach_target_cli
~~~

~~~text
reach target> 0.50 0.00 0.20
~~~

Keep this CLI open if the target needs to be updated while the policy is
running.

4. After the ready-pose motion and target publication are complete, start the
   policy from terminal 4:

~~~bash
ros2 service call /ppo_reach_policy_node/start_policy std_srvs/srv/Trigger {}
~~~

Stop both arms with:

~~~bash
ros2 service call /ppo_reach_policy_node/stop_policy std_srvs/srv/Trigger {}
~~~

### Reach trajectory log

A successful `start_policy` request starts a 20 Hz trajectory log. On launch
shutdown, the node writes the CSV data, metadata, and four PNG plots below:

~~~text
fr3_husky_nn_policy/log/dual_fr3_reach/<start time>/
├── eef_trajectory.csv
├── metadata.json
├── eef_x_vs_target.png
├── eef_y_vs_target.png
├── eef_z_vs_target.png
└── eef_trajectory_3d.png
~~~

The EEF coordinates use the same definition as training: each `fr3_link7`
frame plus `[0, 0, 0.132] m` in its local frame. The plotted per-arm targets
are the commanded center plus/minus `0.20 m` on the base-frame Y axis. Set the
`log_root` ROS parameter to override the default output root.

## Re-exporting an actor

Run the exporter only in an Isaac/RSL-RL environment containing PyTorch:

~~~bash
export_ppo_actor /path/to/model_4999.pt /path/to/output_actor.npz
~~~

For a tanh-squashed policy such as Reach Sim2Real v1, include the final
deterministic distribution transform explicitly:

~~~bash
export_ppo_actor \
  /path/to/dual_fr3_reach_sim2real_v1/2026-09-03_18-57-37/model_4999.pt \
  /path/to/dual_fr3_reach_actor.npz \
  --output-activation tanh
~~~

Examples of output names are dual_fr3_lift_v3_actor.npz and
dual_fr3_reach_actor.npz. The exported archive contains numeric arrays and
metadata, including the final output activation, and is loaded at runtime with
allow_pickle=False. Artifacts without `output_activation` remain format-v1
identity actors for backward compatibility.
