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
| Reach | Relative `0.1 * action` offset | Refresh `q_target = measured_q + offset` at 100 Hz, then Isaac-style effort PD |

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
its relative target refresh rate is `relative_target_refresh_hz` (100 Hz by
default).

Before enabling either policy, verify the joint names, state update rate,
coordinate frame, initial pose, policy outputs, and configured safety limits in
shadow mode.

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

### Shadow mode

~~~bash
ros2 launch fr3_husky_nn_policy dual_fr3_lift_policy.launch.py shadow_mode:=true
~~~

Confirm that the object pose is fresh and inside the configured workspace, and
that the target lies inside the training range.

### Command mode

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
interval and recomputes the absolute target from the latest measured joint
positions every 10 ms, matching the five 100 Hz Isaac physics substeps.

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

### Shadow mode

Start the policy without robot commands:

~~~bash
ros2 launch fr3_husky_nn_policy dual_fr3_reach_policy.launch.py shadow_mode:=true
~~~

In a second terminal, start the target CLI:

~~~bash
source /home/dyros/etri_ws/install/setup.bash
ros2 run fr3_husky_nn_policy reach_target_cli
~~~

Enter a target as three values in meters:

~~~text
reach target> 0.50 0.00 0.20
~~~

The same CLI can publish new targets while the policy is running.

### Command mode

Move both arms to the training ready pose. Then launch without auto-start,
publish the desired target from reach_target_cli, and explicitly start control:

~~~bash
ros2 launch fr3_husky_nn_policy dual_fr3_reach_policy.launch.py shadow_mode:=false auto_start:=false

ros2 service call /ppo_reach_policy_node/start_policy std_srvs/srv/Trigger {}
~~~

Stop both arms with:

~~~bash
ros2 service call /ppo_reach_policy_node/stop_policy std_srvs/srv/Trigger {}
~~~

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
