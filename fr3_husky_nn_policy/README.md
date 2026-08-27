# FR3 Husky NN policy

This package deploys the `dual_fr3_lift_v3/model_4999.pt` deterministic
actor without PyTorch or a GPU at runtime.

## Build

```bash
cd /home/dyros/etri_ws
colcon build --packages-up-to fr3_husky_nn_policy fr3_husky_controller
source install/setup.bash
```

## Inputs and outputs

- Input: `/joint_states` (`sensor_msgs/JointState`)
- Input: `/object_pose` (`geometry_msgs/PoseStamped`, frame `base`)
- Optional input: configured target pose topic
- Output: `/policy_joint_command`
- Controller action: `/fr3_policy_control`
- Start service: `/ppo_liftcube_policy_node/start_policy`
- Stop service: `/ppo_liftcube_policy_node/stop_policy`

The default launch is shadow-only and never sends robot commands.

## Run in shadow mode

```bash
ros2 launch fr3_husky_nn_policy dual_fr3_lift_policy.launch.py shadow_mode:=true
```

Verify joint names, pose freshness, the `base` frame, target position, and actor
outputs before enabling control.

## Run the real controller

Move both arms to the training ready pose and open the right gripper first.
Then launch command mode without auto-start:

```bash
ros2 launch fr3_husky_nn_policy dual_fr3_lift_policy.launch.py \
  shadow_mode:=false auto_start:=false
ros2 service call /ppo_liftcube_policy_node/start_policy std_srvs/srv/Trigger {}
```

Stop at any time:

```bash
ros2 service call /ppo_liftcube_policy_node/stop_policy std_srvs/srv/Trigger {}
```

## Re-export the actor

Run this only in the Isaac/RSL-RL environment containing PyTorch:

```bash
export_ppo_actor /path/to/model_4999.pt \
  /path/to/dual_fr3_lift_v3_actor.npz
```

The deployed NPZ contains only numeric arrays and can be loaded with
`allow_pickle=False`.
