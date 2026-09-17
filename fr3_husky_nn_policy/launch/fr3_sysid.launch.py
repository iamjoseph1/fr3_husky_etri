"""Launch the real rich-excitation data collector with the dual-FR3 controller."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    controller_share = FindPackageShare("fr3_husky_controller")
    # Keeping real and MuJoCo raw data apart is important: an offline fit must
    # never silently combine two plants in one training/validation set.
    log_root = PythonExpression(
        [
            "'",
            LaunchConfiguration("log_base"),
            "/' + ('mujoco' if '",
            LaunchConfiguration("use_mujoco"),
            "' in ('true', 'True', '1') else 'real')",
        ]
    )

    controller = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [controller_share, "launch", "fr3_action_controller.launch.py"]
            )
        ),
        launch_arguments={
            "robot_side": "dual",
            "load_mobile": LaunchConfiguration("load_mobile"),
            "use_mujoco": LaunchConfiguration("use_mujoco"),
            "use_fake_hardware": LaunchConfiguration("use_fake_hardware"),
            "load_gripper": "false",
            "with_realsense": LaunchConfiguration("with_realsense"),
            "launch_rviz": LaunchConfiguration("launch_rviz"),
            "launch_move_group": LaunchConfiguration("launch_move_group"),
        }.items(),
    )

    sysid = Node(
        package="fr3_husky_nn_policy",
        executable="fr3_sysid_node",
        name="fr3_sysid_node",
        output="screen",
        parameters=[
            {
                "joint_states_topic": LaunchConfiguration("joint_states_topic"),
                "sweep_mode": LaunchConfiguration("sweep_mode"),
                "target_joint": LaunchConfiguration("target_joint"),
                "command_rate_hz": LaunchConfiguration("command_rate_hz"),
                "return_to_start_between_trials": "true",
                "home_position_tolerance_rad": LaunchConfiguration(
                    "home_position_tolerance_rad"
                ),
                "home_velocity_tolerance_rad_s": LaunchConfiguration(
                    "home_velocity_tolerance_rad_s"
                ),
                "initial_settle_s": LaunchConfiguration("initial_settle_s"),
                "home_settle_s": LaunchConfiguration("home_settle_s"),
                "home_timeout_s": LaunchConfiguration("home_timeout_s"),
                "do_friction_sweep": LaunchConfiguration("do_friction_sweep"),
                "friction_amplitude_rad": LaunchConfiguration("friction_amplitude_rad"),
                "friction_blend_s": LaunchConfiguration("friction_blend_s"),
                "friction_dwell_s": LaunchConfiguration("friction_dwell_s"),
                "do_inertia_sine": LaunchConfiguration("do_inertia_sine"),
                "inertia_amplitude_rad": LaunchConfiguration("inertia_amplitude_rad"),
                "inertia_cycles": LaunchConfiguration("inertia_cycles"),
                "inertia_dwell_s": LaunchConfiguration("inertia_dwell_s"),
                "q0_prehold_s": LaunchConfiguration("q0_prehold_s"),
                "wrist_excitation_scale": LaunchConfiguration("wrist_excitation_scale"),
                "max_live_excursion_rad": LaunchConfiguration("max_live_excursion_rad"),
                "max_live_speed_rad_s": LaunchConfiguration("max_live_speed_rad_s"),
                "max_live_accel_rad_s2": LaunchConfiguration("max_live_accel_rad_s2"),
                "dry_run": LaunchConfiguration("dry_run"),
                "auto_start": LaunchConfiguration("auto_start"),
                "log_root": log_root,
            },
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("load_mobile", default_value="true"),
            DeclareLaunchArgument("use_mujoco", default_value="false"),
            # SysID uses the plate EE with no camera; keep the MuJoCo model
            # D435i-less so it matches the real plate setup.
            DeclareLaunchArgument("with_realsense", default_value="false"),
            DeclareLaunchArgument("use_fake_hardware", default_value="false"),
            DeclareLaunchArgument("launch_rviz", default_value="true"),
            DeclareLaunchArgument("launch_move_group", default_value="false"),
            DeclareLaunchArgument(
                "joint_states_topic", default_value="/dual_fr3/joint_states"
            ),
            # Default to one known, manually verified joint.  A full 14-joint
            # run must be selected explicitly after validating every q0 path.
            DeclareLaunchArgument("sweep_mode", default_value="false"),
            DeclareLaunchArgument("target_joint", default_value="5"),
            DeclareLaunchArgument("command_rate_hz", default_value="100.0"),
            DeclareLaunchArgument("home_position_tolerance_rad", default_value="0.01"),
            DeclareLaunchArgument(
                "home_velocity_tolerance_rad_s", default_value="0.02"
            ),
            DeclareLaunchArgument("initial_settle_s", default_value="0.75"),
            DeclareLaunchArgument("home_settle_s", default_value="0.75"),
            DeclareLaunchArgument("home_timeout_s", default_value="45.0"),
            DeclareLaunchArgument("do_friction_sweep", default_value="true"),
            DeclareLaunchArgument("friction_amplitude_rad", default_value="0.15"),
            DeclareLaunchArgument("friction_blend_s", default_value="0.30"),
            DeclareLaunchArgument("friction_dwell_s", default_value="0.40"),
            DeclareLaunchArgument("do_inertia_sine", default_value="true"),
            DeclareLaunchArgument("inertia_amplitude_rad", default_value="0.05"),
            DeclareLaunchArgument("inertia_cycles", default_value="3"),
            DeclareLaunchArgument("inertia_dwell_s", default_value="0.40"),
            DeclareLaunchArgument("q0_prehold_s", default_value="1.0"),
            DeclareLaunchArgument("wrist_excitation_scale", default_value="0.50"),
            DeclareLaunchArgument("max_live_excursion_rad", default_value="0.20"),
            DeclareLaunchArgument("max_live_speed_rad_s", default_value="0.40"),
            DeclareLaunchArgument("max_live_accel_rad_s2", default_value="5.0"),
            # Live hardware motion is opt-in even though the node itself also
            # defaults to dry-run.  Change this only after a dry preview and a
            # verified collision-free q0.
            DeclareLaunchArgument("dry_run", default_value="true"),
            DeclareLaunchArgument("auto_start", default_value="false"),
            DeclareLaunchArgument(
                "log_base",
                default_value=os.path.join(os.getcwd(), "logs", "sysid_raw"),
            ),
            controller,
            sysid,
        ]
    )
