import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    controller_share = FindPackageShare("fr3_husky_controller")

    # Route logs into <log_base>/mujoco or <log_base>/real automatically so
    # simulation and hardware runs never share a directory.
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
            "launch_rviz": LaunchConfiguration("launch_rviz"),
            "launch_move_group": LaunchConfiguration("launch_move_group"),
        }.items(),
    )

    probe = Node(
        package="fr3_husky_nn_policy",
        executable="fr3_consistency_probe_node",
        name="fr3_consistency_probe_node",
        output="screen",
        parameters=[
            {
                "joint_states_topic": LaunchConfiguration("joint_states_topic"),
                "sweep_mode": LaunchConfiguration("sweep_mode"),
                "target_joint": LaunchConfiguration("target_joint"),
                "profile_type": LaunchConfiguration("profile_type"),
                "amplitude_rad": LaunchConfiguration("amplitude_rad"),
                "amplitude_wrist_rad": LaunchConfiguration("amplitude_wrist_rad"),
                "t_start_s": LaunchConfiguration("t_start_s"),
                "ramp_time_s": LaunchConfiguration("ramp_time_s"),
                "pulse_width_s": LaunchConfiguration("pulse_width_s"),
                "hold_time_s": LaunchConfiguration("hold_time_s"),
                "command_rate_hz": LaunchConfiguration("command_rate_hz"),
                "return_to_start_between_trials": LaunchConfiguration(
                    "return_to_start_between_trials"
                ),
                "return_to_start_after_final_trial": LaunchConfiguration(
                    "return_to_start_after_final_trial"
                ),
                "home_position_tolerance_rad": LaunchConfiguration(
                    "home_position_tolerance_rad"
                ),
                "home_velocity_tolerance_rad_s": LaunchConfiguration(
                    "home_velocity_tolerance_rad_s"
                ),
                "initial_settle_s": LaunchConfiguration("initial_settle_s"),
                "home_settle_s": LaunchConfiguration("home_settle_s"),
                "home_timeout_s": LaunchConfiguration("home_timeout_s"),
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
            DeclareLaunchArgument("use_fake_hardware", default_value="false"),
            DeclareLaunchArgument("launch_rviz", default_value="true"),
            DeclareLaunchArgument("launch_move_group", default_value="false"),
            DeclareLaunchArgument(
                "joint_states_topic", default_value="/dual_fr3/joint_states"
            ),
            DeclareLaunchArgument("sweep_mode", default_value="false"),
            DeclareLaunchArgument("target_joint", default_value="5"),
            DeclareLaunchArgument("profile_type", default_value="pulse"),
            DeclareLaunchArgument("amplitude_rad", default_value="0.05"),
            DeclareLaunchArgument("amplitude_wrist_rad", default_value="0.03"),
            DeclareLaunchArgument("t_start_s", default_value="1.0"),
            DeclareLaunchArgument("ramp_time_s", default_value="1.0"),
            DeclareLaunchArgument("pulse_width_s", default_value="0.5"),
            DeclareLaunchArgument("hold_time_s", default_value="2.0"),
            DeclareLaunchArgument("command_rate_hz", default_value="100.0"),
            DeclareLaunchArgument(
                "return_to_start_between_trials", default_value="false"
            ),
            DeclareLaunchArgument(
                "return_to_start_after_final_trial", default_value="true"
            ),
            DeclareLaunchArgument("home_position_tolerance_rad", default_value="0.01"),
            DeclareLaunchArgument(
                "home_velocity_tolerance_rad_s", default_value="0.02"
            ),
            DeclareLaunchArgument("initial_settle_s", default_value="0.75"),
            DeclareLaunchArgument("home_settle_s", default_value="0.75"),
            DeclareLaunchArgument("home_timeout_s", default_value="45.0"),
            DeclareLaunchArgument("dry_run", default_value="false"),
            DeclareLaunchArgument("auto_start", default_value="false"),
            # Raw probe CSVs default under the directory `ros2 launch` was
            # invoked from (normally the repo root after `cd fr3_husky_etri`),
            # so runs are versioned in logs/ alongside their analysis output.
            # Override log_base:= to write elsewhere.
            DeclareLaunchArgument(
                "log_base",
                default_value=os.path.join(os.getcwd(), "logs", "consistency_raw"),
            ),
            controller,
            probe,
        ]
    )
