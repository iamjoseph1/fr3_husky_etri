from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    policy_share = FindPackageShare("fr3_husky_nn_policy")
    controller_share = FindPackageShare("fr3_husky_controller")

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
            "use_fake_hardware": "false",
            "load_gripper": "false",
            "launch_rviz": LaunchConfiguration("launch_rviz"),
            "launch_move_group": LaunchConfiguration("launch_move_group"),
            "control_rate_hz": LaunchConfiguration("control_rate_hz"),
            "mujoco_armature": LaunchConfiguration("mujoco_armature"),
            "mujoco_damping": LaunchConfiguration("mujoco_damping"),
            "mujoco_frictionloss": LaunchConfiguration("mujoco_frictionloss"),
        }.items(),
    )

    replay = Node(
        package="fr3_husky_nn_policy",
        executable="ppo_reach_policy_node",
        name="ppo_reach_policy_node",
        output="screen",
        parameters=[
            PathJoinSubstitution([policy_share, "config", "dual_fr3_reach.yaml"]),
            {
                "trajectory_path": LaunchConfiguration("trajectory_path"),
                "trajectory_noise_scale": LaunchConfiguration("noise_scale"),
                "shadow_mode": LaunchConfiguration("shadow_mode"),
                "auto_start": LaunchConfiguration("auto_start"),
                "reach_policy_step_action": ParameterValue(
                    LaunchConfiguration("reach_policy_step_action"), value_type=bool
                ),
                "log_task_name": LaunchConfiguration("log_task_name"),
                "experiment_control_rate_hz": ParameterValue(
                    LaunchConfiguration("control_rate_hz"), value_type=int
                ),
                "experiment_mujoco_armature": ParameterValue(
                    LaunchConfiguration("mujoco_armature"), value_type=float
                ),
                "experiment_mujoco_damping": ParameterValue(
                    LaunchConfiguration("mujoco_damping"), value_type=float
                ),
                "experiment_mujoco_frictionloss": ParameterValue(
                    LaunchConfiguration("mujoco_frictionloss"), value_type=float
                ),
            },
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("load_mobile", default_value="true"),
            DeclareLaunchArgument("use_mujoco", default_value="true"),
            DeclareLaunchArgument("launch_rviz", default_value="true"),
            DeclareLaunchArgument("launch_move_group", default_value="false"),
            DeclareLaunchArgument("shadow_mode", default_value="true"),
            DeclareLaunchArgument("auto_start", default_value="false"),
            DeclareLaunchArgument(
                "control_rate_hz",
                default_value="1000",
                description="Low-level controller update rate; use 100 or 1000",
            ),
            DeclareLaunchArgument(
                "mujoco_armature",
                default_value="0.1",
                description="Armature applied to every FR3 arm joint",
            ),
            DeclareLaunchArgument(
                "mujoco_damping",
                default_value="0.003",
                description="Viscous damping applied to every FR3 arm joint",
            ),
            DeclareLaunchArgument(
                "mujoco_frictionloss",
                default_value="0.2",
                description="Coulomb friction loss applied to every FR3 arm joint",
            ),
            DeclareLaunchArgument(
                "reach_policy_step_action",
                default_value="false",
                description=(
                    "Use a policy-step absolute joint target instead of refreshing "
                    "the measured-relative target at every controller step"
                ),
            ),
            DeclareLaunchArgument(
                "noise_scale",
                default_value="1.0",
                description=(
                    "Multiplier for the selected trajectory's noise columns; "
                    "0 is nominal and 1 is the stored profile"
                ),
            ),
            DeclareLaunchArgument(
                "trajectory_path",
                default_value=PathJoinSubstitution(
                    [
                        policy_share,
                        "trajectories",
                        "reach_1khz_matched_nominal_100hz_matched_noise.csv",
                    ]
                ),
            ),
            DeclareLaunchArgument(
                "log_task_name", default_value="dual_fr3_reach_trajectory"
            ),
            controller,
            replay,
        ]
    )
