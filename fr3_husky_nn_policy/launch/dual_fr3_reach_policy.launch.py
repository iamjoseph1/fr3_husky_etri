from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
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
            "use_fake_hardware": LaunchConfiguration("use_fake_hardware"),
            "load_gripper": "false",
            "launch_rviz": LaunchConfiguration("launch_rviz"),
            "launch_move_group": LaunchConfiguration("launch_move_group"),
        }.items(),
    )

    policy = Node(
        package="fr3_husky_nn_policy",
        executable="ppo_reach_policy_node",
        name="ppo_reach_policy_node",
        output="screen",
        parameters=[
            PathJoinSubstitution([policy_share, "config", "dual_fr3_reach.yaml"]),
            {
                "model_path": LaunchConfiguration("model_path"),
                "shadow_mode": LaunchConfiguration("shadow_mode"),
                "auto_start": LaunchConfiguration("auto_start"),
                "reach_policy_step_action": ParameterValue(
                    LaunchConfiguration("reach_policy_step_action"), value_type=bool
                ),
                "reach_goal_sequence": LaunchConfiguration("reach_goal_sequence"),
                "reach_goal_sequence_loop": LaunchConfiguration(
                    "reach_goal_sequence_loop"
                ),
                # Keep simulation and real-robot runs separate while using the
                # exact same ReachRunLogger fields and plots for both.
                "log_task_name": PythonExpression(
                    [
                        "'dual_fr3_reach' if '",
                        LaunchConfiguration("use_mujoco"),
                        "'.lower() == 'true' else 'dual_fr3_reach_real'",
                    ]
                ),
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
            DeclareLaunchArgument("shadow_mode", default_value="true"),
            DeclareLaunchArgument("auto_start", default_value="false"),
            DeclareLaunchArgument(
                "reach_policy_step_action",
                default_value="false",
                description=(
                    "If true, hold q_measured(t_k)+0.1*action[k] as an absolute "
                    "target until the next policy step; false preserves the "
                    "per-controller-step measured-relative target."
                ),
            ),
            DeclareLaunchArgument("reach_goal_sequence", default_value=""),
            DeclareLaunchArgument("reach_goal_sequence_loop", default_value="false"),
            DeclareLaunchArgument(
                "model_path",
                default_value=PathJoinSubstitution(
                    [policy_share, "models", "dual_fr3_reach_actor_friction_w_1000hz_policystep.npz"]
                ),
            ),
            controller,
            policy,
        ]
    )
