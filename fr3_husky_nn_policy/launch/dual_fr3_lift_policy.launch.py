from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    policy_share = FindPackageShare("fr3_husky_nn_policy")
    controller_share = FindPackageShare("fr3_husky_controller")

    model_path = LaunchConfiguration("model_path")
    shadow_mode = LaunchConfiguration("shadow_mode")
    auto_start = LaunchConfiguration("auto_start")

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
            "load_gripper": LaunchConfiguration("load_gripper"),
            "launch_rviz": LaunchConfiguration("launch_rviz"),
            "launch_move_group": "false",
        }.items(),
    )

    policy = Node(
        package="fr3_husky_nn_policy",
        executable="ppo_liftcube_policy_node",
        name="ppo_liftcube_policy_node",
        output="screen",
        parameters=[
            PathJoinSubstitution([policy_share, "config", "dual_fr3_lift_v3.yaml"]),
            {
                "model_path": model_path,
                "shadow_mode": shadow_mode,
                "auto_start": auto_start,
            },
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("load_mobile", default_value="true"),
            DeclareLaunchArgument("use_mujoco", default_value="false"),
            DeclareLaunchArgument("use_fake_hardware", default_value="false"),
            DeclareLaunchArgument("load_gripper", default_value="true"),
            DeclareLaunchArgument("launch_rviz", default_value="true"),
            DeclareLaunchArgument("shadow_mode", default_value="true"),
            DeclareLaunchArgument("auto_start", default_value="false"),
            DeclareLaunchArgument(
                "model_path",
                default_value=PathJoinSubstitution(
                    [policy_share, "models", "dual_fr3_lift_v3_actor.npz"]
                ),
            ),
            controller,
            policy,
        ]
    )
