"""Launches the whole object-search + active-perception stack in one shot
instead of 4 separate `ros2 run`/`ros2 launch` calls:

  1. qwen_vlm_server        - shared Qwen2.5-VL, one model load
  2. active_perception_node - periodic chassis-camera scene-gate check
  3. object_search_action_server - the FindObject tool
  4. stc_cpp's stc.launch.py - the coverage orchestrator (included, not
     duplicated - see stc_cpp/launch/stc.launch.py for its own params_file
     argument, which this file's params_file is passed straight through to)

All 4 load the same all_params.yaml (see husky_ur10_cam/config/
all_params.yaml, including its "/**" block - the one place the shared
target_objects list is set for every node here).

Still separate, unrelated prerequisites this does NOT launch: the Gazebo
sim itself, and Nav2 localization/navigation (both need to already be up -
see the README's quickstart).

Usage:
    ros2 launch husky_ur10_cam object_search.launch.py
    ros2 launch husky_ur10_cam object_search.launch.py params_file:=/path/to/override.yaml
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('husky_ur10_cam'), 'config', 'all_params.yaml']),
        description='YAML file with ros__parameters for every node in this '
                     'launch (qwen_vlm_server, active_perception_node, '
                     'object_search_action_server, kruskal_stc_node) - see '
                     'husky_ur10_cam/config/all_params.yaml, the default and '
                     'single place all of them (including the shared '
                     'target_objects list) are tuned. Point this at your '
                     'own copy to override it wholesale for one run.'
    )
    params_file = LaunchConfiguration('params_file')

    qwen_vlm_server = Node(
        package='husky_ur10_cam', executable='qwen_vlm_server.py',
        name='qwen_vlm_server', output='screen', parameters=[params_file],
    )
    active_perception_node = Node(
        package='husky_ur10_cam', executable='active_perception_node.py',
        name='active_perception_node', output='screen', parameters=[params_file],
    )
    object_search_action_server = Node(
        package='husky_ur10_cam', executable='object_search_action_server.py',
        name='object_search_action_server', output='screen', parameters=[params_file],
    )
    stc = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('stc_cpp'), 'launch', 'stc.launch.py'])),
        launch_arguments={'params_file': params_file}.items(),
    )

    return LaunchDescription([
        params_file_arg,
        qwen_vlm_server,
        active_perception_node,
        object_search_action_server,
        stc,
    ])
