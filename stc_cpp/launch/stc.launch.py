from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('husky_ur10_cam'), 'config', 'all_params.yaml']),
        description='YAML file with kruskal_stc_node\'s ros__parameters '
                     '(enable_object_search, search_objects, '
                     'search_max_object_distance, subcell_per_cell) - see '
                     'husky_ur10_cam/config/all_params.yaml, the default and '
                     'single place all coverage + object-search params are '
                     'tuned. Point this at your own file to override it '
                     'wholesale for one run.'
    )

    stc_node = Node(
        package='stc_cpp',
        executable='stc',
        name='kruskal_stc_node',
        output='screen',
        parameters=[LaunchConfiguration('params_file')],
    )

    return LaunchDescription([
        params_file_arg,
        stc_node,
    ])
