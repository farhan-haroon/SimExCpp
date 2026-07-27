from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    search_objects_arg = DeclareLaunchArgument(
        'search_objects',
        default_value='chair',
        description='Comma-separated YOLO/COCO class name(s) the FindObject '
                     'action searches for when the coverage path enters a new '
                     'major cell, e.g. "chair,book,laptop"'
    )
    search_max_object_distance_arg = DeclareLaunchArgument(
        'search_max_object_distance',
        default_value='0.0',
        description='Meters; forwarded as FindObject goal.max_object_distance. '
                     '<= 0.0 leaves it to the object_search_action_server\'s own default.'
    )
    subcell_per_cell_arg = DeclareLaunchArgument(
        'subcell_per_cell',
        default_value='2',
        description='Major cell size, as a multiple of the robot-footprint-sized '
                     'subcell (major cell size = subcell_size * subcell_per_cell). '
                     'Raise this to make major cells bigger (fewer FindObject '
                     'searches, coarser coverage grouping) without changing the '
                     'robot\'s fine-grained subcell step size.'
    )

    stc_node = Node(
        package='stc_cpp',
        executable='stc',
        name='kruskal_stc_node',
        output='screen',
        parameters=[{
            'search_objects': LaunchConfiguration('search_objects'),
            'search_max_object_distance': LaunchConfiguration('search_max_object_distance'),
            'subcell_per_cell': LaunchConfiguration('subcell_per_cell'),
        }],
    )

    return LaunchDescription([
        search_objects_arg,
        search_max_object_distance_arg,
        subcell_per_cell_arg,
        stc_node,
    ])
