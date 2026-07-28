"""Thin wrapper around nav2_bringup's navigation_launch.py that reroutes
its final velocity command to the topic this robot's diff_drive_controller
actually listens on.

nav2_bringup's own launch file remaps velocity_smoother's output
('cmd_vel_smoothed' in code) to the generic '/cmd_vel', which nothing here
subscribes to. SetRemap overrides that internal remap (matched by the
in-code topic name, before nav2's own remap is applied) rather than
relaying/republishing on top of it.

Target topic: diff_drive_controller's own 'cmd_vel' expects
TwistStamped by default (use_stamped_vel: true); this repo's
merged.yaml sets use_stamped_vel: false instead, since nav2's
velocity_smoother here only ever publishes plain Twist - with that
off, ros2_control's DiffDriveController opens a SEPARATE topic,
'~/cmd_vel_unstamped', for the Twist subscription (its 'cmd_vel'
topic still exists but has no subscriber). That's the one to target.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import SetRemap
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use the Gazebo /clock instead of wall time')

    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('nav2_bringup'), 'launch', 'navigation_launch.py'])),
        launch_arguments={'use_sim_time': LaunchConfiguration('use_sim_time')}.items(),
    )

    return LaunchDescription([
        use_sim_time_arg,
        GroupAction([
            SetRemap(src='cmd_vel_smoothed', dst='/diff_drive_controller/cmd_vel_unstamped'),
            navigation,
        ]),
    ])
