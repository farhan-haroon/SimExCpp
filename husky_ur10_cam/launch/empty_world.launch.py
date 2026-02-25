import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.actions import TimerAction


def generate_launch_description():
    launch_file_dir = os.path.join(get_package_share_directory('husky_ur10_cam'), 'launch')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    control_yaml = PathJoinSubstitution([
        FindPackageShare("husky_ur10_cam"),
        "config",
        "merged.yaml"
    ])

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    x_pose = LaunchConfiguration('x_pose', default='0.0')
    y_pose = LaunchConfiguration('y_pose', default='0.0')

    world = os.path.join(
        get_package_share_directory('husky_ur10_cam'),
        'worlds',
        'empty_world.world'
    )

    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world}.items()
    )

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        )
    )

    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    spawn_custom_husky_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'spawn_custom_husky.launch.py')
        ),
        launch_arguments={
            'x_pose': x_pose,
            'y_pose': y_pose
        }.items()
    )

    spawn_diff_drive_controller = Node(
        package = "controller_manager",
        executable = "spawner",
        arguments = ["diff_drive_controller", "-c", "/controller_manager"],
        output = "screen",
    )

    spawn_joint_state_broadcaster = Node(
        package = "controller_manager",
        executable = "spawner",
        arguments = ["joint_state_broadcaster", "-c", "/controller_manager"],
        output = "screen",
    )
    
    spawn_ur_controller = Node(
       package="controller_manager",
        executable="spawner",
        arguments=["joint_trajectory_controller", "-c", "/controller_manager"],
        output="screen",
    )

    pcl_to_ls = Node(
        package = "pointcloud_to_laserscan",
        executable = "pointcloud_to_laserscan_node",
        name = "pointcloud_to_laserscan",
        remappings = [
            ('cloud_in', '/velodyne_points'),
            ('scan', '/scan')
        ],
        parameters = [{
            'target_frame': 'velodyne',
            'transform_tolerance': 0.05,
            'min_height': -0.65,
            'max_height': 0.65,
            'angle_min': -3.14,
            'angle_max': 3.14,
            'angle_increment': 0.0087,
            'scan_time': 0.05,
            'range_min': 0.2,
            'range_max': 100.0,
            'use_inf': True,
            'inf_epsilon': 2.0
        }]
    )

    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                FindPackageShare("husky_ur10_moveit_config"),
                "/launch",
                "/ur_moveit.launch.py",
            ]
        ),
        launch_arguments={
            "ur_type": "ur10",
            "use_sim_time": "true",
            "launch_rviz": "true",
            "use_fake_hardware": "false",   # IMPORTANT
        }.items(),
    )

    ld = LaunchDescription()

    # Add the commands to the launch description
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)
    ld.add_action(robot_state_publisher_cmd)
    
    ld.add_action(
        TimerAction(
            period=3.0,
            actions=[spawn_custom_husky_cmd]
        )
    )

    ld.add_action(
        TimerAction(
            period=5.0,
            actions=[
                spawn_joint_state_broadcaster,
                spawn_diff_drive_controller,
                spawn_ur_controller
            ]
        )
    )

    ld.add_action(
        TimerAction(
            period=8.0,
            actions=[moveit_launch]
        )
    )

    ld.add_action(pcl_to_ls)

    return ld
