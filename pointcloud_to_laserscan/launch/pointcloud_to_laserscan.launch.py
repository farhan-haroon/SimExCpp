from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            name='scanner', default_value='scanner',
            description='Namespace for sample topics'
        ),
        Node(
            package='pointcloud_to_laserscan', executable='pointcloud_to_laserscan_node',
            remappings=[
            	('cloud_in', '/velodyne_points'),
                ('scan', '/scan')
            ],
            parameters=[{
                'target_frame': 'velodyne',
                'transform_tolerance': 0.05,
                'min_height': -0.65,
                'max_height': -0.65,
                'angle_min': -3.14,
                'angle_max': 3.14,
                'angle_increment': 0.0087,
                'scan_time': 0.2,
                'range_min': 0.2,
                'range_max': 10.0,
                'use_inf': True,
                'inf_epsilon': 2.0
            }],
            name='pointcloud_to_laserscan'
        )
    ])
