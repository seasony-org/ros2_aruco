#!/usr/bin/env python
"""
    This launches aurco detection for realsense sensor
"""
from time import sleep
import os
from launch import LaunchDescription
import launch
from launch.actions import SetEnvironmentVariable, DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config_aruco = os.path.join(get_package_share_directory('ros2_aruco'),
                                'config', 'parameters_usb.yaml')
    print(config_aruco)

    aurco_node_realsense = Node(
        package='ros2_aruco',
        executable='aruco_node_realsense_usb',
        name='aruco_node_realsense_usb',
        output='screen',
        parameters=[config_aruco]
    )

    return LaunchDescription([
        aurco_node_realsense
    ])
