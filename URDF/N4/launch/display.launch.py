import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
# 【新增】引入 ParameterValue，专门解决 yaml 解析报错
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    package_name = 'N4'
    
    pkg_share = get_package_share_directory(package_name)
    
    # 确保这里是你真正的文件名 (robot.urdf)
    default_model_path = os.path.join(pkg_share, 'urdf', 'robot.urdf')

    model_arg = DeclareLaunchArgument(name='model', default_value=default_model_path,
                                      description='Absolute path to robot urdf file')

    # 【核心修复】将 Command 结果包装为字符串，防止被解析为 YAML
    robot_description = ParameterValue(Command(['xacro ', LaunchConfiguration('model')]), value_type=str)

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}]
    )

    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen'
    )

    return LaunchDescription([
        model_arg,
        robot_state_publisher_node,
        joint_state_publisher_gui_node,
        rviz_node
    ])