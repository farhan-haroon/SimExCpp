cmake_minimum_required(VERSION 3.8)
project(husky_ur10_cam)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(gazebo_ros_pkgs REQUIRED)
find_package(xacro REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(std_srvs REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(helical_path src/helical_path.cpp)

ament_target_dependencies(
  helical_path
  rclcpp
  geometry_msgs
  moveit_ros_planning_interface
  std_srvs
)

install(TARGETS
  helical_path
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch worlds urdf config meshes models rviz
  DESTINATION share/${PROJECT_NAME}/
)

install(PROGRAMS
  scripts/roi_voxels.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
