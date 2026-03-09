#include <memory>
#include <vector>
#include <cmath>
#include <tf2/LinearMath/Quaternion.h>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_msgs/msg/robot_trajectory.hpp>

class HelicalScanner : public rclcpp::Node
{
public:
  HelicalScanner() : Node("helical_path")
  {
    service_ = this->create_service<std_srvs::srv::Trigger>(
        "/start_scan",
        std::bind(&HelicalScanner::start_scan, this,
                  std::placeholders::_1,
                  std::placeholders::_2));

    RCLCPP_INFO(this->get_logger(), "Helical scan service ready: /start_scan");
  }

private:
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr service_;

  void start_scan(
      const std::shared_ptr<std_srvs::srv::Trigger::Request>,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    using moveit::planning_interface::MoveGroupInterface;

    auto move_group =
        std::make_shared<MoveGroupInterface>(shared_from_this(), "ur10_ur_manipulator");

    move_group->setMaxVelocityScalingFactor(0.1);
    move_group->setMaxAccelerationScalingFactor(0.1);

    /* -------- WORKSPACE LIMIT -------- */

    move_group->setWorkspace(
        -1.5, -1.5, 0.6,   // min_x, min_y, min_z
         1.5,  1.5, 1.5    // max_x, max_y, max_z
    );

    /* -------- HELIX PARAMETERS -------- */

    double radius = 0.8;
    double z_min = 0.8;
    double z_max = 1.0;

    int turns = 2;
    int points_per_turn = 40;

    int N = turns * points_per_turn;

    std::vector<geometry_msgs::msg::Pose> waypoints;

    for (int i = 0; i < N; i++)
    {
      double theta = 2 * M_PI * i / points_per_turn;

      geometry_msgs::msg::Pose pose;

      pose.position.x = radius * cos(theta);
      pose.position.y = radius * sin(theta);

      double z = z_min + (z_max - z_min) * ((double)i / N);

      z = std::max(z_min, std::min(z, z_max));
      pose.position.z = z;

      double yaw = theta;
      double pitch = 0.0;
      double roll = -M_PI / 2.0;   // tilt camera forward

      tf2::Quaternion q;
      q.setRPY(roll, pitch, yaw);

      pose.orientation.x = q.x();
      pose.orientation.y = q.y();
      pose.orientation.z = q.z();
      pose.orientation.w = q.w();

      waypoints.push_back(pose);
    }

    const double eef_step = 0.02;
    const double jump_threshold = 0.0;

    /* -------- EXECUTE HELIX IN SEGMENTS -------- */

    for (size_t i = 0; i < waypoints.size() - 1; i++)
    {
      std::vector<geometry_msgs::msg::Pose> segment;

      segment.push_back(waypoints[i]);
      segment.push_back(waypoints[i + 1]);

      moveit_msgs::msg::RobotTrajectory trajectory;

      double fraction = move_group->computeCartesianPath(
          segment,
          eef_step,
          jump_threshold,
          trajectory,
          true);

      if (fraction > 0.9)
      {
        move_group->execute(trajectory);
      }
      else
      {
        RCLCPP_WARN(this->get_logger(),
                    "Skipping segment %ld (IK failure)", i);
      }
    }

    response->success = true;
    response->message = "Helical scan completed";

    RCLCPP_INFO(this->get_logger(), "Helical scan finished");
  }
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<HelicalScanner>();

  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}