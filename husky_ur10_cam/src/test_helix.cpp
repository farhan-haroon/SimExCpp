#include <memory>
#include <vector>
#include <cmath>
#include <chrono>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <tf2/LinearMath/Quaternion.h>

#include <moveit/move_group_interface/move_group_interface.h>

class HelixScanner : public rclcpp::Node
{
public:
  HelixScanner() : Node("helix_scanner")
  {
    service_ = this->create_service<std_srvs::srv::Trigger>(
        "/start_scan",
        std::bind(&HelixScanner::start_scan, this,
                  std::placeholders::_1,
                  std::placeholders::_2));

    RCLCPP_INFO(this->get_logger(), "Helix scanner service ready.");
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

    /* ---------- Planner Settings ---------- */

    move_group->setPlanningTime(5.0);
    move_group->setNumPlanningAttempts(5);
    move_group->setMaxVelocityScalingFactor(0.5);
    move_group->setMaxAccelerationScalingFactor(0.5);

    /* ---------- HELIX PARAMETERS ---------- */

    double radius = 0.80;   // reduced radius to avoid robot body
    double z_min = 0.8;
    double z_max = 1.2;

    int turns = 3;
    int points_per_turn = 4;
    int N = turns * points_per_turn;

    std::vector<geometry_msgs::msg::Pose> waypoints;

    for (int i = 0; i < N; i++)
    {
      double theta = 2 * M_PI * i / points_per_turn;

      geometry_msgs::msg::Pose pose;

      pose.position.x = radius * cos(theta);
      pose.position.y = radius * sin(theta);

      double z = z_min + (z_max - z_min) * ((double)i / N);
      pose.position.z = z;

      double yaw = theta;
      double pitch = 0.0;
      double roll = -M_PI / 2.0;

      tf2::Quaternion q;
      q.setRPY(roll, pitch, yaw);

      pose.orientation.x = q.x();
      pose.orientation.y = q.y();
      pose.orientation.z = q.z();
      pose.orientation.w = q.w();

      waypoints.push_back(pose);
    }

    /* ---------- Execute Waypoints Safely ---------- */

    for (size_t i = 0; i < waypoints.size(); i++)
    {
      move_group->setStartStateToCurrentState();

      move_group->setPoseTarget(waypoints[i]);

      MoveGroupInterface::Plan plan;

      bool success =
          (move_group->plan(plan) ==
           moveit::core::MoveItErrorCode::SUCCESS);

      if (success)
      {
        RCLCPP_INFO(this->get_logger(),
                    "Executing waypoint %ld", i);

        move_group->execute(plan);

        rclcpp::sleep_for(std::chrono::milliseconds(200));
      }
      else
      {
        RCLCPP_WARN(this->get_logger(),
                    "Planning failed for waypoint %ld", i);
      }
    }

    response->success = true;
    response->message = "Helical scan completed";

    RCLCPP_INFO(this->get_logger(), "Helical scan finished.");
  }
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<HelixScanner>();

  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}