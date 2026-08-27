#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>

#include <Eigen/Core>
#include <fr3_husky_msgs/action/run_policy_control.hpp>
#include <fr3_husky_msgs/msg/policy_joint_command.hpp>
#include <rclcpp/rclcpp.hpp>
#include <realtime_tools/realtime_buffer.h>

#include <fr3_husky_controller/model/fr3_model_updater.hpp>
#include <fr3_husky_controller/servers/action_server_base.hpp>

namespace fr3_husky_controller::servers::fr3
{

class PolicyControl final : public ActionServerBase<fr3_husky_msgs::action::RunPolicyControl>
{
public:
    using ActionT = fr3_husky_msgs::action::RunPolicyControl;
    using Base = ActionServerBase<ActionT>;
    using ComputeResult = typename Base::ComputeResult;
    using StopReason = typename Base::StopReason;
    using ResultPtr = typename Base::ResultPtr;

    PolicyControl(const std::string& name, const NodePtr& node, ModelUpdaterBase& model_updater);
    ~PolicyControl() override = default;

    int priority() const override { return 9; }
    bool allowPreemption() const override { return false; }

private:
    struct CommandData
    {
        std::array<double, FR3_DOF> target_positions{};
        double gripper_action{0.0};
        double received_time_s{-1.0};
        uint64_t sequence{0};
        int robot_index{-1};
        bool valid{false};
    };

    bool acceptGoal(const ActionT::Goal& goal) override;
    void onGoalAccepted(const ActionT::Goal& goal) override;
    void onStart() override;
    ComputeResult compute(const rclcpp::Time& time, const rclcpp::Duration& period) override;
    void onStop(StopReason reason) override;
    ResultPtr makeResult(StopReason reason) override;

    void commandCallback(const fr3_husky_msgs::msg::PolicyJointCommand::SharedPtr msg);
    int resolveRobotIndex(const std::string& robot_name) const;
    bool validateTarget(const CommandData& command, std::string& error) const;
    void writeDesiredCommand(const Eigen::VectorXd& q_desired, const Eigen::VectorXd& qdot_desired);

    FR3ModelUpdater& fr3_model_updater_;
    rclcpp::Subscription<fr3_husky_msgs::msg::PolicyJointCommand>::SharedPtr command_sub_;
    realtime_tools::RealtimeBuffer<CommandData> command_buffer_;

    std::string controlled_robot_;
    int controlled_robot_index_{-1};
    double command_timeout_s_{0.15};
    double max_duration_s_{8.0};
    double max_target_step_rad_{0.15};
    bool control_gripper_{true};

    double activation_time_s_{0.0};
    double last_command_time_s_{0.0};
    uint64_t last_sequence_{0};
    bool has_command_{false};
    int last_gripper_state_{-1};
    Eigen::VectorXd q_hold_;
    Eigen::VectorXd q_desired_;

    std::string result_message_{"not started"};

    static constexpr std::array<double, FR3_DOF> kLowerLimits{
        -2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508};
    static constexpr std::array<double, FR3_DOF> kUpperLimits{
        2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508};
    static constexpr double kJointLimitMargin = 0.02;
};

}  // namespace fr3_husky_controller::servers::fr3
