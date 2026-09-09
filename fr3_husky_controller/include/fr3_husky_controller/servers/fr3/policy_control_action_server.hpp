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
    bool requiresRobotDataUpdate() const override { return !isaac_relative_control_; }
    bool allowPreemption() const override { return false; }

private:
    struct CommandData
    {
        std::array<double, 2 * FR3_DOF> target_positions{};
        size_t target_count{0};
        double gripper_action{0.0};
        double received_time_s{-1.0};
        uint64_t sequence{0};
        int robot_index{-1};
        bool dual{false};
        bool relative_position_offsets{false};
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
    void updateRateLimitedTarget(double period_s);
    void refreshIsaacRelativeTarget();
    void writeDesiredCommand(const Eigen::VectorXd& q_desired, const Eigen::VectorXd& qdot_desired);
    void writeIsaacEffortCommand(double period_s);

    FR3ModelUpdater& fr3_model_updater_;
    rclcpp::Subscription<fr3_husky_msgs::msg::PolicyJointCommand>::SharedPtr command_sub_;
    realtime_tools::RealtimeBuffer<CommandData> command_buffer_;

    std::string controlled_robot_;
    int controlled_robot_index_{-1};
    size_t controlled_dof_{FR3_DOF};
    bool controlled_dual_{false};
    double command_timeout_s_{0.15};
    double max_duration_s_{8.0};
    double max_policy_target_delta_rad_{1.0};
    double max_actuator_step_rad_{0.001};
    double joint_velocity_scale_{0.10};
    double joint_acceleration_scale_{0.20};
    bool isaac_relative_control_{false};
    bool control_gripper_{true};

    double activation_time_s_{0.0};
    double last_command_time_s_{0.0};
    uint64_t last_sequence_{0};
    bool has_command_{false};
    int last_gripper_state_{-1};
    Eigen::VectorXd q_hold_;
    Eigen::VectorXd q_target_;
    Eigen::VectorXd q_desired_;
    Eigen::VectorXd qdot_desired_;
    Eigen::VectorXd q_offset_;
    // Matches dual_fr3_lab TorqueRateLimitedPDActuator.  The state is reset
    // at policy start and is used only by the Isaac-relative effort path.
    Eigen::VectorXd previous_applied_effort_;

    std::string result_message_{"not started"};

    static constexpr std::array<double, FR3_DOF> kLowerLimits{
        -2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508};
    static constexpr std::array<double, FR3_DOF> kUpperLimits{
        2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508};
    // franka_description/robots/fr3/joint_limits.yaml
    static constexpr std::array<double, FR3_DOF> kMaxJointVelocities{
        2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26};
    // fr3_husky_moveit_config/config/dual/dual_fr3_joint_limits.yaml
    static constexpr std::array<double, FR3_DOF> kMaxJointAccelerations{
        3.75, 1.875, 2.5, 3.125, 3.75, 5.0, 5.0};
    // dual_fr3_lab DUAL_FR3_PLATE_EE_CFG implicit actuator parameters.
    static constexpr std::array<double, FR3_DOF> kIsaacStiffness{
        80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0};
    static constexpr std::array<double, FR3_DOF> kIsaacDamping{
        4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0};
    static constexpr std::array<double, FR3_DOF> kIsaacEffortLimits{
        87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0};
    static constexpr double kIsaacTorqueRateLimit = 1000.0;  // Nm/s
    static constexpr double kJointLimitMargin = 0.02;
    static constexpr double kPositionTolerance = 1.0e-6;
    static constexpr double kNominalControlPeriodS = 0.001;
    static constexpr double kMaxLimiterPeriodS = kNominalControlPeriodS;
};

}  // namespace fr3_husky_controller::servers::fr3
