#include <fr3_husky_controller/servers/fr3/policy_control_action_server.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fr3_husky_controller::servers::fr3
{

namespace
{
FR3ModelUpdater& getFR3ModelUpdater(ModelUpdaterBase& model_updater, const std::string& server_name)
{
    auto* updater = dynamic_cast<FR3ModelUpdater*>(&model_updater);
    if (!updater)
    {
        throw std::runtime_error("[" + server_name + "] requires FR3ModelUpdater");
    }
    return *updater;
}
}  // namespace

PolicyControl::PolicyControl(
    const std::string& name, const NodePtr& node, ModelUpdaterBase& model_updater)
: Base(name, node, model_updater),
  fr3_model_updater_(getFR3ModelUpdater(model_updater, name))
{
    command_buffer_.writeFromNonRT(CommandData{});
    command_sub_ = node_->create_subscription<fr3_husky_msgs::msg::PolicyJointCommand>(
        "policy_joint_command",
        rclcpp::QoS(1).best_effort(),
        std::bind(&PolicyControl::commandCallback, this, std::placeholders::_1));

    RCLCPP_INFO(
        node_->get_logger(),
        "[%s] PolicyControl created; command topic: policy_joint_command",
        name_.c_str());
}

int PolicyControl::resolveRobotIndex(const std::string& robot_name) const
{
    for (size_t i = 0; i < model_updater_.robot_names_.size(); ++i)
    {
        if (model_updater_.robot_names_[i] == robot_name)
        {
            return static_cast<int>(i);
        }
    }
    return -1;
}

bool PolicyControl::acceptGoal(const ActionT::Goal& goal)
{
    const bool dual = goal.robot_name == "dual";
    if (dual && (model_updater_.robot_names_.size() != 2 ||
                 model_updater_.manipulator_dof_ != 2 * FR3_DOF))
    {
        RCLCPP_WARN(
            node_->get_logger(), "[%s] robot_name 'dual' requires exactly two FR3 arms",
            name_.c_str());
        return false;
    }
    if (!dual && resolveRobotIndex(goal.robot_name) < 0)
    {
        RCLCPP_WARN(
            node_->get_logger(), "[%s] Unknown robot_name '%s'",
            name_.c_str(), goal.robot_name.c_str());
        return false;
    }
    if (dual && goal.control_gripper)
    {
        RCLCPP_WARN(
            node_->get_logger(), "[%s] dual policy control does not support grippers",
            name_.c_str());
        return false;
    }
    if (goal.command_timeout_s <= 0.0 || goal.command_timeout_s > 1.0)
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] command_timeout_s must be in (0, 1]", name_.c_str());
        return false;
    }
    if (goal.max_duration_s < 0.0 || goal.max_duration_s > 300.0)
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] max_duration_s must be in [0, 300]", name_.c_str());
        return false;
    }
    if (goal.max_target_step_rad <= 0.0 || goal.max_target_step_rad > 1.0)
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] max_target_step_rad must be in (0, 1]", name_.c_str());
        return false;
    }
    return true;
}

void PolicyControl::onGoalAccepted(const ActionT::Goal& goal)
{
    controlled_robot_ = goal.robot_name;
    controlled_dual_ = goal.robot_name == "dual";
    controlled_robot_index_ = controlled_dual_ ? 0 : resolveRobotIndex(goal.robot_name);
    controlled_dof_ = controlled_dual_ ? 2 * FR3_DOF : FR3_DOF;
    command_timeout_s_ = goal.command_timeout_s;
    max_duration_s_ = goal.max_duration_s;
    max_target_step_rad_ = goal.max_target_step_rad;
    control_gripper_ = goal.control_gripper;
}

void PolicyControl::onStart()
{
    fr3_model_updater_.setInitFromCurrent();
    q_hold_ = fr3_model_updater_.q_total_;
    q_desired_ = q_hold_;
    activation_time_s_ = node_->now().seconds();
    last_command_time_s_ = activation_time_s_;
    last_sequence_ = 0;
    has_command_ = false;
    last_gripper_state_ = -1;
    result_message_ = "running";

    RCLCPP_INFO(
        node_->get_logger(),
        "[%s] started robot=%s timeout=%.3fs max_duration=%.3fs max_step=%.3frad gripper=%s",
        name_.c_str(), controlled_robot_.c_str(), command_timeout_s_, max_duration_s_,
        max_target_step_rad_, control_gripper_ ? "true" : "false");
}

void PolicyControl::commandCallback(
    const fr3_husky_msgs::msg::PolicyJointCommand::SharedPtr msg)
{
    CommandData command;
    command.dual = msg->robot_name == "dual";
    command.robot_index = command.dual ? 0 : resolveRobotIndex(msg->robot_name);
    command.sequence = msg->sequence;
    command.gripper_action = msg->gripper_action;
    command.received_time_s = node_->now().seconds();
    command.target_count = msg->target_positions.size();

    const size_t expected_count = command.dual ? 2 * FR3_DOF : FR3_DOF;
    command.valid = command.robot_index >= 0 &&
                    command.target_count == expected_count &&
                    command.target_count <= command.target_positions.size() &&
                    std::isfinite(command.gripper_action);

    for (size_t i = 0;
         i < std::min(command.target_count, command.target_positions.size()); ++i)
    {
        command.target_positions[i] = msg->target_positions[i];
        command.valid = command.valid && std::isfinite(command.target_positions[i]);
    }
    command_buffer_.writeFromNonRT(command);
}

bool PolicyControl::validateTarget(const CommandData& command, std::string& error) const
{
    if (!command.valid)
    {
        error = "invalid policy command dimension or non-finite value";
        return false;
    }
    if (command.robot_index != controlled_robot_index_ ||
        command.dual != controlled_dual_ ||
        command.target_count != controlled_dof_)
    {
        error = "policy command dimension/robot_name does not match active goal";
        return false;
    }

    const Eigen::Index offset = static_cast<Eigen::Index>(controlled_robot_index_ * FR3_DOF);
    for (size_t i = 0; i < controlled_dof_; ++i)
    {
        const double target = command.target_positions[i];
        const size_t joint_index = i % FR3_DOF;
        if (target < kLowerLimits[joint_index] + kJointLimitMargin ||
            target > kUpperLimits[joint_index] - kJointLimitMargin)
        {
            error = "joint" + std::to_string(i + 1) + " target violates position limit margin";
            return false;
        }
        const double step = std::abs(
            target - fr3_model_updater_.q_total_(offset + static_cast<Eigen::Index>(i)));
        if (step > max_target_step_rad_)
        {
            error = "joint" + std::to_string(i + 1) + " target step exceeds max_target_step_rad";
            return false;
        }
    }
    return true;
}

void PolicyControl::writeDesiredCommand(
    const Eigen::VectorXd& q_desired, const Eigen::VectorXd& qdot_desired)
{
    if (model_updater_.HasEffortCommandInterface())
    {
        if (!fr3_model_updater_.robot_controller_)
        {
            throw std::runtime_error("robot_controller is null");
        }
        const Eigen::VectorXd torque =
            fr3_model_updater_.robot_controller_->moveJointTorqueStep(
                q_desired, qdot_desired, false);
        fr3_model_updater_.torque_desired_total_ = torque - fr3_model_updater_.g_total_;
        fr3_model_updater_.writeCommand(fr3_model_updater_.torque_desired_total_);
    }
    else if (model_updater_.HasVelocityCommandInterface())
    {
        fr3_model_updater_.qdot_desired_total_ = qdot_desired;
        fr3_model_updater_.writeCommand(fr3_model_updater_.qdot_desired_total_);
    }
    else if (model_updater_.HasPositionCommandInterface())
    {
        fr3_model_updater_.q_desired_total_ = q_desired;
        fr3_model_updater_.writeCommand(fr3_model_updater_.q_desired_total_);
    }
    else
    {
        throw std::runtime_error("no supported manipulator command interface");
    }
}

PolicyControl::ComputeResult PolicyControl::compute(
    const rclcpp::Time& time, const rclcpp::Duration& /*period*/)
{
    const double now_s = time.seconds();
    const CommandData* command = command_buffer_.readFromRT();

    if (command && command->received_time_s >= activation_time_s_ &&
        command->sequence > last_sequence_)
    {
        std::string error;
        if (!validateTarget(*command, error))
        {
            result_message_ = error;
            RCLCPP_ERROR(node_->get_logger(), "[%s] %s", name_.c_str(), error.c_str());
            return ComputeResult::ABORTED;
        }

        const Eigen::Index offset =
            static_cast<Eigen::Index>(controlled_robot_index_ * FR3_DOF);
        for (size_t i = 0; i < controlled_dof_; ++i)
        {
            q_desired_(offset + static_cast<Eigen::Index>(i)) =
                command->target_positions[i];
        }
        last_sequence_ = command->sequence;
        last_command_time_s_ = command->received_time_s;
        has_command_ = true;

        if (control_gripper_ && !controlled_dual_)
        {
            const int gripper_state = command->gripper_action < 0.0 ? 1 : 0;
            if (gripper_state != last_gripper_state_)
            {
                const bool ok = gripper_state == 1
                    ? fr3_model_updater_.GripperGrasp(controlled_robot_)
                    : fr3_model_updater_.GripperOpen(controlled_robot_);
                if (!ok)
                {
                    RCLCPP_WARN(
                        node_->get_logger(), "[%s] failed to send gripper command",
                        name_.c_str());
                }
                last_gripper_state_ = gripper_state;
            }
        }
    }

    const double command_age_s = now_s - last_command_time_s_;
    if (command_age_s > command_timeout_s_)
    {
        result_message_ =
            has_command_ ? "policy command timeout" : "no policy command received";
        RCLCPP_ERROR(
            node_->get_logger(), "[%s] %s (age %.3fs)",
            name_.c_str(), result_message_.c_str(), command_age_s);
        return ComputeResult::ABORTED;
    }

    if (max_duration_s_ > 0.0 &&
        now_s - activation_time_s_ >= max_duration_s_)
    {
        result_message_ = "maximum policy duration reached";
        return ComputeResult::SUCCEEDED;
    }

    const Eigen::VectorXd qdot_desired =
        Eigen::VectorXd::Zero(model_updater_.manipulator_dof_);
    writeDesiredCommand(q_desired_, qdot_desired);

    double max_joint_error = 0.0;
    const Eigen::Index offset =
        static_cast<Eigen::Index>(controlled_robot_index_ * FR3_DOF);
    for (size_t i = 0; i < controlled_dof_; ++i)
    {
        max_joint_error = std::max(
            max_joint_error,
            std::abs(
                q_desired_(offset + static_cast<Eigen::Index>(i)) -
                fr3_model_updater_.q_total_(offset + static_cast<Eigen::Index>(i))));
    }

    auto feedback = std::make_shared<ActionT::Feedback>();
    feedback->command_age_s = command_age_s;
    feedback->max_joint_error = max_joint_error;
    feedback->last_sequence = last_sequence_;
    feedback->status = has_command_ ? "tracking" : "waiting for first command";
    publishFeedback(feedback);
    return ComputeResult::RUNNING;
}

void PolicyControl::onStop(StopReason reason)
{
    model_updater_.haltCommands();
    if (reason == StopReason::CANCELED)
    {
        result_message_ = "policy control canceled";
    }
    else if (reason == StopReason::ABORTED && result_message_ == "running")
    {
        result_message_ = "policy control aborted";
    }
    RCLCPP_INFO(
        node_->get_logger(), "[%s] stopped: %s",
        name_.c_str(), result_message_.c_str());
}

PolicyControl::ResultPtr PolicyControl::makeResult(StopReason reason)
{
    auto result = std::make_shared<ActionT::Result>();
    result->success = reason == StopReason::SUCCEEDED;
    result->message = result_message_;
    result->last_sequence = last_sequence_;
    return result;
}

REGISTER_FR3_ACTION_SERVER(PolicyControl, "fr3_policy_control")

}  // namespace fr3_husky_controller::servers::fr3
