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
    if (!std::isfinite(goal.command_timeout_s) ||
        goal.command_timeout_s <= 0.0 || goal.command_timeout_s > 1.0)
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] command_timeout_s must be in (0, 1]", name_.c_str());
        return false;
    }
    if (!std::isfinite(goal.max_duration_s) ||
        goal.max_duration_s < 0.0 || goal.max_duration_s > 300.0)
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] max_duration_s must be in [0, 300]", name_.c_str());
        return false;
    }
    if (!std::isfinite(goal.max_policy_target_delta_rad) ||
        goal.max_policy_target_delta_rad <= 0.0 ||
        goal.max_policy_target_delta_rad > 1.0)
    {
        RCLCPP_WARN(
            node_->get_logger(),
            "[%s] max_policy_target_delta_rad must be in (0, 1]",
            name_.c_str());
        return false;
    }
    if (!std::isfinite(goal.max_actuator_step_rad) ||
        goal.max_actuator_step_rad <= 0.0 || goal.max_actuator_step_rad > 0.01)
    {
        RCLCPP_WARN(
            node_->get_logger(),
            "[%s] max_actuator_step_rad must be in (0, 0.01]",
            name_.c_str());
        return false;
    }
    if (!std::isfinite(goal.joint_velocity_scale) ||
        goal.joint_velocity_scale <= 0.0 || goal.joint_velocity_scale > 1.0)
    {
        RCLCPP_WARN(
            node_->get_logger(),
            "[%s] joint_velocity_scale must be in (0, 1]",
            name_.c_str());
        return false;
    }
    if (!std::isfinite(goal.joint_acceleration_scale) ||
        goal.joint_acceleration_scale <= 0.0 || goal.joint_acceleration_scale > 1.0)
    {
        RCLCPP_WARN(
            node_->get_logger(),
            "[%s] joint_acceleration_scale must be in (0, 1]",
            name_.c_str());
        return false;
    }
    if (goal.isaac_relative_control)
    {
        if (!dual)
        {
            RCLCPP_WARN(
                node_->get_logger(), "[%s] Isaac-relative control is reach-specific and requires robot_name 'dual'",
                name_.c_str());
            return false;
        }
        if (!model_updater_.HasEffortCommandInterface())
        {
            RCLCPP_WARN(
                node_->get_logger(), "[%s] Isaac-relative control requires an effort command interface",
                name_.c_str());
            return false;
        }
        if (!std::isfinite(goal.relative_target_refresh_hz) ||
            goal.relative_target_refresh_hz <= 0.0 ||
            goal.relative_target_refresh_hz > 1000.0)
        {
            RCLCPP_WARN(
                node_->get_logger(), "[%s] relative_target_refresh_hz must be in (0, 1000]",
                name_.c_str());
            return false;
        }
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
    max_policy_target_delta_rad_ = goal.max_policy_target_delta_rad;
    max_actuator_step_rad_ = goal.max_actuator_step_rad;
    joint_velocity_scale_ = goal.joint_velocity_scale;
    joint_acceleration_scale_ = goal.joint_acceleration_scale;
    isaac_relative_control_ = goal.isaac_relative_control;
    relative_target_refresh_hz_ = goal.relative_target_refresh_hz;
    control_gripper_ = goal.control_gripper;
}

void PolicyControl::onStart()
{
    fr3_model_updater_.setInitFromCurrent();
    q_hold_ = fr3_model_updater_.q_total_;
    q_target_ = q_hold_;
    q_desired_ = q_hold_;
    qdot_desired_ = Eigen::VectorXd::Zero(model_updater_.manipulator_dof_);
    q_offset_ = Eigen::VectorXd::Zero(model_updater_.manipulator_dof_);
    activation_time_s_ = node_->now().seconds();
    next_relative_refresh_time_s_ = activation_time_s_;
    last_command_time_s_ = activation_time_s_;
    last_sequence_ = 0;
    has_command_ = false;
    last_gripper_state_ = -1;
    result_message_ = "running";

    RCLCPP_INFO(
        node_->get_logger(),
        "[%s] started robot=%s timeout=%.3fs max_duration=%.3fs "
        "policy_delta=%.3frad actuator_step=%.4frad velocity_scale=%.2f "
        "acceleration_scale=%.2f mode=%s relative_refresh=%.1fHz gripper=%s",
        name_.c_str(), controlled_robot_.c_str(), command_timeout_s_, max_duration_s_,
        max_policy_target_delta_rad_, max_actuator_step_rad_, joint_velocity_scale_,
        joint_acceleration_scale_, isaac_relative_control_ ? "isaac-relative" : "rate-limited-absolute",
        isaac_relative_control_ ? relative_target_refresh_hz_ : 0.0,
        control_gripper_ ? "true" : "false");
}

void PolicyControl::commandCallback(
    const fr3_husky_msgs::msg::PolicyJointCommand::SharedPtr msg)
{
    CommandData command;
    command.dual = msg->robot_name == "dual";
    command.robot_index = command.dual ? 0 : resolveRobotIndex(msg->robot_name);
    command.sequence = msg->sequence;
    command.gripper_action = msg->gripper_action;
    command.relative_position_offsets = msg->relative_position_offsets;
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
    if (command.relative_position_offsets != isaac_relative_control_)
    {
        error = "policy command relative/absolute semantics do not match active goal";
        return false;
    }

    const Eigen::Index offset = static_cast<Eigen::Index>(controlled_robot_index_ * FR3_DOF);
    for (size_t i = 0; i < controlled_dof_; ++i)
    {
        const double requested = command.target_positions[i];
        const double measured =
            fr3_model_updater_.q_total_(offset + static_cast<Eigen::Index>(i));
        const double target = command.relative_position_offsets
            ? measured + requested
            : requested;
        const size_t joint_index = i % FR3_DOF;
        if (target < kLowerLimits[joint_index] + kJointLimitMargin ||
            target > kUpperLimits[joint_index] - kJointLimitMargin)
        {
            error = "joint" + std::to_string(i + 1) + " target violates position limit margin";
            return false;
        }
        const double step = std::abs(target - measured);
        if (step > max_policy_target_delta_rad_)
        {
            error = "joint" + std::to_string(i + 1) +
                    " policy target delta exceeds max_policy_target_delta_rad";
            return false;
        }
    }
    return true;
}

void PolicyControl::updateRateLimitedTarget(double period_s)
{
    if (!std::isfinite(period_s) || period_s <= 0.0)
    {
        period_s = kNominalControlPeriodS;
    }
    period_s = std::min(period_s, kMaxLimiterPeriodS);

    const Eigen::Index offset =
        static_cast<Eigen::Index>(controlled_robot_index_ * FR3_DOF);
    for (size_t i = 0; i < controlled_dof_; ++i)
    {
        const Eigen::Index index = offset + static_cast<Eigen::Index>(i);
        const size_t joint_index = i % FR3_DOF;
        const double max_velocity = std::min(
            kMaxJointVelocities[joint_index] * joint_velocity_scale_,
            max_actuator_step_rad_ / period_s);
        const double max_acceleration =
            kMaxJointAccelerations[joint_index] * joint_acceleration_scale_;
        const double error = q_target_(index) - q_desired_(index);
        const double current_velocity = qdot_desired_(index);

        if (std::abs(error) <= kPositionTolerance &&
            std::abs(current_velocity) <= max_acceleration * period_s)
        {
            q_desired_(index) = q_target_(index);
            qdot_desired_(index) = 0.0;
            continue;
        }

        const double braking_velocity =
            std::sqrt(2.0 * max_acceleration * std::abs(error));
        const double target_velocity = std::copysign(
            std::min(max_velocity, braking_velocity), error);
        const double velocity_delta = std::clamp(
            target_velocity - current_velocity,
            -max_acceleration * period_s,
            max_acceleration * period_s);
        const double next_velocity = std::clamp(
            current_velocity + velocity_delta, -max_velocity, max_velocity);
        const double max_step = std::min(
            max_actuator_step_rad_, max_velocity * period_s);
        const double position_step = std::clamp(
            next_velocity * period_s, -max_step, max_step);

        if (position_step * error > 0.0 &&
            std::abs(position_step) >= std::abs(error))
        {
            q_desired_(index) = q_target_(index);
            qdot_desired_(index) = 0.0;
        }
        else
        {
            q_desired_(index) += position_step;
            qdot_desired_(index) = position_step / period_s;
        }
    }
}

bool PolicyControl::refreshIsaacRelativeTarget(std::string& error)
{
    const Eigen::Index offset =
        static_cast<Eigen::Index>(controlled_robot_index_ * FR3_DOF);
    for (size_t i = 0; i < controlled_dof_; ++i)
    {
        const Eigen::Index index = offset + static_cast<Eigen::Index>(i);
        const size_t joint_index = i % FR3_DOF;
        const double target = fr3_model_updater_.q_total_(index) + q_offset_(index);
        if (target < kLowerLimits[joint_index] + kJointLimitMargin ||
            target > kUpperLimits[joint_index] - kJointLimitMargin)
        {
            error = "joint" + std::to_string(i + 1) +
                    " refreshed relative target violates position limit margin";
            return false;
        }
        q_target_(index) = target;
    }
    return true;
}

void PolicyControl::writeIsaacEffortCommand()
{
    fr3_model_updater_.torque_desired_total_.setZero();
    const Eigen::Index offset =
        static_cast<Eigen::Index>(controlled_robot_index_ * FR3_DOF);
    for (size_t i = 0; i < controlled_dof_; ++i)
    {
        const Eigen::Index index = offset + static_cast<Eigen::Index>(i);
        const size_t joint_index = i % FR3_DOF;
        const double effort =
            kIsaacStiffness[joint_index] *
                (q_desired_(index) - fr3_model_updater_.q_total_(index)) +
            kIsaacDamping[joint_index] *
                (qdot_desired_(index) - fr3_model_updater_.qdot_total_(index));
        fr3_model_updater_.torque_desired_total_(index) = std::clamp(
            effort,
            -kIsaacEffortLimits[joint_index],
            kIsaacEffortLimits[joint_index]);
    }
    // The Franka hardware layer provides gravity compensation. This matches
    // the training asset, where robot-link gravity is disabled.
    fr3_model_updater_.writeCommand(fr3_model_updater_.torque_desired_total_);
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
    const rclcpp::Time& time, const rclcpp::Duration& period)
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
            const Eigen::Index index = offset + static_cast<Eigen::Index>(i);
            if (isaac_relative_control_)
            {
                q_offset_(index) = command->target_positions[i];
            }
            else
            {
                q_target_(index) = command->target_positions[i];
            }
        }
        if (isaac_relative_control_)
        {
            if (!refreshIsaacRelativeTarget(error))
            {
                result_message_ = error;
                RCLCPP_ERROR(node_->get_logger(), "[%s] %s", name_.c_str(), error.c_str());
                return ComputeResult::ABORTED;
            }
            next_relative_refresh_time_s_ =
                now_s + 1.0 / relative_target_refresh_hz_;
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

    if (isaac_relative_control_)
    {
        if (now_s >= next_relative_refresh_time_s_)
        {
            std::string error;
            if (!refreshIsaacRelativeTarget(error))
            {
                result_message_ = error;
                RCLCPP_ERROR(node_->get_logger(), "[%s] %s", name_.c_str(), error.c_str());
                return ComputeResult::ABORTED;
            }
            const double refresh_period_s = 1.0 / relative_target_refresh_hz_;
            do
            {
                next_relative_refresh_time_s_ += refresh_period_s;
            }
            while (next_relative_refresh_time_s_ <= now_s);
        }
        // Keep the measured-relative target refresh at its configured rate,
        // but advance the PD position/velocity reference through the existing
        // per-control-cycle velocity, acceleration, and step limiter.
        updateRateLimitedTarget(period.seconds());
        writeIsaacEffortCommand();
    }
    else
    {
        updateRateLimitedTarget(period.seconds());
        writeDesiredCommand(q_desired_, qdot_desired_);
    }

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
