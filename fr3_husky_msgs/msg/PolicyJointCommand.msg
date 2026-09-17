std_msgs/Header header
uint64 sequence
string robot_name
# 7 values for a single arm, or 14 values (left then right) for robot_name="dual".
# When relative_position_offsets is false these are absolute joint-position
# targets held until the next policy command. When it is true they are offsets
# from the latest measured joint positions and the controller refreshes the
# absolute targets at every controller step.
float64[] target_positions
bool relative_position_offsets
float64 gripper_action
