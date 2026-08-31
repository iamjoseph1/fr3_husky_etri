std_msgs/Header header
uint64 sequence
string robot_name
# 7 values for a single arm, or 14 values (left then right) for robot_name="dual".
float64[] target_positions
float64 gripper_action
