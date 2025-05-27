import rospy
import sys
import copy

import numpy as np
from geometry_msgs.msg import Pose, PoseStamped
from yk_msgs.srv import SetPose, SetPoseRequest, SetPoseResponse

from scipy.spatial.transform import Rotation as R

ROBOT_NAMESPACE = "yk_destroyer"
# XTRAVEL_G = 0.002 -0.012 # 0.025 # m
# YTRAVEL_G = 0.0179 - 0.0279 #0.0369 - 0.045 # 0.025 # m
XTRAVEL_G = -0.0036 -0.012 # 0.025 # m
YTRAVEL_G = -0.0009 - 0.0009 #0.0369 - 0.045 # 0.025 # m
ZTRAVEL_G = 0.234 - 0.045   # Aruco marker depth - box height (in meters) 

R_OFFSET = np.eye(3)

SPEED_FACTOR = 0.1

def move_robot():    
        
    # Get the current pose, force readings of the robot
    current_pose = rospy.wait_for_message(f'/{ROBOT_NAMESPACE}/tool0_pose', PoseStamped)
    current_pose = current_pose.pose
    
    # Create a service proxy to call the find_edge_x service
    rospy.wait_for_service(f'/{ROBOT_NAMESPACE}/yk_set_pose')
    set_pose = rospy.ServiceProxy(f'/{ROBOT_NAMESPACE}/yk_set_pose', SetPose)
    
    # Create a request object
    request = SetPoseRequest()
    
    # SET FRAME TRANSLATIONAL OFFSET
    request.pose = current_pose
    T_w_g0 = np.eye(4)
    T_w_g1 = np.eye(4)
    T_g0_g1 = np.eye(4)
    
    T_w_g0[:3,:3] = R.from_quat([current_pose.orientation.x, 
                                 current_pose.orientation.y, 
                                 current_pose.orientation.z, 
                                 current_pose.orientation.w]).as_matrix()
    T_w_g0[:3,3] = [current_pose.position.x, current_pose.position.y, current_pose.position.z]
    T_g0_g1[:3,3] = [XTRAVEL_G, YTRAVEL_G, ZTRAVEL_G]
    
    T_w_g1 = np.dot(T_w_g0, T_g0_g1)
    
    T_w_g1_quat = R.from_matrix(T_w_g1[:3,:3]).as_quat()
    T_w_g1_pos = T_w_g1[:3,3]
    
    request.pose.position.x = T_w_g1_pos[0]
    request.pose.position.y = T_w_g1_pos[1]
    request.pose.position.z = T_w_g1_pos[2]
    
    request.pose.orientation.x = T_w_g1_quat[0]
    request.pose.orientation.y = T_w_g1_quat[1]
    request.pose.orientation.z = T_w_g1_quat[2]
    request.pose.orientation.w = T_w_g1_quat[3]
    
    print("Target Position in world frame: ", T_w_g1_pos)
    
    # SET FRAME ROTATIONAL OFFSET
    current_orient_quat = current_pose.orientation
    current_orient_mat = R.from_quat([current_orient_quat.x, current_orient_quat.y, current_orient_quat.z, current_orient_quat.w]).as_matrix()
    target_orient_mat = np.dot(current_orient_mat, R_OFFSET)
    target_orient_quat = R.from_matrix(target_orient_mat).as_quat()
    request.pose.orientation.x = target_orient_quat[0]
    request.pose.orientation.y = target_orient_quat[1]
    request.pose.orientation.z = target_orient_quat[2]
    request.pose.orientation.w = target_orient_quat[3]
    
    request.max_velocity_scaling_factor = SPEED_FACTOR
    request.max_acceleration_scaling_factor = SPEED_FACTOR
        
    # Call the service and get the response
    try:
        response = set_pose(request)
    except rospy.ServiceException as e:
        print("Service call failed")
    

if __name__ == '__main__':
    try:
        rospy.init_node('test_move_yk')
        move_robot()
    except rospy.ROSInterruptException:
        print("program interrupted before completion", file=sys.stderr)