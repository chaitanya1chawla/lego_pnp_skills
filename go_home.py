from yk_msgs.srv import SetJoints, SetJointsRequest
import rospy

ROBOT_NAMESPACE = "yk_destroyer"
PI = 3.14159265358979323846

def go_to_home():
    print("GO TO HOME")

    # Create a service proxy to call the set_joints service
    rospy.wait_for_service(f'/{ROBOT_NAMESPACE}/yk_set_joints')
    set_joints = rospy.ServiceProxy(f'/{ROBOT_NAMESPACE}/yk_set_joints', SetJoints)

    # Create a request object
    request = SetJointsRequest()
    request.state.name = ["joint_1_s", "joint_2_l", "joint_3_u", "joint_4_r", "joint_5_b", "joint_6_t"]
    request.state.position = [0, 0, 0, 0, -PI/2, 0]

    # Call the service and get the response
    try:
        response = set_joints(request)
    except rospy.ServiceException as e:
        print("Service call failed")
        return None
    
if __name__ == "__main__":
    go_to_home()