#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def callback(msg):
    bridge = CvBridge()
    try:
        # Convert ROS Image to OpenCV image
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Save the image
        cv2.imwrite("image_saved.jpg", cv_image)
        rospy.loginfo("Image saved!")
        
        # Shutdown after saving one image
        rospy.signal_shutdown("Image saved")
    except Exception as e:
        rospy.logerr(f"Failed to convert or save image: {e}")

if __name__ == "__main__":
    rospy.init_node("image_saver")
    rospy.Subscriber("/og_frame", Image, callback)
    rospy.spin()
