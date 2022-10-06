#! /usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
def callback(data):
    br=CvBridge()
    rospy.loginfo("receiving msg")
    print(data)
    # cv2.imshow("img",br.imgmsg_to_cv2(data))
    # cv2.waitKey()
def listener():
    rospy.init_node('listener',anonymous=True)
    rospy.Subscriber('imager',String,callback)
    rospy.spin()
if __name__ == '__main__':
    listener()
    