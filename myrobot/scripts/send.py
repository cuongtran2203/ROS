#! /usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
import cv2
from test_model import *
def talker():
    model=Inference()
    pub=rospy.Publisher("imager",String,queue_size=10)
    rospy.init_node("talker",anonymous=True)
    rate=rospy.Rate(2)
    img=cv2.imread("src/myrobot/scripts/meov3.jpg")
    results=model.run(img)
    while not rospy.is_shutdown():
        rospy.loginfo("publish image")
        pub.publish(results)
        rate.sleep()
if __name__ == "__main__":
    try:
        talker()
    except rospy.ROSInterruptException:
        pass