#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def webcam_publisher():
    # Initialize the ROS node
    rospy.init_node('webcam_publisher', anonymous=True)

    # Create a publisher for the /nao_robot/camera/bottom/camera/image_raw topic
    image_pub = rospy.Publisher('/nao_robot/camera/bottom/camera/image_raw', Image, queue_size=10)

    # Create a CvBridge to convert OpenCV images to ROS Image messages
    bridge = CvBridge()

    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Set the frame width and height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    rate = rospy.Rate(10)  # 10Hz

    while not rospy.is_shutdown():
        # Capture frame from webcam
        ret, frame = cap.read()

        if ret:
            # Convert the OpenCV image to a ROS Image message
            image_msg = bridge.cv2_to_imgmsg(frame, "bgr8")

            # Publish the image message
            image_pub.publish(image_msg)

        rate.sleep()

    # Release the webcam when the script is stopped
    cap.release()

if __name__ == '__main__':
    try:
        webcam_publisher()
    except rospy.ROSInterruptException:
        pass
