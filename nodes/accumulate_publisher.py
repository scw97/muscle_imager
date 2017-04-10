#!/usr/bin/env python

from __future__ import division
import rospy
import rosparam
import copy
import cProfile
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

sizeImage = 128+1024*1024 # Size of header + data.


class ImageAccumulator:
    def __init__(self):
        rospy.init_node('accumulate_publisher')
        self.nodename = rospy.get_name().rstrip('/')
        self.img1 = None
        self.cvbridge = CvBridge()
        self.pubImage       = rospy.Publisher(self.nodename+'/image_output', Image,  queue_size=2)
        self.subImage       = rospy.Subscriber('/kine_camera_1/image_raw', Image,  self.image_callback,   queue_size=2, buff_size=2*sizeImage, tcp_nodelay=True)

    def image_callback(self, rosimg):
        if self.img1 is None:
            outimg = self.cvbridge.imgmsg_to_cv2(rosimg, 'passthrough')
            self.img1 = outimg.astype(np.float)
            rosimgOutput = self.cvbridge.cv2_to_imgmsg(outimg, 'passthrough')
            rosimgOutput.encoding = 'bgr8'
            self.pubImage.publish(rosimgOutput)
        else:
            inimg = self.cvbridge.imgmsg_to_cv2(rosimg, 'passthrough').astype(float)
            cv2.accumulateWeighted(inimg.astype(np.float), self.img1, 0.25)
            outimg = self.img1.astype(np.uint8)
            rosimgOutput = self.cvbridge.cv2_to_imgmsg(outimg, 'passthrough')
            #rosimgOutput.encoding = 'bgr8'
            self.pubImage.publish(rosimgOutput)

if __name__ == '__main__':
    accumulator = ImageAccumulator()
    rospy.spin()
