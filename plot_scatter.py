import os
import yaml
import time
import shutil
import torch
import random
import argparse
from cStringIO import StringIO

import numpy as np
import torch.nn as nn
import torchvision

from torch.utils import data
from torchvision.utils import make_grid
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.utils import get_logger

import cv2

import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import PIL
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray,MultiArrayDimension
import rospy


class Img_Sub():
    def __init__(self):
        self.bridge = CvBridge()
        self.image_raw_sub= rospy.Subscriber("/image_raw", Image, self.callback)
        # self.image_raw_sub= rospy.Subscriber("/Drive/main_point", Float32MultiArray, self.mainpoint_callback)
        # self.drive_scatter_pub  = rospy.Publisher("Drive/scatter", Image)
        self.image_ok = False
    def callback(self, msg):
        self.image_raw = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.image_ok = True



class MidPoint_Sub():
    def __init__(self):

        self.mid_point_sub= rospy.Subscriber("/Drive/main_point", Float32MultiArray, self.callback)
        self.mid_point =[]
        self.mid_point_ok = False
    def callback(self, msg):
        self.mid_point = np.array(msg.data).reshape((-1,2))
        self.mid_point_ok = True

        




def main(img_sub,mid_point_sub,drive_scatter_pub):
    rate = rospy.Rate(5)
    bridge = CvBridge()
    while not rospy.is_shutdown():
        print('a')
        img_draw = img_sub.image_raw.copy()
        mid_point = mid_point_sub.mid_point

        if len(mid_point):
            plt.imshow(img_draw)
            plt.scatter(mid_point[:,0],mid_point[:,1])
            buffer_ = StringIO()#using buffer,great way!
            plt.savefig(buffer_,format = 'jpeg')
            buffer_.seek(0)
            dataPIL = PIL.Image.open(buffer_)
            data = np.asarray(dataPIL)
            drive_scatter_pub.publish(bridge.cv2_to_imgmsg(data,"rgb8"))
            buffer_.close()
            plt.close()
        rate.sleep()
    return

if __name__ == "__main__":
    rospy.init_node('DriveScatter', anonymous=True)
    img_sub = Img_Sub()
    mid_point_sub = MidPoint_Sub()
    drive_scatter_pub  = rospy.Publisher("Drive/scatter", Image)
    
    while 1:
        if img_sub.image_ok and mid_point_sub.mid_point_ok:
            break

    main(img_sub,mid_point_sub,drive_scatter_pub)
