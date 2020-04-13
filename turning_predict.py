#!/usr/bin/env python
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
import rospkg


class Img_Sub():
    def __init__(self):
        self.bridge = CvBridge()
        self.pred_sub= rospy.Subscriber("/Drive/pred", Image, self.callback)
        
        self.pred_ok = False

    def callback(self, msg):
        self.pred_raw = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.pred_ok = True





if __name__ == "__main__":
    rospy.init_node('DriveArea_turning_Predicter', anonymous=True)
    rospack = rospkg.RosPack()
    pkg_root = os.path.join(rospack.get_path('drive_area_detection'),'src','FCHarDNet')
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default=os.path.join(pkg_root,"configs/demo.yml"),
        help="Configuration file to use",
    )
    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.load(fp)

    # check if 
    img_sub = Img_Sub()
    drive_pred_pub = rospy.Publisher("Drive/pred_debug", Image)
    while 1:
        if img_sub.pred_ok and img_sub.pred_ok:
            break
        time.sleep(1)
    bridge = CvBridge()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        pred = img_sub.pred_raw.copy()
        center_point = cfg["camera"]["center_point"]
        bar_height = 50


        mask =  pred.argmax(2).copy()

        mask[mask>=2] = 0
        height,width = mask.shape
        left_mask_bar = mask[height-bar_height:,:center_point]
        right_mask_bar = mask[height-bar_height:,center_point:]
        
        l_percent = left_mask_bar.sum()/float(left_mask_bar.shape[0]*left_mask_bar.shape[1])
        r_percent = right_mask_bar.sum()/float(right_mask_bar.shape[0]*right_mask_bar.shape[1])
        cv2.circle(pred,(width-50, 50), 15, (0, 255, 0), -1)
        left_turn,righ_turn = False,False
        if l_percent > cfg["camera"]["left_threshold"]:
            left_turn = True
            cv2.circle(pred,(50, 50), 15, (0, 255, 0), -1)
        if r_percent > cfg["camera"]["right_threshold"]:
            right_turn = True
            cv2.circle(pred,(width-50, 50), 15, (0, 255, 0), -1)
        

        drive_pred_pub.publish(bridge.cv2_to_imgmsg(pred))
        print('Turn:{} {}'.format(l_percent,r_percent))
        rate.sleep()

    
    

