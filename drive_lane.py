#!/usr/bin/env python
import os
import yaml
from collections import OrderedDict
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


def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

class Img_Sub():
    def __init__(self):
        self.bridge = CvBridge()
        self.image_raw_sub= rospy.Subscriber("/image_raw", Image, self.callback)
        self.lane_mask_sub= rospy.Subscriber("/Lane/mask", Image, self.lane_callback)
        # self.image_raw_sub= rospy.Subscriber("/Drive/main_point", Float32MultiArray, self.mainpoint_callback)
        # self.drive_scatter_pub  = rospy.Publisher("Drive/scatter", Image)
        self.image_ok = False
        self.lane_ok = False
    def callback(self, msg):
        self.image_raw = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.image_ok = True
    def lane_callback(self,msg):
        self.lane_mask = self.bridge.imgmsg_to_cv2(msg, "mono8")
        self.lane_ok = True






def demo(cfg,pkg_root,img_sub,drive_pub):
     # Setup device

    print(torch.cuda.device_count())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Setup Model
    n_classes = cfg["testing"]["n_classes"]
    img_size = (cfg["data"]["img_rows"], cfg["data"]["img_cols"])

    model = get_model(cfg["model"], n_classes)
    # model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    # pretrained_path=os.path.join(pkg_root,'weights/hardnet_petite_base.pth')
    # weights = torch.load(pretrained_path)
    # model.module.base.load_state_dict(weights)
    total_params = sum(p.numel() for p in model.parameters())
    print( 'Parameters:',total_params )
    print(pkg_root)
    if cfg["testing"]["resume"] is not None:
        resume_dir = os.path.join(pkg_root,cfg["testing"]["resume"])
        if os.path.isfile(resume_dir):
            print(
                "Loading model and optimizer from checkpoint '{}'".format(resume_dir)
            )
            checkpoint = torch.load(resume_dir)
            new_state_dict = OrderedDict()
            for k, v in checkpoint["model_state"].items():
                name = k[7:]
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            model = model.to(device)
            print(
                "Loaded checkpoint '{}' (iter {})".format(
                    resume_dir, checkpoint["epoch"]
                )
            )
        else:
            print("No checkpoint found at '{}'".format(resume_dir))
             
    # image = cv2.imread('pic/10.jpg')
    bridge = CvBridge()
    rate = rospy.Rate(cfg["testing"]["publish_rate"])
    erosion_kernel = np.ones((cfg["testing"]["erode_kernel"],cfg["testing"]["erode_kernel"]),np.uint8)
    with torch.no_grad():
        while not rospy.is_shutdown():
            
            image = img_sub.image_raw

            ToTensor = torchvision.transforms.ToTensor()
            image = ToTensor(image).unsqueeze(0).to(device)
            out,cls = model(image) 
            print(cls[0])
            for j in range(4):
                if cls[0][j] > 0.65:
                    # print(out[0,1+j,:,:])
                    out[0,1+j,:,:] = out[0,1+j,:,:]*1.15
                    # print(out[0,1+j,:,:])
            # out[0,1,:,:]*cls[0][0]
            # out[0,2:,:,:]*cls[0][1]
            # out[0,3:,:,:]*cls[0][2]
            # out[0,4:,:,:]*cls[0][3]
            

            # pred = out[0].max(0)[1].cpu().numpy()
            # main_area = ((pred==1) * 1).astype(np.uint8)
            
            
            # erosion = cv2.erode(main_area,erosion_kernel,iterations = 1)


            # lane = img_sub.lane_mask


            # main_area = subtract_lane(erosion,lane)
            
            # delta,delta_start = cfg["testing"]["sample_delta"],cfg["testing"]["sample_delta_start"]
            # mid_point , main_area_fine = find_mid(main_area,delta,delta_start)


            out = torch.nn.functional.softmax(out[0],dim=0)
            out_max = out.argmax(0)
            out_max = out_max.detach().cpu().numpy()
            out = out.detach().cpu().numpy()
            
            
            
            
            start_time = time.time()
            
            print("--- %s seconds ---" % (time.time() - start_time))
            out_main = np.clip(np.transpose(out[1:4,:,:]*255.0, (1,2,0)),0,255).astype(np.uint8)
            out_lane = np.clip(np.transpose(out[3:,:,:]*255.0, (1,2,0)),0,255).astype(np.uint8)


            out_max_color = np.zeros((cfg['data']['img_rows'],cfg['data']['img_cols'],3),dtype=np.uint8)
            
            
            
            for key in cfg["vis"]:
                out_max_color[out_max==cfg["vis"][key]["id"]] = np.array(cfg["vis"][key]["color"])
            # #publish mid points
            # mid_point_size = mid_point.shape
            # mid_point = mid_point.flatten().tolist()
            # msg_midpoint = Float32MultiArray()
            # msg_midpoint.data = mid_point
            # line_points_pub.publish(msg_midpoint)

            #publish rgb pred map
            drive_pub.publish(bridge.cv2_to_imgmsg(out_max_color,'bgr8'))

            # drive_pred_lane_pub.publish(bridge.cv2_to_imgmsg(out_lane,'bgr8'))
            #publish main_area_maks
            # drive_pub.publish(bridge.cv2_to_imgmsg(main_area_fine.astype(np.uint8),'mono8'))

            # drive_pred_max.publish(bridge.cv2_to_imgmsg(out_max_color.astype(np.uint8),'bgr8'))

            rate.sleep()
    return

def subtract_lane(mask,lane):
    lane[lane>1] = 1
    mask = mask + lane
    mask[mask>1] = 1
    mask = ((mask - lane)*255.0).astype(np.uint8)
    return mask
def find_mid(main_area_course,delta=20,delta_y=20):
    main_area_course[main_area_course>1] = 1
    # Find Connective component
    start_time = time.time()
    output  = cv2.connectedComponentsWithStats(main_area_course)
    
    area_stat = output[2][:,4]
    label_index = np.argsort(area_stat)[-2]
    main_area = (output[1]==label_index).astype(np.uint8)*254
    lapsobelx = cv2.Sobel(main_area,cv2.CV_64F,1,0,ksize=9)


    mid_point = []
    
    while True:
        sample_y = lapsobelx.shape[0]-delta_y
        sample_line = lapsobelx[sample_y]

        left_point = np.where(sample_line>0)[0]
        right_point = np.where(sample_line<0)[0]
        if len(left_point)==0 or len(right_point)==0:
            break
        sample_x = (left_point[0]+ right_point[-1]) // 2
        delta_y +=delta
        mid_point.append([sample_x,sample_y])
    mid_point = np.array(mid_point)
    return mid_point , main_area
    





if __name__ == "__main__":
    rospy.init_node('DriveArea', anonymous=True)
    rospack = rospkg.RosPack()
    pkg_root = os.path.join(rospack.get_path('drive_area_detection'),'src','FCHarDNet')
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default=os.path.join(pkg_root,"configs/demo_culane.yml"),
        help="Configuration file to use",
    )
    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.load(fp)

    

    img_sub = Img_Sub()
    drive_pub = rospy.Publisher("Drive/lane", Image)
    
    
    for key in cfg["vis"]:
        print(key)
    while 1:
        if img_sub.image_ok:
            break
    
    demo(cfg,pkg_root,img_sub,drive_pub)

    
