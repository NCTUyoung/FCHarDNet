#!/usr/bin/env python
import os
import yaml
from collections import OrderedDict
import time
# import shutil
import torch
# import random
import argparse
from cStringIO import StringIO

import numpy as np

import torch.nn as nn
import torchvision

from torch.utils import data
# from torchvision.utils import make_grid
# from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.utils import get_logger

import cv2


# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt
# import PIL
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray,MultiArrayDimension
from rospy.numpy_msg import numpy_msg
from drive_area_detection.msg import CnnOutput
import rospy
import rospkg


def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

class Img_Sub():
    def __init__(self,cfg):
        self.bridge = CvBridge()
        self.image_raw_sub= rospy.Subscriber(cfg['image_src'], Image, self.callback)
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

    with torch.no_grad():
        while not rospy.is_shutdown():
            
            image = img_sub.image_raw

            ToTensor = torchvision.transforms.ToTensor()
            image = ToTensor(image).unsqueeze(0).to(device)
            out = model(image) 
            

            



            out = torch.nn.functional.softmax(out[0],dim=0)




            ### Clustering Input 
            out_max = out.argmax(0)
            out_max = out_max.detach().cpu().numpy()
            
                 

            start_time = time.time()
            
            print("--- %s seconds ---" % (time.time() - start_time))


            out_max_color = np.zeros((cfg['data']['img_rows'],cfg['data']['img_cols'],3),dtype=np.uint8)
            for index,color in enumerate(cfg["vis"]['colors']):
                out_max_color[out_max==index] = np.array(color)





            drive_pub.publish(bridge.cv2_to_imgmsg(out_max_color.astype(np.uint8),'bgr8'))


            rate.sleep()
    return


    





if __name__ == "__main__":
    rospy.init_node('drive_Seg', anonymous=True)
    rospack = rospkg.RosPack()
    pkg_root = os.path.join(rospack.get_path('drive_area_detection'),'src','FCHarDNet')

    # Load basic config from config yaml file --------
    with open(os.path.join(pkg_root,"configs/demo_segmentation.yml")) as fp:
        cfg = yaml.load(fp)
    # Load ROS param  --------
    poblished_rate = rospy.get_param("~det_rate",10)
    image_src = rospy.get_param('~image_src','image_raw')
    cfg['image_src'] = image_src
    cfg["testing"]["publish_rate"] = poblished_rate

    img_sub = Img_Sub(cfg)

    # Publish node init  --------
    drive_pub = rospy.Publisher("Drive/segmentation", Image,queue_size=10)

    while 1:
        if img_sub.image_ok:
            print("drive_seg image_src is not ready")
            time.sleep(0.5)
            break
    print("drive_seg is ok!!")
    demo(cfg,pkg_root,img_sub,drive_pub)

    
