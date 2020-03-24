import os
import yaml
import time
import shutil
import torch
import random
import argparse
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
import matplotlib.pyplot as plt
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import rospy

class Img_Sub():
    def __init__(self):
        self.bridge = CvBridge()
        self.image_raw_sub= rospy.Subscriber("/image_raw", Image, self.callback)
        self.image_ok = False
    def callback(self, msg):
        
        self.image_raw = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.image_ok = True


def demo(cfg,img_sub,drive_pub,drive_pred_pub):
     # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Setup Model
    n_classes = cfg["testing"]["n_classes"]
    img_size = (cfg["data"]["img_rows"], cfg["data"]["img_cols"])
    model = get_model(cfg["model"], n_classes).to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    pretrained_path='weights/hardnet_petite_base.pth'
    weights = torch.load(pretrained_path)
    model.module.base.load_state_dict(weights)
    total_params = sum(p.numel() for p in model.parameters())
    print( 'Parameters:',total_params )

    if cfg["testing"]["resume"] is not None:
        if os.path.isfile(cfg["testing"]["resume"]):
            print(
                "Loading model and optimizer from checkpoint '{}'".format(cfg["testing"]["resume"])
            )
            checkpoint = torch.load(cfg["testing"]["resume"])
            model.load_state_dict(checkpoint["model_state"])
            print(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg["testing"]["resume"], checkpoint["epoch"]
                )
            )
        else:
            print("No checkpoint found at '{}'".format(cfg["testing"]["resume"]))
             
    # image = cv2.imread('pic/10.jpg')
    bridge = CvBridge()
    rate = rospy.Rate(5)
    with torch.no_grad():
        while not rospy.is_shutdown():
            image = img_sub.image_raw

            ToTensor = torchvision.transforms.ToTensor()
            image = ToTensor(image).unsqueeze(0).to(device)
            out = model(image) 

            pred = out[0].max(0)[1].cpu().numpy()
            main_area = ((pred==1) * 255).astype(np.uint8)
            
            # mask = pred
            # out = np.clip((out[0,1,:,:].detach().cpu().numpy() *255 ),0,255).astype(np.uint8)
            # ret2,prob =  cv2.threshold(out,80,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            out = torchvision.utils.make_grid(out[:,:3,:,:]).detach().cpu().numpy() * 50.0
            out = np.clip(np.transpose(out, (1,2,0)),0,255).astype(np.uint8)
            drive_pred_pub.publish(bridge.cv2_to_imgmsg(out,'bgr8'))
            drive_pub.publish(bridge.cv2_to_imgmsg(main_area.astype(np.uint8),'mono8'))
            rate.sleep()

    

if __name__ == "__main__":
    rospy.init_node('DriveArea', anonymous=True)
    img_sub = Img_Sub()
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/demo.yml",
        help="Configuration file to use",
    )
    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.load(fp)
    img_sub = Img_Sub()
    drive_pub = rospy.Publisher("Drive/mask", Image)
    drive_pred_pub = rospy.Publisher("Drive/pred", Image)

    while 1:
        if img_sub.image_ok:
            break
    demo(cfg,img_sub,drive_pub,drive_pred_pub)
