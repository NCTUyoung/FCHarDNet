import os
import yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np
import torch.nn as nn


from torch.utils import data
from torchvision.utils import make_grid
from tqdm import tqdm

from ptsemseg.models import get_model


def demo(cfg):
     # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Setup Model
    n_classes = cfg["model"]["n_classes"]
    img_size = (cfg["data"]["img_rows"], cfg["data"]["img_cols"])
    model = get_model(cfg["model"], n_classes).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print( 'Parameters:',total_params )

    if cfg["testing"]["resume"] is not None:
        if os.path.isfile(cfg["testing"]["resume"]):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg["testing"]["resume"])
            )
            checkpoint = torch.load(cfg["testing"]["resume"])
            model.load_state_dict(checkpoint["model_state"])
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg["testing"]["resume"], checkpoint["epoch"]
                )
            )
        else:
            logger.info("No checkpoint found at '{}'".format(cfg["testing"]["resume"]))
            raise 
if __name__ == "__main__":
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
    demo(cfg)
