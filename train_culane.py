import os
import yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.utils import data
from torchvision.utils import make_grid
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function
from ptsemseg.loader import get_loader
from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer

import wandb


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)

def train(cfg):

    # Setup seeds
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Augmentations
    augmentations = cfg["training"].get("augmentations", None)
    data_aug = get_composed_augmentations(augmentations)

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]

    t_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg["data"]["train_split"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
        augmentations=data_aug,
    )

    v_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg["data"]["val_split"],
        img_size=(590,1640),
    )

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(
        t_loader,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["n_workers"],
        shuffle=True,
    )

    valloader = data.DataLoader(
        v_loader, batch_size=cfg["training"]["batch_size"], num_workers=cfg["training"]["n_workers"]
    )

    # Setup Metrics
    running_metrics_val = runningScore(n_classes)

    # Setup Model
    model = get_model(cfg["model"], n_classes).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print( 'Parameters:',total_params )

    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.apply(weights_init)
    pretrained_path='weights/hardnet_petite_base.pth'
    weights = torch.load(pretrained_path)
    model.module.base.load_state_dict(weights)

    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k: v for k, v in cfg["training"]["optimizer"].items() if k != "name"}

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    print("Using optimizer {}".format(optimizer))

    scheduler = get_scheduler(optimizer, cfg["training"]["lr_schedule"])

    loss_fn = get_loss_function(cfg)
    print("Using loss {}".format(loss_fn))

    start_iter = 0
    if cfg["training"]["resume"] is not None:
        if os.path.isfile(cfg["training"]["resume"]):
            print(
                "Loading model and optimizer from checkpoint '{}'".format(cfg["training"]["resume"])
            )
            checkpoint = torch.load(cfg["training"]["resume"])
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_iter = checkpoint["epoch"]
            print(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg["training"]["resume"], checkpoint["epoch"]
                )
            )
        else:
            print("No checkpoint found at '{}'".format(cfg["training"]["resume"]))

    if cfg["training"]["finetune"] is not None:
        if os.path.isfile(cfg["training"]["finetune"]):
            print(
                "Loading model and optimizer from checkpoint '{}'".format(cfg["training"]["finetune"])
            )
            checkpoint = torch.load(cfg["training"]["finetune"])
            model.load_state_dict(checkpoint["model_state"])
        else:
            print("No Resume checkpoint found at '{}'".format(cfg["training"]["resume"]))

    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    best_iou = -100.0
    i = start_iter
    flag = True
    loss_all = 0
    loss_n = 0
    while i <= cfg["training"]["train_iters"] and flag:
        for (images, labels, labels_cls) in trainloader:
            i += 1
            start_ts = time.time()
            scheduler.step()
            model.train()
            images = images.to(device)
            labels = labels.to(device)
            labels_cls = labels_cls.to(device)
            optimizer.zero_grad()
            outputs,cls = model(images)
            

            loss_pix = loss_fn(input=outputs, target=labels)
            # print(cls.shape,labels_cls.shape)
            loss_cls = F.binary_cross_entropy(cls, labels_cls.float(), size_average=True, reduction='mean')
            loss =loss_cls + loss_pix 
            loss.backward()
            optimizer.step()
            c_lr = scheduler.get_lr()

            time_meter.update(time.time() - start_ts)
            loss_all += loss.item()
            loss_n += 1
            if (i + 1) % cfg["training"]["print_interval_image"] == 0:
                wandb.log({"images/original":wandb.Image(make_grid(images.detach().cpu())),"images/gt":wandb.Image(make_grid(outputs[:,1:4,:,:].detach().cpu()))})
                # wandb.log({"cls": cls.detach().cpu().numpy()}, i + 1)
            if (i + 1) % cfg["training"]["print_interval"] == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}  lr={:.6f}"
                print_str = fmt_str.format(
                    i + 1,
                    cfg["training"]["train_iters"],
                    loss_all / loss_n,
                    time_meter.avg / cfg["training"]["batch_size"],
                    c_lr[0],
                )
                

                print(print_str)
                
                print(loss.item())
                wandb.log({"loss/train_loss_total": loss.item()}, i + 1)
                wandb.log({"loss/train_loss_pix": loss_pix.item()}, i + 1)
                wandb.log({"loss/train_loss_cls": loss_cls.item()}, i + 1)
                
                time_meter.reset()
                

            
            if (i + 1) % cfg["training"]["val_interval"] == 0 or (i + 1) == cfg["training"][
                "train_iters"
            ]:
                torch.cuda.empty_cache()
                model.eval()
                loss_all = 0
                loss_n = 0
                with torch.no_grad():
                    for i_val, (images_val, labels_val, labels_val_cls) in tqdm(enumerate(valloader)):
                        images_val = images_val.to(device)
                        labels_val = labels_val.to(device)
                        labels_val_cls = labels_val_cls.to(device)
                        outputs,cls = model(images_val)
                        val_loss= loss_fn(input=outputs, target=labels_val)
                        # val_loss_cls = F.binary_cross_entropy(cls, labels_cls.float(), size_average=True, reduction='mean')
                        # val_loss =val_loss_pix + val_loss_cls 

                        pred = outputs.data.max(1)[1].cpu().numpy()
                        gt = labels_val.data.cpu().numpy()

                        running_metrics_val.update(gt, pred)
                        val_loss_meter.update(val_loss.item())

                wandb.log({"loss/val_loss": val_loss_meter.avg}, i + 1)
                

                score, class_iou = running_metrics_val.get_scores()
                for k, v in score.items():
                    print(k, v)
                    
                    wandb.log({"val_metrics/{}".format(k): v}, i + 1)

                for k, v in class_iou.items():
                    
                    wandb.log({"val_metrics/cls_{}".format(k): v}, i + 1)

                val_loss_meter.reset()
                running_metrics_val.reset()
                
                state = {
                      "epoch": i + 1,
                      "model_state": model.state_dict(),
                      "optimizer_state": optimizer.state_dict(),
                      "scheduler_state": scheduler.state_dict(),
                }
                save_path = os.path.join(wandb.run.dir,
                    "{}_{}_checkpoint.pkl".format(cfg["model"]["arch"], cfg["data"]["dataset"]),
                )
                torch.save(state, save_path)
                wandb.save(save_path)

                if score["Mean IoU : \t"] >= best_iou:
                    best_iou = score["Mean IoU : \t"]
                    state = {
                        "epoch": i + 1,
                        "model_state": model.state_dict(),
                        "best_iou": best_iou,
                    }
                    save_path = os.path.join(
                        wandb.run.dir,
                        "{}_{}_best_model.pkl".format(cfg["model"]["arch"], cfg["data"]["dataset"]),
                    )
                    torch.save(state, save_path)
                    wandb.save(save_path)
                torch.cuda.empty_cache()

            if (i + 1) == cfg["training"]["train_iters"]:
                flag = False
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/hardnet_culane.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()
    wandb.init(project="culane",config=args)
    with open(args.config) as fp:
        cfg = yaml.load(fp)

    run_id = random.randint(1, 100000)
    

    

    train(cfg)