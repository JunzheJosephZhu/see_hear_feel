import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import torch

from svl_project.datasets.imi_dataset import ImitationOverfitDataset, ImitationDatasetFramestack
from svl_project.models.encoders import make_vision_encoder
from svl_project.models.imi_models import Imitation_Baseline_Actor_Tuning
from svl_project.engines.imi_engine import ImiBaselineLearn_Tuning
from torch.utils.data import DataLoader
from itertools import cycle
import os
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from svl_project.boilerplate import *

def main(args):

    print(sys.getrecursionlimit())
    sys.setrecursionlimit(8000)
    print(sys.getrecursionlimit())

    # train_set = torch.utils.data.ConcatDataset([ImitationOverfitDataset(args.train_csv, i, args.data_folder) for i in range(args.num_episode)])
    # val_set = torch.utils.data.ConcatDataset([ImitationOverfitDataset(args.val_csv, i, args.data_folder) for i in range(args.num_episode)])

    train_set = torch.utils.data.ConcatDataset([ImitationDatasetFramestack(args.train_csv, args, i, args.data_folder) for i in range(args.num_episode)])
    val_set = torch.utils.data.ConcatDataset([ImitationDatasetFramestack(args.val_csv, args, i, args.data_folder) for i in range(70 - args.num_episode)])

    # train_set = ImitationOverfitDataset(args.train_csv, args.data_folder)
    # val_set = ImitationOverfitDataset(args.val_csv, args.data_folder)
    # train_set = ImitationDatasetFramestack(args.train_csv, args, args.data_folder)
    # val_set = ImitationDatasetFramestack(args.val_csv, args, args.data_folder)
    train_loader= DataLoader(train_set, args.batch_size, num_workers=12, shuffle=True)
    val_loader= DataLoader(val_set, args.batch_size, num_workers=12, shuffle=False)
    v_encoder = make_vision_encoder(args.conv_bottleneck, args.embed_dim, (3, 4)) # 3,4
    imi_model = Imitation_Baseline_Actor_Tuning(v_encoder, args)
    optimizer = torch.optim.Adam(imi_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.period, gamma=args.gamma)
    # save config
    exp_dir = save_config(args)
    # pl stuff
    pl_module = ImiBaselineLearn_Tuning(imi_model, optimizer, train_loader, val_loader, scheduler, args)
    start_training(args, exp_dir, pl_module)

if __name__ == "__main__":
    import configargparse

    p = configargparse.ArgParser()
    p.add("-c", "--config", is_config_file=True, default="conf/imi/imi_learn.yaml")
    p.add("--batch_size", default=8)
    p.add("--lr", default=1e-4, type=float)
    p.add("--gamma", default=0.9)
    p.add("--period", default=3)
    p.add("--epochs", default=150)
    p.add("--resume", default=None)
    p.add("--num_workers", default=8, type=int)
    # imi_stuff
    p.add("--conv_bottleneck", required=True, type=int)
    p.add("--embed_dim", required=True, type=int)
    p.add("--action_dim", default=3, type=int)
    p.add("--num_stack", required=True, type=int)
    p.add("--frameskip", required=True, type=int)
    p.add("--loss_type", default="mse")
    p.add("--pretrained", default=None)
    p.add("--freeze_till", required=True, type=int)
    # data
    p.add("--train_csv", default="train.csv")
    p.add("--val_csv", default="val.csv")
    p.add("--data_folder", default="data/test_recordings")
    p.add("--resized_height", required=True, type=int)
    p.add("--resized_width", required=True, type=int)
    p.add("--num_episode", required=True, type=int)
    p.add("--crop_percent", required=True, type=float)


    args = p.parse_args()
    main(args)