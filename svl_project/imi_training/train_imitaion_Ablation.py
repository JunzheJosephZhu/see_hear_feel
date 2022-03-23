import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import torch

from svl_project.datasets.imi_dataset import ImitationOverfitDataset, ImitationDatasetFramestack, ImitationDatasetSingleCam, ImitationDatasetFramestackMulti
from svl_project.models.encoders import make_vision_encoder, make_tactile_encoder, make_audio_encoder,make_tactile_flow_encoder
from svl_project.models.imi_models import Imitation_Baseline_Actor_Tuning, Imitation_Actor_Ablation
from svl_project.engines.imi_engine import ImiBaselineLearn_Tuning, ImiBaselineLearn_Ablation
from torch.utils.data import DataLoader
from itertools import cycle
import os
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from svl_project.boilerplate import *
import pandas as pd

def main(args):

    print(sys.getrecursionlimit())
    sys.setrecursionlimit(8000)
    print(sys.getrecursionlimit())
    
    train_csv = pd.read_csv(args.train_csv)
    val_csv = pd.read_csv(args.val_csv)
    if args.num_episode is None:
        train_num_episode = len(train_csv)
        val_num_episode = len(val_csv)
    else:
        train_num_episode = args.num_episode
        val_num_episode = args.num_episode
    train_set = torch.utils.data.ConcatDataset([ImitationDatasetFramestackMulti(args.train_csv, args, i, args.data_folder) for i in range(train_num_episode)])
    val_set = torch.utils.data.ConcatDataset([ImitationDatasetFramestackMulti(args.val_csv, args, i, args.data_folder, False) for i in range(val_num_episode)])
    train_loader = DataLoader(train_set, args.batch_size, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_set, args.batch_size, num_workers=1, shuffle=False)
    # train_loader = DataLoader(train_set, 1, num_workers=0, shuffle=False)
    # val_loader = DataLoader(val_set, 1, num_workers=0, shuffle=False)
    v_encoder = make_vision_encoder() # 3,4/4,5
    if args.use_flow:
        t_encoder = make_tactile_flow_encoder(args.embed_dim_t)
    else:
        t_encoder = make_tactile_encoder(args.embed_dim_t)
    a_encoder = make_audio_encoder(args.conv_bottleneck, args.embed_dim_a)
    imi_model = Imitation_Actor_Ablation(v_encoder, t_encoder, a_encoder, args).cuda()
    optimizer = torch.optim.Adam(imi_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.period, gamma=args.gamma)
    # save config
    exp_dir = save_config(args)
    # pl stuff
    pl_module = ImiBaselineLearn_Ablation(imi_model, optimizer, train_loader, val_loader, scheduler, args)
    start_training(args, exp_dir, pl_module)

if __name__ == "__main__":
    import configargparse
    p = configargparse.ArgParser()
    p.add("-c", "--config", is_config_file=True, default="conf/imi/imi_learn_ablation.yaml")
    p.add("--batch_size", default=4)
    p.add("--lr", default=1e-3, type=float)
    p.add("--gamma", default=0.9, type=float)
    p.add("--period", default=3)
    p.add("--epochs", default=30, type=int)
    p.add("--resume", default=None)
    p.add("--num_workers", default=8, type=int)
    # imi_stuff
    p.add("--conv_bottleneck", required=True, type=int)
    p.add("--embed_dim_v", required=True, type=int)
    p.add("--embed_dim_t", required=True, type=int)
    p.add("--embed_dim_a", required=True, type=int)
    p.add("--action_dim", default=3, type=int)
    p.add("--num_stack", required=True, type=int)
    p.add("--frameskip", required=True, type=int)
    p.add("--loss_type", default="cce")
    p.add("--pretrained", default=None)
    p.add("--freeze_till", required=True, type=int)
    # data
    p.add("--train_csv", default="train.csv")
    p.add("--val_csv", default="val.csv")
    p.add("--data_folder", default="data/test_recordings")
    p.add("--resized_height_v", required=True, type=int)
    p.add("--resized_width_v", required=True, type=int)
    p.add("--resized_height_t", required=True, type=int)
    p.add("--resized_width_t", required=True, type=int)
    p.add("--num_episode", default=None, type=int)
    p.add("--crop_percent", required=True, type=float)
    p.add("--num_camera", required=True, type=int)
    p.add("--total_episode", required=True, type=int)
    p.add("--ablation", required=True)
    p.add("--num_heads", required=True, type=int)
    p.add("--use_flow", default=False, type=bool)


    args = p.parse_args()
    main(args)