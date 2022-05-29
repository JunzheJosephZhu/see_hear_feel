import sys
from telnetlib import KERMIT
from tomlkit import key
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import torch

# from svl_project.datasets.imi_dataset import ImitationDatasetFramestackMulti, ImitationDatasetLabelCount
from svl_project.datasets.imi_dataset import ImitationDataset, ImitationDatasetLabelCount
from svl_project.models.encoders import make_vision_encoder, make_tactile_encoder, make_audio_encoder,make_tactile_flow_encoder
from svl_project.models.imi_models import Imitation_Actor_Ablation
from svl_project.engines.imi_engine import ImiEngine
from torch.utils.data import DataLoader
from itertools import cycle
import os
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from svl_project.boilerplate import *
import pandas as pd
import numpy as np
import time

def strip_sd(state_dict, prefix):
    """
    strip prefix from state dictionary
    """
    return {k.lstrip(prefix): v for k, v in state_dict.items() if k.startswith(prefix)}


def main(args):
    torch.multiprocessing.set_sharing_strategy("file_system")
    
    train_csv = pd.read_csv(args.train_csv)
    val_csv = pd.read_csv(args.val_csv)
    if args.num_episode is None:
        train_num_episode = len(train_csv)
        val_num_episode = len(val_csv)
    else:
        train_num_episode = args.num_episode
        val_num_episode = args.num_episode
        
    train_label_set = torch.utils.data.ConcatDataset([ImitationDatasetLabelCount(args.train_csv, args, i, args.data_folder) for i in range(train_num_episode)])
    train_set = torch.utils.data.ConcatDataset([ImitationDataset(args.train_csv, args, i, args.data_folder) for i in range(train_num_episode)])
    val_set = torch.utils.data.ConcatDataset([ImitationDataset(args.val_csv, args, i, args.data_folder, False) for i in range(val_num_episode)])

    # create weighted sampler to balance samples
    train_label = []
    for keyboard in train_label_set:
        train_label.append(keyboard)
    class_sample_count = np.zeros(pow(3, args.action_dim))
    for t in np.unique(train_label):
        class_sample_count[t] = len(np.where(train_label == t)[0])
    weight = 1. / (class_sample_count + 1e-5)
    samples_weight = np.array([weight[t] for t in train_label])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

    train_loader = DataLoader(train_set, args.batch_size, num_workers=8, sampler=sampler, persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(val_set, 1, num_workers=2, shuffle=False, persistent_workers=True, pin_memory=True)
    
    # v encoder
    v_encoder = make_vision_encoder(args.encoder_dim)
    # t encoder
    if args.use_flow:
        t_encoder = make_tactile_flow_encoder(args.encoder_dim)
    else:
        t_encoder = make_tactile_encoder(args.encoder_dim)
    # a encoder
    a_encoder = make_audio_encoder(args.encoder_dim * (2 * args.num_stack - 1), args.norm_audio)
    imi_model = Imitation_Actor_Ablation(v_encoder, t_encoder, a_encoder, args).cuda()
    optimizer = torch.optim.Adam(imi_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.period, gamma=args.gamma)
    # save config
    exp_dir = save_config(args)
    # pl stuff
    pl_module = ImiEngine(imi_model, optimizer, train_loader, val_loader, scheduler, args)
    start_training(args, exp_dir, pl_module)

if __name__ == "__main__":
    import configargparse
    p = configargparse.ArgParser()
    p.add("-c", "--config", is_config_file=True, default="conf/imi/imi_learn.yaml")
    p.add("--batch_size", default=16)
    p.add("--lr", default=1e-4, type=float)
    p.add("--gamma", default=0.9, type=float)
    p.add("--period", default=3)
    p.add("--epochs", default=60, type=int)
    p.add("--resume", default=None)
    p.add("--num_workers", default=8, type=int)
    # imi_stuff
    p.add("--conv_bottleneck", required=True, type=int)
    p.add("--exp_name", required=True, type=str)
    p.add("--encoder_dim", required=True, type=int)
    p.add("--action_dim", default=3, type=int)
    p.add("--num_stack", required=True, type=int)
    p.add("--frameskip", required=True, type=int)
    p.add("--use_mha", default=False, action="store_true")
    # data
    p.add("--train_csv", default="train.csv")
    p.add("--val_csv", default="val.csv")
    p.add("--data_folder", default="data/data_0528_flat/test_recordings")
    p.add("--resized_height_v", required=True, type=int)
    p.add("--resized_width_v", required=True, type=int)
    p.add("--resized_height_t", required=True, type=int)
    p.add("--resized_width_t", required=True, type=int)
    p.add("--num_episode", default=None, type=int)
    p.add("--crop_percent", required=True, type=float)
    p.add("--ablation", required=True)
    p.add("--num_heads", required=True, type=int)
    p.add("--use_flow", default=False, action="store_true")
    p.add("--use_holebase", default=False, action="store_true")
    p.add("--task", type=str)
    p.add("--norm_audio", default=False, action="store_true")
    p.add("--aux_multiplier", type=float)
    p.add("--minus_first", default=False, action="store_true")



    


    args = p.parse_args()
    main(args)

