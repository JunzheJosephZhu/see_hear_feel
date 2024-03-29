import torch
from src.datasets.imi_dataset import TransformerEpisode
from src.models.encoders import (
    make_vision_encoder,
    make_tactile_encoder,
    make_audio_encoder,
    make_tactile_flow_encoder,
)
from src.models.multimodal_timesformer import MultiModal_TimeSformer
from src.engines.engine import ImiEngine
from torch.utils.data import DataLoader
from src.train_utils import *
import pandas as pd
import numpy as np

torch.multiprocessing.set_sharing_strategy("file_system")


def strip_sd(state_dict, prefix):
    """
    strip prefix from state dictionary
    """
    return {k.lstrip(prefix): v for k, v in state_dict.items() if k.startswith(prefix)}


def main(args):
    train_csv = pd.read_csv(args.train_csv)
    val_csv = pd.read_csv(args.val_csv)
    if args.num_episode is None:
        train_num_episode = len(train_csv)
        val_num_episode = len(val_csv)
    else:
        train_num_episode = args.num_episode
        val_num_episode = args.num_episode

    train_set = torch.utils.data.ConcatDataset(
        [
            TransformerEpisode(args.train_csv, args, i, args.data_folder)
            for i in range(train_num_episode)
        ]
    )
    val_set = torch.utils.data.ConcatDataset(
        [
            TransformerEpisode(args.val_csv, args, i, args.data_folder, False)
            for i in range(val_num_episode)
        ]
    )

    # create weighted sampler to balance samples
    train_label = []
    for episode in train_set.datasets:
        for idx in range(len(episode)):
            train_label.append(episode.get_demo(idx))
    # histogram of label
    class_sample_count = np.zeros(pow(3, args.action_dim))
    for t in np.unique(train_label):
        class_sample_count[t] = len(np.where(train_label == t)[0])
    weight = 1.0 / (class_sample_count + 1e-5)
    samples_weight = np.array([weight[t] for t in train_label])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = torch.utils.data.WeightedRandomSampler(
        samples_weight.type("torch.DoubleTensor"), len(samples_weight)
    )

    train_loader = DataLoader(
        train_set, args.batch_size, num_workers=8, sampler=sampler
    )
    val_loader = DataLoader(val_set, 1, num_workers=8, shuffle=False)

    _crop_height_v = int(args.resized_height_v * (1.0 - args.crop_percent))
    _crop_width_v = int(args.resized_width_v * (1.0 - args.crop_percent))
    _crop_height_t = int(args.resized_height_t * (1.0 - args.crop_percent))
    _crop_width_t = int(args.resized_width_t * (1.0 - args.crop_percent))

    imi_model = MultiModal_TimeSformer(
        image_size=(_crop_height_v, _crop_width_v),
        tactile_size=(_crop_height_t, _crop_width_t),
        patch_size=args.patch_size,
        num_stack=args.num_stack,
        frameskip=args.frameskip,
        fps=10,
        last_layer_stride=args.last_layer_stride,
        num_classes=3**args.action_dim,
        dim=args.dim,
        depth=args.depth,
        qkv_bias=args.qkv_bias,
        heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        ablation=args.ablation,
        channels=3,
        audio_channels=1,
        use_1dconv=args.use_1dconv,
        learn_time_embedding=args.learn_time_embedding,
        drop_path_rate=args.drop_path,
    ).cuda()
    optimizer = torch.optim.Adam(imi_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.period, gamma=args.gamma
    )
    # save config
    exp_dir = save_config(args)
    # pl stuff
    pl_module = ImiEngine(
        imi_model, optimizer, train_loader, val_loader, scheduler, args
    )
    start_training(args, exp_dir, pl_module)


if __name__ == "__main__":
    import configargparse

    p = configargparse.ArgParser()
    import time

    p.add("-c", "--config", is_config_file=True, default="conf/imi/transformer.yaml")
    p.add("--batch_size", default=16, type=int)
    p.add("--lr", default=5e-4, type=float)
    p.add("--gamma", default=0.9, type=float)
    p.add("--period", default=3, type=int)
    p.add("--epochs", default=1000, type=int)
    p.add("--resume", default=None)
    p.add("--num_workers", default=8, type=int)
    # imi_stuff
    p.add("--exp_name", required=True, type=str)
    p.add("--action_dim", default=3, type=int)
    p.add("--num_stack", required=True, type=int)
    p.add("--frameskip", required=True, type=int)
    p.add("--use_mha", default=False, action="store_true")
    # data
    p.add("--train_csv", default="data/train.csv")
    p.add("--val_csv", default="data/val.csv")
    p.add("--data_folder", default="data/test_recordings")
    p.add("--resized_height_v", required=True, type=int)
    p.add("--resized_width_v", required=True, type=int)
    p.add("--resized_height_t", required=True, type=int)
    p.add("--resized_width_t", required=True, type=int)
    p.add("--patch_size", default=16, type=int)
    p.add("--dim", default=768, type=int)
    p.add("--depth", default=12, type=int)
    p.add("--heads", default=12, type=int)
    p.add("--mlp_ratio", default=4, type=int)
    p.add("--qkv_bias", action="store_false", default=True)
    p.add("--last_layer_stride", default=1, type=int)
    p.add("--learn_time_embedding", default=False, action="store_true")
    p.add("--drop_path", default=0.1, type=float)
    p.add("--no_jitter", default=False, action="store_true")

    p.add("--num_episode", default=None, type=int)
    p.add("--crop_percent", required=True, type=float)
    p.add("--ablation", required=True)
    p.add("--use_flow", default=False, action="store_true")
    p.add("--use_holebase", default=False, action="store_true")
    p.add("--task", type=str)
    p.add("--norm_audio", default=False, action="store_true")
    p.add("--aux_multiplier", type=float)
    p.add("--minus_first", default=False, action="store_true")
    p.add("--nocrop", default=False, action="store_true")
    p.add("--use_1dconv", default=False, action="store_true")

    args = p.parse_args()
    args.batch_size *= torch.cuda.device_count()
    main(args)
