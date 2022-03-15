import torch
from svl_project.datasets.repr_dataset import VisionFixedDataset
from svl_project.models.decoders import make_vision_decoder
from svl_project.models.encoders import make_vision_encoder
from svl_project.models.repr_models import VAE
from svl_project.engines.vae import VAELearn
from torch.utils.data import DataLoader
import os
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from svl_project.boilerplate import *
import sys

def main(args):
    print(sys.getrecursionlimit())
    sys.setrecursionlimit(8000)
    print(sys.getrecursionlimit())
    train_set = VisionFixedDataset(args.train_csv, args.data_folder)
    val_set = VisionFixedDataset(args.val_csv, args.data_folder)
    train_loader = DataLoader(train_set, args.batch_size, num_workers=8)
    val_loader = DataLoader(val_set, 1, num_workers=8)
    v_encoder = make_vision_encoder()
    v_decoder = make_vision_decoder(args.latent_dim)
    vae_model = VAE(v_encoder, v_decoder, 512, args.latent_dim)
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.period, gamma=args.gamma)
    # save config
    exp_dir = save_config(args)
    # pl stuff
    pl_module = VAELearn(vae_model, train_loader, val_loader, args.beta, args.prior_scale, optimizer, scheduler, args)
    start_training(args, exp_dir, pl_module)

if __name__ == "__main__":
    import configargparse

    p = configargparse.ArgParser()
    p.add("-c", "--config", is_config_file=True, default="conf/repr/vision_vae_fixed.yaml")
    p.add("--batch_size", default=4)
    p.add("--lr", default=0.001, type=float)
    p.add("--gamma", default=0.9)
    p.add("--period", default=3)
    p.add("--epochs", default=5000)
    p.add("--resume", default=None)
    p.add("--num_workers", default=4, type=int)
    # VAE stuff
    # p.add("--conv_bottleneck", required=True, type=int)
    # p.add("--embed_dim", required=True, type=int)
    p.add("--latent_dim", required=True, type=int)
    p.add("--beta", required=True, type=float)
    p.add("--prior_scale", default=1.0, type=float)
    p.add("--allow_mismatch", default=False, type=bool)
    # data
    p.add("--train_csv", default="train.csv")
    p.add("--val_csv", default="val.csv")
    p.add("--data_folder", default="data/test_recordings_0214")
    p.add("--loss_type", required=True)

    args = p.parse_args()
    main(args)