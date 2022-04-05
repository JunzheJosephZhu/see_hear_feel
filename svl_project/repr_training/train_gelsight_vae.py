import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import torch
from svl_project.datasets.repr_dataset import GelsightFrameDataset
from svl_project.models.decoders import make_tactile_decoder
from svl_project.models.encoders import make_tactile_encoder
from svl_project.models.repr_models import VAE
from svl_project.engines.vae import VAELearn
from torch.utils.data import DataLoader
import os
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from svl_project.boilerplate import *

def main(args):
    train_set = GelsightFrameDataset(args.train_csv, args.data_folder)
    val_set = GelsightFrameDataset(args.val_csv, args.data_folder)
    train_loader = DataLoader(train_set, args.batch_size, num_workers=4)
    val_loader = DataLoader(val_set, 1, num_workers=1, shuffle=True)
    t_encoder = make_tactile_encoder(512)
    t_decoder = make_tactile_decoder(args.latent_dim)
    vae_model = VAE(t_encoder, t_decoder, 512, args.latent_dim)
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
    p.add("-c", "--config", is_config_file=True, default="conf/repr/gelsight_vae.yaml")
    p.add("--batch_size", default=4)
    p.add("--lr", default=0.001, type=float)
    p.add("--gamma", default=0.9, type=float)
    p.add("--period", default=3, type=int)
    p.add("--epochs", default=200, type=int)
    p.add("--resume", default=None)
    p.add("--num_workers", default=4, type=int)
    # VAE stuff
    p.add("--latent_dim", required=True, type=int)
    p.add("--beta", required=True, type=float)
    p.add("--prior_scale", default=1.0, type=float)
    p.add("--allow_mismatch", default=False, type=bool)
    p.add("--loss_type", required=True, type=str)
    # data
    p.add("--train_csv", default="train.csv")
    p.add("--val_csv", default="val.csv")
    p.add("--data_folder", default="../data_0331/test_recordings")

    args = p.parse_args()
    main(args)