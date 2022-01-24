import torch
from dataset import ImmitationDataSet
from models import make_audio_encoder, make_vision_encoder, make_tactile_encoder
from engine import MetricLearn
from torch.utils.data import DataLoader
import os
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

def main(args):
    def strip_sd(state_dict, prefix):
        """
        strip prefix from state dictionary
        """
        return {k.lstrip(prefix): v for k, v in state_dict.items() if k.startswith(prefix)}

    # get pretrained model    
    train_set = ImmitationDataSet(args.train_csv)
    val_set = ImmitationDataSet(args.val_csv)
    train_loader = DataLoader(train_set, args.batch_size, num_workers=1)
    val_loader = DataLoader(val_set, 1, num_workers=1)
    v_encoder = make_vision_encoder(args.embed_dim)
    a_encoder = make_audio_encoder(args.embed_dim)
    t_encoder = make_tactile_encoder(args.embed_dim)
    state_dict = torch.load(args.pretrained)["state_dict"]
    v_encoder.load_state_dict(strip_sd(state_dict, "v_model."))
    a_encoder.load_state_dict(strip_sd(state_dict, "a_model."))
    t_encoder.load_state_dict(strip_sd(state_dict, "t_model."))
    # freeze stuff
    def freeze_net(network):
        for p in network.parameters():
            p.requires_grad = False
    if args.freeze:
        freeze_net(v_encoder)
        freeze_net(a_encoder)
        freeze_net(t_encoder)
    # verifying
    for cam_frame, gs_frame, log_spec, action in train_loader:
        cam_frame, gs_frame, log_spec, action
        v_embed = v_encoder(cam_frame)
        a_embed = a_encoder(log_spec)
        t_embed = t_encoder(gs_frame)
        def l2d(a, b):
            return (a - b).pow(2).sum(1)
        print(l2d(v_embed, a_embed))
        print(l2d(v_embed, t_embed))
        



if __name__ == "__main__":
    import configargparse

    p = configargparse.ArgParser()
    p.add("-c", "--config", is_config_file=True, default="conf/immi_learn.yaml")
    p.add("--batch_size", default=8)
    p.add("--lr", default=0.001)
    p.add("--gamma", default=0.9)
    p.add("--period", default=3)
    p.add("--epochs", default=100)
    p.add("--resume", default=None)
    p.add("--num_workers", default=4, type=int)
    # model
    p.add("--embed_dim", required=True, type=int)
    p.add("--pretrained", required=True)
    p.add("--freeze_till", default=True)
    # data
    p.add("--train_csv", default="train.csv")
    p.add("--val_csv", default="val.csv")

    args = p.parse_args()
    main(args)