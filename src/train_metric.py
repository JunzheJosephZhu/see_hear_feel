import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import torch
from dataset import TripletDataset
from models import make_audio_encoder, make_vision_encoder, make_tactile_encoder
from engine import MetricLearn
from torch.utils.data import DataLoader
import os
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

def main(args):
    train_set = TripletDataset(args.train_csv, args.sil_ratio)
    val_set = TripletDataset(args.val_csv, args.sil_ratio)
    train_loader = DataLoader(train_set, args.batch_size, num_workers=0)
    val_loader = DataLoader(val_set, 1, num_workers=0)
    v_encoder = make_vision_encoder(args.embed_dim)
    a_encoder = make_audio_encoder(args.embed_dim)
    t_encoder = make_tactile_encoder(args.embed_dim)
    parameters = list(v_encoder.parameters()) + list(a_encoder.parameters()) + list(t_encoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.period, gamma=args.gamma)
    # save config
    config_name = os.path.basename(args.config).split(".yaml")[0]
    exp_dir = os.path.join("exp", config_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    with open(os.path.join(exp_dir, "conf.yaml"), "w") as outfile:
        yaml.safe_dump(vars(args), outfile)
    # pl stuff
    pl_module = MetricLearn(v_encoder, a_encoder, t_encoder, optimizer, train_loader, val_loader, scheduler, args)
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(exp_dir, "checkpoints"),
        filename="{epoch}-{step}",
        save_top_k=-1,
        save_last=True,
    )
    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint],
        default_root_dir=exp_dir,
        gpus=-1,
        strategy="dp",
        limit_val_batches=10,
        check_val_every_n_epoch=1,
        log_every_n_steps=5
    )
    trainer.fit(
        pl_module,
        ckpt_path=None
        if args.resume is None
        else os.path.join(os.getcwd(), args.resume),
    )
    print("best_model", checkpoint.best_model_path)

if __name__ == "__main__":
    import configargparse

    p = configargparse.ArgParser()
    p.add("-c", "--config", is_config_file=True, default="conf/metric_learn.yaml")
    p.add("--batch_size", default=8)
    p.add("--lr", default=0.001)
    p.add("--gamma", default=0.9)
    p.add("--period", default=3)
    p.add("--epochs", default=100)
    p.add("--resume", default=None)
    p.add("--num_workers", default=4, type=int)
    # TripleLoss
    p.add("--gap", required=True, type=float)
    p.add("--embed_dim", required=True, type=int)
    # data
    p.add("--train_csv", default="train.csv")
    p.add("--val_csv", default="val.csv")
    p.add("--sil_ratio", default=0.2)

    args = p.parse_args()
    main(args)