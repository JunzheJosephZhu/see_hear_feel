import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import torch
from dataset import FuturePredDataset
from models.actors import make_audio_encoder, make_vision_encoder, make_tactile_encoder, Attention_Fusion, Forward_Model
from engines.imi_engine import Future_Prediction
from torch.utils.data import DataLoader
import os
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

def main(args):
    def strip_sd(state_dict, prefix):
        return {k.lstrip(prefix): v for k, v in state_dict.items() if k.startswith(prefix)}

    train_set = FuturePredDataset(args.train_csv, args.max_len)
    val_set = FuturePredDataset(args.val_csv, -1)
    train_loader = DataLoader(train_set, 1, num_workers=8)
    val_loader = DataLoader(val_set, 1, num_workers=4)
    # encoder models
    v_encoder = make_vision_encoder(args.embed_dim)
    a_encoder = make_audio_encoder(args.embed_dim)
    t_encoder = make_tactile_encoder(args.embed_dim)
    # load pretrained model
    state_dict = torch.load(args.pretrained, map_location="cpu")["state_dict"]
    v_encoder.load_state_dict(strip_sd(state_dict, "v_model."))
    a_encoder.load_state_dict(strip_sd(state_dict, "a_model."))
    t_encoder.load_state_dict(strip_sd(state_dict, "t_model."))
    # fusion and forward model
    fusion = Attention_Fusion(args.embed_dim, args.num_heads)
    forward_model = Forward_Model(args.embed_dim, args.action_dim)

    parameters = list(fusion.parameters()) + list(forward_model.parameters())
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
    pl_module = Future_Prediction(v_encoder, a_encoder, t_encoder, fusion, forward_model, optimizer, train_loader, val_loader, scheduler, args)
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
    p.add("-c", "--config", is_config_file=True, default="conf/future_prediction.yaml")
    p.add("--lr", default=0.001)
    p.add("--gamma", default=0.9)
    p.add("--period", default=3)
    p.add("--epochs", default=100)
    p.add("--resume", default=None)
    p.add("--num_workers", default=4, type=int)
    p.add("--pretrained", required=True)
    # fusion model
    p.add("--num_heads", type=int)
    p.add("--embed_dim", required=True, type=int)
    p.add("--action_dim", default=3, type=int)
    # data
    p.add("--train_csv", default="train_0331.csv")
    p.add("--val_csv", default="val_0331.csv")
    p.add("--max_len", default=10, type=int)

    args = p.parse_args()
    main(args)