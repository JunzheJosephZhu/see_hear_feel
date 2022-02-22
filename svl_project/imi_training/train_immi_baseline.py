import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import torch
from dataset import ImmitationDataSet
from models.actors import make_vision_encoder, Immitation_Baseline_Actor
from engines.imi_engine import ImmiBaselineLearn
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
    train_loader = DataLoader(train_set, args.batch_size, num_workers=0)
    val_loader = DataLoader(val_set, 1, num_workers=0)
    v_gripper_encoder = make_vision_encoder(args.embed_dim)
    v_fixed_encoder = make_vision_encoder(args.embed_dim)

    # state_dict = torch.load(args.pretrained, map_location="cpu")["state_dict"]
    # v_gripper_encoder.load_state_dict(strip_sd(state_dict, "v_model."))

    actor = Immitation_Baseline_Actor(v_gripper_encoder, v_fixed_encoder, args.embed_dim, args.action_dim)
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.period, gamma=args.gamma)
    # save config
    config_name = os.path.basename(args.config).split(".yaml")[0]
    exp_dir = os.path.join("exp_baseline", config_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    with open(os.path.join(exp_dir, "conf.yaml"), "w") as outfile:
        yaml.safe_dump(vars(args), outfile)
    # pl stuff
    pl_module = ImmiBaselineLearn(actor, optimizer, train_loader, val_loader, scheduler, args)
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
    p.add("-c", "--config", is_config_file=True, default="conf/immi_learn_pose_baseline.yaml")
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
    p.add("--freeze_till", required=True, type=int)
    p.add("--action_dim", default=3, type=int)
    p.add("--num_heads", type=int)
    # data
    p.add("--train_csv", default="train.csv")
    p.add("--val_csv", default="val.csv")

    args = p.parse_args()
    main(args)