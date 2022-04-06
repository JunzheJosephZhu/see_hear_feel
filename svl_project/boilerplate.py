import os
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

def save_config(args):
    config_name = os.path.basename(args.config).split(".yaml")[0]
    exp_dir = os.path.join("exp", config_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    with open(os.path.join(exp_dir, "conf.yaml"), "w") as outfile:
        yaml.safe_dump(vars(args), outfile)
    with open(os.path.join(exp_dir, "conf.yaml"), "w") as outfile:
        yaml.safe_dump(vars(args), outfile)
    return exp_dir

def start_training(args, exp_dir, pl_module):
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(exp_dir, "checkpoints"),
        filename="{epoch}-{step}",
        save_top_k=-1,
        save_last=True,
        monitor='val/val_acc',
        mode='max'
    )
    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint],
        default_root_dir=exp_dir,
        gpus=-1,
        strategy="dp",
        limit_val_batches=100,
        check_val_every_n_epoch=5,
        log_every_n_steps=5
    )
    trainer.fit(
        pl_module,
        ckpt_path=None
        if args.resume is None
        else os.path.join(os.getcwd(), args.resume),
    )
    print("best_model", checkpoint.best_model_path)