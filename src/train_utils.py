import os
from datetime import datetime
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime
import numpy as np


def save_config(args):
    config_name = os.path.basename(args.config).split(".yaml")[0]
    now = datetime.now()
    dt = now.strftime("%m%d%Y")
    exp_dir = os.path.join("exp" + dt + args.task, config_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    with open(os.path.join(exp_dir, "conf.yaml"), "w") as outfile:
        yaml.safe_dump(vars(args), outfile)
    return exp_dir


def start_training(args, exp_dir, pl_module, monitor="val/acc"):
    jobid = np.random.randint(0, 1000)
    jobid = os.environ.get("SLURM_JOB_ID", 0)
    exp_time = datetime.now().strftime("%m-%d-%H:%M:%S") + "-jobid=" + str(jobid)
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(exp_dir, "checkpoints", args.exp_name),
        filename=exp_time + "-{epoch}-{step}",
        save_top_k=4,
        save_last=True,
        monitor=monitor,
        mode="max",
    )

    logger = TensorBoardLogger(
        save_dir=exp_dir, version=exp_time + args.exp_name, name="lightning_logs"
    )
    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint],
        default_root_dir=exp_dir,
        gpus=-1,
        strategy="dp",
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        logger=logger,
    )
    trainer.fit(
        pl_module,
        ckpt_path=None
        if args.resume is None
        else os.path.join(os.getcwd(), args.resume),
    )
    print("best_model", checkpoint.best_model_path)
