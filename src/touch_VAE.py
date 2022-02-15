import json
import os
import pandas as pd
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data.dataset import Dataset
import torchaudio
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from models import make_tactile_encoder
from dataset import ImmitationDataSet_hdf5

class VAEDataset(Dataset):
    def __init__(self, log_file, max_len, data_folder="data"):
        """
        neg_ratio: ratio of silence audio clips to sample
        """
        super().__init__()
        self.logs = pd.read_csv(log_file)
        self.data_folder = data_folder
        sr = 16000
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=int(sr * 0.025), hop_length=int(sr * 0.01), n_mels=64
        )
        self.max_len = max_len

    def __getitem__(self, idx):
        trial = os.path.join(self.data_folder, self.logs.iloc[idx].Time)
        # print("cam_frames shape: {}".format(cam_frames.shape))

        # read gelsight frames
        gs_video = cv2.VideoCapture(os.path.join(trial, "gs.avi"))
        success, gs_frame = gs_video.read()
        gs_frames = []
        while success:
            gs_frames.append(gs_frame)
            success, gs_frame = gs_video.read()
        gs_frames = torch.as_tensor(np.stack(gs_frames, 0)).permute(0, 3, 1, 2) / 255

        # print("gs_frames shape: {}".format(gs_frames.shape))
        # see how many frames there are
        num_frames = gs_frames.size(0)

        # clip number of frames used for training
        if num_frames > self.max_len and self.max_len > 0:
            start = torch.randint(high=num_frames - self.max_len + 1, size=())
            end = start + self.max_len
        else:
            start = 0
            end = num_frames

        gs_frames = gs_frames[start:end]
        return gs_frames

    def __len__(self):
        return len(self.logs)

def independent_multivariate_normal(mean, stddev):
    # Create a normal distribution, which by default will assume all dimensions but one are a batch dimension
    dist = torch.distributions.Normal(mean, stddev, validate_args=True)
    # Wrap the distribution in an Independent wrapper, which reclassifies all but one dimension as part of the actual
    # sample shape, but keeps variances defined only on the diagonal elements of what would be the MultivariateNormal
    multivariate_mimicking_dist = torch.distributions.Independent(dist, len(mean.shape) - 1)
    return multivariate_mimicking_dist

class VAE(nn.Module):
    def __init__(self, encoder, decoder, out_dim, latent_dim):
        self.encoder = encoder
        self.decoder = decoder
        self.mean_layer = nn.Linear(out_dim, latent_dim)
        self.scale_layer = nn.Linear(out_dim, latent_dim)
        self.decoder_initial = nn.Linear(latent_dim, out_dim)

    def forward(self, inp):
        shared_repr = self.encoder(inp)
        mean = self.mean_layer(shared_repr)
        scale = torch.exp(self.scale_layer(shared_repr))
        z_dist = independent_multivariate_normal(mean=mean,
                                               stddev=scale)
        z = self.get_vector(z_dist)
        decoder_inp = self.decoder_initial(z)
        pixels = self.decoder(decoder_inp)
        return pixels, z_dist

class VAELearn(LightningModule):
    def __init__(self, vae, train_loader, val_loader, beta, prior_scale, optimizer, scheduler):
        super().__init__()
        self.vae = vae
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.beta = beta  # The relative weight on the KL Divergence/regularization loss, relative to reconstruction
        self.prior_scale = prior_scale  # The scale parameter used to construct prior used in KLD
        self.optimizer = optimizer
        self.scheduler = scheduler

    def compute_loss(self, predicted_pixels, gt_pixels, encoded_context_dist):
        recon_loss = F.mse_loss(predicted_pixels, gt_pixels)

        prior = torch.distributions.Normal(torch.zeros(encoded_context_dist.batch_shape +
                                                       encoded_context_dist.event_shape).to(self.device),
                                           1)
        independent_prior = torch.distributions.Independent(prior,
                                                            len(encoded_context_dist.event_shape))
        kld = torch.distributions.kl.kl_divergence(encoded_context_dist, independent_prior)


        loss = recon_loss + self.beta * torch.mean(kld)
        return loss, recon_loss, kld

    def training_step(self, batch, batch_idx):
        inp = batch
        pixels, prior = self.vae(inp)
        loss, recon_loss, kld = self.compute_loss(pixels, inp.detach(), prior)
        self.log('train/loss_recon', recon_loss.item())
        self.log('train/loss_kld', torch.mean(kld).item())
        if batch_idx < 10:
            if pixels.size(1) == 3:
                self.logger.experiment.add_image(f"train/original_{str(batch_idx)}", pixels, global_step=self.current_epoch)
            # TODO: add two channel audio logging
        return loss

    def validation_step(self, batch, batch_idx):
        inp = batch
        pixels, prior = self.vae(inp)
        loss, recon_loss, kld = self.compute_loss(pixels, inp.detach(), prior)
        self.log('val/loss_recon', recon_loss.item())
        self.log('val/loss_kld', torch.mean(kld).item())
        self.log('train/loss_recon', recon_loss.item())
        self.log('train/loss_kld', torch.mean(kld).item())
        if batch_idx < 10:
            self.logger.experiment.add_image(f"train/original_{str(batch_idx)}", pixels, global_step=self.current_epoch)

        return loss

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader


    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]

def main(args):
    def strip_sd(state_dict, prefix):
        """
        strip prefix from state dictionary
        """
        return {k.lstrip(prefix): v for k, v in state_dict.items() if k.startswith(prefix)}

    # get pretrained model
    train_set = ImmitationDataSet_hdf5(args.train_csv, num_stack=1)
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