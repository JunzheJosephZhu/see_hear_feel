from pytorch_lightning.core.lightning import LightningModule
import torch.nn.functional as F
import torch

class VAELearn(LightningModule):
    def __init__(self, vae, train_loader, val_loader, beta, prior_scale, optimizer, scheduler, config):
        super().__init__()
        self.vae = vae
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.beta = beta  # The relative weight on the KL Divergence/regularization loss, relative to reconstruction
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.prior_scale = prior_scale
        self.config = config
        self.save_hyperparameters(vars(config))

    def compute_loss(self, predicted_pixels, gt_pixels, encoded_context_dist):
        if self.config.allow_mismatch:
            gt_pixels = gt_pixels[:, :, :predicted_pixels.size(2), :predicted_pixels.size(3)]
        if self.config.loss_type == "l1":
            recon_loss = F.l1_loss(predicted_pixels, gt_pixels)
        elif self.config.loss_type == "l2":
            recon_loss = F.mse_loss(predicted_pixels, gt_pixels)
        else:
            raise NotImplementedError

        prior = torch.distributions.Normal(torch.zeros(encoded_context_dist.batch_shape +
                                                       encoded_context_dist.event_shape).to(self.device),
                                           self.prior_scale)
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
                self.logger.experiment.add_image(f"train/{str(batch_idx)}_original", batch[0], global_step=self.current_epoch)
                self.logger.experiment.add_image(f"train/{str(batch_idx)}_reconstruct", pixels[0], global_step=self.current_epoch)
            # TODO: add two channel audio logging
        return loss

    def validation_step(self, batch, batch_idx):
        inp = batch
        pixels, prior = self.vae(inp)
        loss, recon_loss, kld = self.compute_loss(pixels, inp.detach(), prior)
        self.log('val/loss_recon', recon_loss.item())
        self.log('val/loss_kld', torch.mean(kld).item())
        if batch_idx < 10:
            if pixels.size(1) == 3:
                self.logger.experiment.add_image(f"val/{str(batch_idx)}_original", batch[0], global_step=self.current_epoch)
                self.logger.experiment.add_image(f"val/{str(batch_idx)}_reconstruct", pixels[0], global_step=self.current_epoch)

        return loss

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]
