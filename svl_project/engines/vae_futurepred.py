from pytorch_lightning.core.lightning import LightningModule
import torch.nn.functional as F
import torch
from svl_project.datasets.repr_dataset import visualize_flow
from svl_project.engines.vae import VAELearn

class VAELearn_FuturePred(VAELearn):
    def __init__(self, vae, train_loader, val_loader, beta, w_futureloss, prior_scale, optimizer, scheduler, config):
        super().__init__(vae, train_loader, val_loader, beta, prior_scale, optimizer, scheduler, config)
        self.vae = vae
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.beta = beta  # The relative weight on the KL Divergence/regularization loss, relative to reconstruction
        self.w_futureloss = w_futureloss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.prior_scale = prior_scale
        self.config = config
        self.save_hyperparameters(vars(config))

    def training_step(self, batch, batch_idx):
        current_img, future_img, tac, future_tac, action = batch
        recon_v, prior_v, recon_t, prior_t, _, pred_future_prior = self.vae(current_img, tac, action)
        with torch.no_grad():
            _, _, _, _, gt_future_prior_v_t, _ = self.vae(future_img, future_tac, torch.zeros_like(action))
        vae_loss_v, recon_loss_v, kld_v = self.compute_loss(recon_v, current_img.detach(), prior_v)
        vae_loss_t, recon_loss_t, kld_t = self.compute_loss(recon_t, tac.detach(), prior_t)
        futurepred_loss = torch.distributions.kl.kl_divergence(pred_future_prior, gt_future_prior_v_t)
        loss = vae_loss_v + vae_loss_t + torch.mean(futurepred_loss) * self.w_futureloss
        self.log('train/loss_recon_v', recon_loss_v.item())
        self.log('train/loss_kld_v', torch.mean(kld_v).item())
        self.log('train/loss_recon_t', recon_loss_t.item())
        self.log('train/loss_kld_t', torch.mean(kld_t).item())
        self.log("train/futurepred_loss", torch.mean(futurepred_loss).item())
        if batch_idx < 10:
            if recon_v.size(1) == 3:
                self.logger.experiment.add_image(f"val/{str(batch_idx)}_original_v", current_img[0],
                                                 global_step=self.current_epoch)
                self.logger.experiment.add_image(f"val/{str(batch_idx)}_reconstruct_v", recon_v[0],
                                                 global_step=self.current_epoch)
                self.logger.experiment.add_image(f"val/{str(batch_idx)}_original_t", tac[0],
                                                 global_step=self.current_epoch)
                self.logger.experiment.add_image(f"val/{str(batch_idx)}_reconstruct_t", recon_t[0],
                                                 global_step=self.current_epoch)
            # TODO: add two channel audio logging
            # elif recon.size(1) == 2 and recon.size(2) == 64:
            #     batch = torch.cat([batch, torch.zeros_like(batch)[:, :1]], dim=1)
            #     recon = torch.cat([recon, torch.zeros_like(recon)[:, :1]], dim=1)
            #     self.logger.experiment.add_image(f"train/{str(batch_idx)}_original", batch[0], global_step=self.current_epoch, dataformats="CHW")
            #     self.logger.experiment.add_image(f"train/{str(batch_idx)}_reconstruct", recon[0], global_step=self.current_epoch, dataformats="CHW")
            # else: # gelsight flow
            #     gt_flow = visualize_flow(batch[0])
            #     pred_flow = visualize_flow(recon[0])
            #     self.logger.experiment.add_image(f"train/{str(batch_idx)}_original", gt_flow, global_step=self.current_epoch, dataformats="HW")
            #     self.logger.experiment.add_image(f"train/{str(batch_idx)}_reconstruct", pred_flow, global_step=self.current_epoch, dataformats="HW")

        return loss

    def validation_step(self, batch, batch_idx):
        current_img, future_img, tac, future_tac, action = batch
        with torch.no_grad():
            pixels_v, prior_v, pixels_t, prior_t, _, pred_future_prior = self.vae(current_img, tac, action)
            _, _, _, _, gt_future_prior_v_t, _ = self.vae(future_img, future_tac, torch.zeros_like(action))
        vae_loss_v, recon_loss_v, kld_v = self.compute_loss(pixels_v, current_img.detach(), prior_v)
        vae_loss_t, recon_loss_t, kld_t = self.compute_loss(pixels_t, tac.detach(), prior_t)
        futurepred_loss = torch.distributions.kl.kl_divergence(pred_future_prior, gt_future_prior_v_t)
        loss = vae_loss_v + vae_loss_t + torch.mean(futurepred_loss) * self.w_futureloss
        self.log('val/loss_recon_v', recon_loss_v.item())
        self.log('val/loss_kld_v', torch.mean(kld_v).item())
        self.log('val/loss_recon_t', recon_loss_t.item())
        self.log('val/loss_kld_t', torch.mean(kld_t).item())
        self.log("val/futurepred_loss", torch.mean(futurepred_loss).item())
        if batch_idx < 10:
            if pixels_v.size(1) == 3:
                self.logger.experiment.add_image(f"val/{str(batch_idx)}_original_v", current_img[0], global_step=self.current_epoch)
                self.logger.experiment.add_image(f"val/{str(batch_idx)}_reconstruct_v", pixels_v[0], global_step=self.current_epoch)
                self.logger.experiment.add_image(f"val/{str(batch_idx)}_original_t", tac[0],
                                                 global_step=self.current_epoch)
                self.logger.experiment.add_image(f"val/{str(batch_idx)}_reconstruct_t", pixels_t[0],
                                                 global_step=self.current_epoch)
            # TODO: add two channel audio logging
            # elif pixels.size(1) == 2 and pixels.size(2) == 64:
            #     batch = torch.cat([batch, torch.zeros_like(batch)[:, :1]], dim=1)
            #     pixels = torch.cat([pixels, torch.zeros_like(pixels)[:, :1]], dim=1)
            #     self.logger.experiment.add_image(f"val/{str(batch_idx)}_original", batch[0], global_step=self.current_epoch, dataformats="CHW")
            #     self.logger.experiment.add_image(f"val/{str(batch_idx)}_reconstruct", pixels[0], global_step=self.current_epoch, dataformats="CHW")
            # else: # gelsight flow
            #     gt_flow = visualize_flow(batch[0])
            #     pred_flow = visualize_flow(pixels[0])
            #     self.logger.experiment.add_image(f"train/{str(batch_idx)}_original", gt_flow, global_step=self.current_epoch, dataformats="HW")
            #     self.logger.experiment.add_image(f"train/{str(batch_idx)}_reconstruct", pred_flow, global_step=self.current_epoch, dataformats="HW")

        return loss
