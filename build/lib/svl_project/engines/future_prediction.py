class Future_Prediction(LightningModule):
    def __init__(self, v_model, a_model, t_model, fusion, forward_model, optimizer, train_loader, val_loader, scheduler, config):
        super().__init__()
        self.v_model = v_model
        self.a_model = a_model
        self.t_model = t_model
        self.fusion = fusion
        self.forward_model = forward_model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.config = config
    def common_step(self, batch):
        cam_frames, log_specs, gs_frames, actions = [ex.squeeze(0) for ex in batch]
        with torch.no_grad():
            v_embed = self.v_model(cam_frames).detach()
            a_embed = self.a_model(log_specs).detach()
            t_embed = self.t_model(gs_frames).detach()
        v_out, a_out, t_out = self.fusion(v_embed[:-1], a_embed[:-1], t_embed[:-1])
        v_pred, a_pred, t_pred = self.forward_model(v_out, a_out, t_out, actions[:-1].float())
        v_target, a_target, t_target = v_embed[1:], a_embed[1:], t_embed[1:]
        return (v_pred, a_pred, t_pred), (v_target, a_target, t_target)

    def training_step(self, batch, batch_idx):
        (v_pred, a_pred, t_pred), (v_target, a_target, t_target) = self.common_step(batch)
        v_loss = F.mse_loss(v_pred, v_target)
        a_loss = F.mse_loss(a_pred, a_target)
        t_loss = F.mse_loss(t_pred, t_target)
        self.log_dict({"train/v_loss": v_loss, "train/a_loss": a_loss, "train/t_loss": t_loss})
        return v_loss + a_loss + t_loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            (v_pred, a_pred, t_pred), (v_target, a_target, t_target) = self.common_step(batch)
        v_loss = F.mse_loss(v_pred, v_target)
        a_loss = F.mse_loss(a_pred, a_target)
        t_loss = F.mse_loss(t_pred, t_target)
        self.log_dict({"val/v_loss": v_loss, "val/a_loss": a_loss, "val/t_loss": t_loss})
        # visualize
        self.logger.experiment.add_image(f"vision/{batch_idx}_pred", v_pred.T + 1, global_step=self.current_epoch, dataformats="HW")
        self.logger.experiment.add_image(f"vision/{batch_idx}_gt", v_target.T + 1, global_step=self.current_epoch, dataformats="HW")
        self.logger.experiment.add_image(f"audio/{batch_idx}_pred", a_pred.T + 1, global_step=self.current_epoch, dataformats="HW")
        self.logger.experiment.add_image(f"audio/{batch_idx}_gt", a_target.T + 1, global_step=self.current_epoch, dataformats="HW")
        self.logger.experiment.add_image(f"touch/{batch_idx}_pred", t_pred.T + 1, global_step=self.current_epoch, dataformats="HW")
        self.logger.experiment.add_image(f"touch/{batch_idx}_gt", t_target.T + 1, global_step=self.current_epoch, dataformats="HW")

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]
