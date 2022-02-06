from pytorch_lightning import LightningModule
import torch
import torch.nn.functional as F

class MetricLearn(LightningModule):
    def __init__(self, v_model, a_model, t_model, optimizer, train_loader, val_loader, scheduler, config):
        super().__init__()
        self.v_model = v_model
        self.a_model = a_model
        self.t_model = t_model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.config = config

    def tripletloss(self, v_pos, a_pos, t_pos, v_neg):
        """
        Args:
            v_pos: Anchor Camera Image Embedding #[batch_size, embed_dim]
            t_pos: Positive Gelsight Image Embedding #[batch_size, embed_dim]
            a_pos: Positive Audio LogMelSpec Embedding #[batch_size, embed_dim]
            v_neg: Negative Camera Image Embedding #[batch_size, embed_dim]
        """
        def l2d(a, b):
            return (a - b).pow(2).sum(1)

        neg_dist = l2d(v_pos, v_neg)
        gs_dist = l2d(v_pos, t_pos)
        audio_dist = l2d(v_pos, a_pos)
        dist_dict = {"neg_dist": neg_dist.mean(), "gs_dist": gs_dist.mean(), "audio_dist": audio_dist.mean()}
        gs_loss = torch.clamp(gs_dist - neg_dist + self.config.gap, min=0).mean()
        audio_loss = torch.clamp(audio_dist - neg_dist + self.config.gap, min=0).mean()
        return gs_loss, audio_loss, dist_dict

    def common_step(self, batch):
        cam_pos, gs_pos, log_spec, cam_neg = batch
        batch_size = cam_pos.size(0)
        cam_collated = torch.cat([cam_pos, cam_neg], dim=0)
        v_pos, v_neg = self.v_model(cam_collated).split(batch_size, 0)
        a_pos = self.a_model(log_spec)
        t_pos = self.t_model(gs_pos)
        gs_loss, audio_loss, dist_dict = self.tripletloss(v_pos=v_pos, a_pos=a_pos, t_pos=t_pos, v_neg=v_neg)
        return gs_loss, audio_loss, dist_dict

    def training_step(self, batch, batch_idx):
        gs_loss, audio_loss, dist_dict = self.common_step(batch)
        self.log_dict({"train/gs_loss": gs_loss, "train/audio_loss": audio_loss})
        self.log_dict({f"train/{k}": v for k, v in dist_dict.items()})
        return gs_loss + audio_loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            gs_loss, audio_loss, dist_dict = self.common_step(batch)
        self.log_dict({"val/gs_loss": gs_loss, "val/audio_loss": audio_loss})
        self.log_dict({f"val/{k}": v for k, v in dist_dict.items()})

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]


class ImmiLearn(LightningModule):
    def __init__(self, actor, optimizer, train_loader, val_loader, scheduler, config):
        super().__init__()
        self.actor = actor
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.config = config
        self.cce = torch.nn.CrossEntropyLoss()
        print("immi learn")

    def training_step(self, batch, batch_idx):
        def compute_loss(pred, demo):
            """
            pred: # [batch, 3 * action_dims]
            demo: # [batch, action_dims]
            """
            batch_size = pred.size(0)
            space_dim = demo.size(-1)
            # [batch, 3, num_dims]
            pred = pred.reshape(batch_size, 3, space_dim)
            return self.cce(pred, demo)
        v_inp, _, t_inp, a_inp, keyboard, _ = batch
        action_pred = self.actor(v_inp, a_inp, t_inp, self.current_epoch < self.config.freeze_till)
        loss = compute_loss(action_pred, keyboard)
        self.log_dict({"train/action_loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        def compute_loss(pred, demo):
            """
            pred: # [batch, 3 * action_dims]
            demo: # [batch, action_dims]
            """
            batch_size = pred.size(0)
            space_dim = demo.size(-1)
            # [batch, 3, num_dims]
            pred = pred.reshape(batch_size, 3, space_dim)
            return self.cce(pred, demo)
        v_inp, _, t_inp, a_inp, keyboard, _ = batch
        action_pred = self.actor(v_inp, a_inp, t_inp, self.current_epoch < self.config.freeze_till)
        with torch.no_grad():
            loss = compute_loss(action_pred, keyboard)
        self.log_dict({"val/action_loss": loss})

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]

class ImmiLearn_Reg(LightningModule):
    def __init__(self, actor, optimizer, train_loader, val_loader, scheduler, config):
        super().__init__()
        self.actor = actor
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.config = config
        self.mse = torch.nn.MSELoss()
        print("immi learn")

    def training_step(self, batch, batch_idx):
        def compute_loss(pred, demo):
            """
            pred: # [batch, 3 * action_dims]
            demo: # [batch, action_dims]
            """
            batch_size = pred.size(0)
            space_dim = demo.size(-1)
            # [batch, 3, num_dims]
            pred = pred.reshape(batch_size, space_dim)
            return self.mse(pred, demo)
        v_inp, _, t_inp, a_inp, keyboard, _ = batch
        keyboard = (keyboard - 1).type(torch.cuda.FloatTensor)
        action_pred = self.actor(v_inp, a_inp, t_inp, self.current_epoch < self.config.freeze_till)
        loss = compute_loss(action_pred, keyboard)
        self.log_dict({"train/action_loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        def compute_loss(pred, demo):
            """
            pred: # [batch, 3 * action_dims]
            demo: # [batch, action_dims]
            """
            batch_size = pred.size(0)
            space_dim = demo.size(-1)
            # [batch, 3, num_dims]
            pred = pred.reshape(batch_size, space_dim)
            return self.mse(pred, demo)
        v_inp, _, t_inp, a_inp, keyboard, _ = batch
        keyboard = (keyboard - 1).type(torch.cuda.FloatTensor)
        action_pred = self.actor(v_inp, a_inp, t_inp, self.current_epoch < self.config.freeze_till)
        with torch.no_grad():
            loss = compute_loss(action_pred, keyboard)
        self.log_dict({"val/action_loss": loss})

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]


class ImmiBaselineLearn(LightningModule):
    def __init__(self, actor, optimizer, train_loader, val_loader, scheduler, config): #v_center_encoder, v_fix_encoder,
        super().__init__()
        # self.v_center_encoder = v_center_encoder
        # self.v_fix_encoder = v_fix_encoder
        self.actor = actor
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.config = config
        # self.cce = torch.nn.CrossEntropyLoss()
        self.mse = torch.nn.MSELoss()
        print("baseline learn")

    def training_step(self, batch, batch_idx):
        def compute_loss(pred, demo):
            """
            pred: # [batch, 3 * action_dims]
            demo: # [batch, action_dims]
            """
            batch_size = pred.size(0)
            space_dim = demo.size(-1)
            # [batch, 3, num_dims]
            pred = pred.reshape(batch_size, space_dim)
            return self.mse(pred, demo)
        v_gripper_inp, v_fixed_inp, _, _, keyboard, _ = batch
        print("gripper_video", v_gripper_inp)
        keyboard = (keyboard - 1).type(torch.cuda.FloatTensor)
        action_pred = self.actor(v_gripper_inp, v_fixed_inp, self.current_epoch < self.config.freeze_till)
        print("action", action_pred)
        print("keyboard", keyboard)
        loss = compute_loss(action_pred, keyboard)
        self.log_dict({"train/action_loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        def compute_loss(pred, demo):
            """
            pred: # [batch, 3 * action_dims]
            demo: # [batch, action_dims]
            """
            batch_size = pred.size(0)
            space_dim = demo.size(-1)
            # [batch, 3, num_dims]
            pred = pred.reshape(batch_size, space_dim)
            return self.mse(pred, demo)
        v_gripper_inp, v_fixed_inp, _, _, keyboard, _ = batch
        keyboard = (keyboard - 1).type(torch.cuda.FloatTensor)
        action_pred = self.actor(v_gripper_inp, v_fixed_inp, self.current_epoch < self.config.freeze_till)
        with torch.no_grad():
            loss = compute_loss(action_pred, keyboard)
        self.log_dict({"val/action_loss": loss})

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]

class ImmiBaselineLearn_Tuning(LightningModule):
    def __init__(self, actor, optimizer, train_loader, val_loader, scheduler, config): #v_center_encoder, v_fix_encoder,
        super().__init__()
        # self.v_center_encoder = v_center_encoder
        # self.v_fix_encoder = v_fix_encoder
        self.actor = actor
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.config = config
        # self.cce = torch.nn.CrossEntropyLoss()
        self.mse = torch.nn.MSELoss()
        print("baseline learn")

    def training_step(self, batch, batch_idx):
        def compute_loss(pred, demo):
            """
            pred: # [batch, 3 * action_dims]
            demo: # [batch, action_dims]
            """
            batch_size = pred.size(0)
            space_dim = demo.size(-1)
            # [batch, 3, num_dims]
            pred = pred.reshape(batch_size, space_dim)
            return self.mse(pred, demo)
        v_gripper_inp, v_fixed_inp, keyboard = batch
        # print("gripper_video", v_gripper_inp)
        # print("fixed_video", v_fixed_inp)
        v_input = torch.cat([v_gripper_inp,  v_fixed_inp], dim=1)
        # print("v_cat", v_input)
        keyboard = (keyboard - 1).type(torch.cuda.FloatTensor)
        action_pred = self.actor(v_input, self.current_epoch < self.config.freeze_till)
        # print("action", action_pred)
        # print("keyboard", keyboard)
        loss = compute_loss(action_pred, keyboard)
        self.log_dict({"train/action_loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        def compute_loss(pred, demo):
            """
            pred: # [batch, 3 * action_dims]
            demo: # [batch, action_dims]
            """
            batch_size = pred.size(0)
            space_dim = demo.size(-1)
            # [batch, 3, num_dims]
            pred = pred.reshape(batch_size, space_dim)
            return self.mse(pred, demo)
        v_gripper_inp, v_fixed_inp, keyboard = batch
        v_input = torch.cat([v_gripper_inp, v_fixed_inp], dim=1)
        keyboard = (keyboard - 1).type(torch.cuda.FloatTensor)
        action_pred = self.actor(v_input, self.current_epoch < self.config.freeze_till)
        with torch.no_grad():
            loss = compute_loss(action_pred, keyboard)
        self.log_dict({"val/action_loss": loss})

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]


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


class ImmiPoseBaselineLearn(LightningModule):
    def __init__(self, actor, optimizer, train_loader, val_loader, scheduler, config):
        super().__init__()
        self.actor = actor
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.config = config
        self.cce = torch.nn.CrossEntropyLoss()
        self.mse = torch.nn.MSELoss()

    def training_step(self, batch, batch_idx):
        def compute_loss(pred, demo):
            """
            pred: # [batch, 3 * action_dims]
            demo: # [batch, action_dims]
            """
            batch_size = pred.size(0)
            space_dim = demo.size(-1)
            # [batch, 3, num_dims]
            pred = pred.reshape(batch_size, 3, space_dim)
            return self.cce(pred, demo)

            # demo = (demo - 1).type(torch.cuda.FloatTensor)
            # # print(f"pred = {pred}, demo = {demo}")
            # return self.mse(pred, demo)
        _, _, _, _, keyboard, pose = batch
        # print('\nbatch {} pose:\n{}'.format(batch_idx, pose))
        action_pred = self.actor(pose, False)
        loss = compute_loss(action_pred, keyboard)
        self.log_dict({"train/loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        def compute_loss(pred, demo):
            """
            pred: # [batch, 3 * action_dims]
            demo: # [batch, action_dims]
            """
            batch_size = pred.size(0)
            space_dim = demo.size(-1)
            # [batch, 3, num_dims]
            pred = pred.reshape(batch_size, 3, space_dim)
            return self.cce(pred, demo)

            # demo = (demo - 1).type(torch.cuda.FloatTensor)
            # return self.mse(pred, demo)

        _, _, _, _, keyboard, pose = batch
        action_pred = self.actor(pose, True)
        with torch.no_grad():
            loss = compute_loss(action_pred, keyboard)
        self.log_dict({"val/loss": loss})
        return loss

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]