from pytorch_lightning import LightningModule
import torch

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
        v_inp, _, t_inp, a_inp, keyboard = batch
        action_pred = self.actor(v_inp, a_inp, t_inp, self.current_epoch < self.config.freeze_till)
        loss = compute_loss(action_pred, keyboard)
        return loss

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]

class ImmiBaselineLearn(LightningModule):
    def __init__(self, actor, optimizer, train_loader, val_loader, scheduler, config):
        super().__init__()
        self.actor = actor
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.config = config
        self.cce = torch.nn.CrossEntropyLoss()

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
        v_gripper_inp, v_fixed_inp, _, _, keyboard = batch
        action_pred = self.actor(v_gripper_inp, v_fixed_inp, self.current_epoch < self.config.freeze_till)
        loss = compute_loss(action_pred, keyboard)
        return loss

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
            # batch_size = pred.size(0)
            # space_dim = demo.size(-1)
            # # [batch, 3, num_dims]
            # pred = pred.reshape(batch_size, 3, space_dim)
            # return self.cce(pred, demo)

            demo = (demo - 1).type(torch.cuda.FloatTensor)
            # print(f"pred = {pred}, demo = {demo}")
            return self.mse(pred, demo)

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
            # batch_size = pred.size(0)
            # space_dim = demo.size(-1)
            # # [batch, 3, num_dims]
            # pred = pred.reshape(batch_size, 3, space_dim)
            # return self.cce(pred, demo)

            demo = (demo - 1).type(torch.cuda.FloatTensor)
            return self.mse(pred, demo)
            
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
