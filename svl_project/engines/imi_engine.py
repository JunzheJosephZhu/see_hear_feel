from pytorch_lightning import LightningModule
import torch
import torch.nn.functional as F

class ImiBaselineLearn_Tuning(LightningModule):
    def __init__(self, actor, optimizer, train_loader, val_loader, scheduler, config):
        super().__init__()
        self.actor = actor
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.config = config
        self.loss_type = self.config.loss_type
        self.loss_cal = None
        if self.loss_type == 'mse':
            self.loss_cal = torch.nn.MSELoss()
        elif self.loss_type == 'cce':
            self.loss_cal = torch.nn.CrossEntropyLoss()
        print("baseline learn")
    
    def compute_loss(self, pred, demo):
        """
        pred: # [batch, 3 * action_dims]
        demo: # [batch, action_dims]
        """
        batch_size = pred.size(0)
        space_dim = demo.size(-1)
        if self.loss_type == 'mse':
            # [batch, 3, num_dims]
            pred = pred.reshape(batch_size, space_dim)
        elif self.loss_type == 'cce':
            # [batch, 3, num_dims]
            pred = pred.reshape(batch_size, 3, space_dim)
        return self.loss_cal(pred, demo)

    def training_step(self, batch, batch_idx):
        # use idx in batch for debugging
        v_gripper_inp, v_fixed_inp, keyboard = batch #, idx = batch
        v_input = torch.cat((v_gripper_inp, v_fixed_inp), dim = 0)
        if self.loss_type == 'mse':
            keyboard = (keyboard - 1.).type(torch.cuda.FloatTensor)
        action_pred = self.actor(v_input, self.current_epoch < self.config.freeze_till) #, idx)
        loss = self.compute_loss(action_pred, keyboard)
        self.log_dict({"train/action_loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        v_gripper_inp, v_fixed_inp, keyboard = batch #, idx = batch
        v_input = torch.cat((v_gripper_inp, v_fixed_inp), dim = 0)
        if self.loss_type == 'mse':
            keyboard = (keyboard - 1.).type(torch.cuda.FloatTensor)
        action_pred = self.actor(v_input, self.current_epoch < self.config.freeze_till) #, idx)
        with torch.no_grad():
            loss = self.compute_loss(action_pred, keyboard)
        self.log_dict({"val/action_loss": loss})

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]

class ImiPoseBaselineLearn(LightningModule):
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