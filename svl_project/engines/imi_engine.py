import time
import cv2
from pytorch_lightning import LightningModule
from tomlkit import key
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class ImiEngine(LightningModule):
    def __init__(self, actor, optimizer, train_loader, val_loader, scheduler, config):
        super().__init__()
        self.actor = actor
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.config = config

        self.loss_cce = torch.nn.CrossEntropyLoss()

        self.wrong = 1
        self.correct = 0
        self.total = 0
        print("baseline learn")

    def compute_loss(self, demo, pred_logits, xyz_gt, xyz_pred):
        immi_loss = self.loss_cce(pred_logits, demo)
        aux_loss = F.mse_loss(xyz_gt, xyz_pred)
        return immi_loss + aux_loss * self.config.aux_multiplier, immi_loss, aux_loss

    def training_step(self, batch, batch_idx):
        # use idx in batch for debugging
        inputs, demo, xyzrpy_gt, optical_flow = batch
        action_logits, xyzrpy_pred, weights = self.actor(inputs)  # , idx)
        loss, immi_loss, aux_loss = self.compute_loss(demo, action_logits, xyzrpy_gt, xyzrpy_pred)
        self.log_dict({"train/immi_loss": immi_loss, "train/aux_loss": aux_loss})
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, demo, xyzrpy_gt, optical_flow = batch  # , idx = batch
        action_logits, xyzrpy_pred, weights = self.actor(inputs)  # , idx)
        loss, immi_loss, aux_loss = self.compute_loss(demo, action_pred, xyzrpy_gt, xyzrpy_pred)
        self.log_dict({"val/immi_loss": immi_loss, "val/aux_loss": aux_loss})
        action_pred = torch.argmax(action_logits, dim=1)
        if weights != None and  batch_idx < 225:
            weights = weights[0]
            df_cm = pd.DataFrame(weights.cpu().numpy(), index = range(weights.shape[0]), columns=range(weights.shape[0]))
            plt.figure(figsize = (10,7))
            fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
            plt.close(fig_)
            self.logger.experiment.add_figure("Confusion matrix", fig_, batch_idx)
        # number of corrects and total number
        return ((action_pred == demo).sum(), action_pred.numel())

    def validation_epoch_end(self, validation_step_outputs):
        numerator = 0
        divider = 0
        for (cor, total) in validation_step_outputs:
            numerator += cor
            divider += total
        acc = numerator / divider
        self.log("val/acc", acc, on_step=False, on_epoch=True)

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]

