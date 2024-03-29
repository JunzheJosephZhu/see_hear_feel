import time
from pytorch_lightning import LightningModule
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import pandas as pd


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
        self.save_hyperparameters(config)
        print("baseline learn")

    def compute_loss(self, demo, pred_logits, xyz_gt, xyz_pred):
        immi_loss = self.loss_cce(pred_logits, demo)
        aux_loss = F.mse_loss(xyz_gt, xyz_pred)
        return immi_loss + aux_loss * self.config.aux_multiplier, immi_loss, aux_loss

    def training_step(self, batch, batch_idx):
        # use idx in batch for debugging
        inputs, demo, xyzrpy_gt, optical_flow, start = batch
        action_logits, xyzrpy_pred, weights = self.actor(inputs, start)  # , idx)
        loss, immi_loss, aux_loss = self.compute_loss(
            demo, action_logits, xyzrpy_gt, xyzrpy_pred
        )
        self.log_dict(
            {"train/immi_loss": immi_loss, "train/aux_loss": aux_loss}, prog_bar=True
        )
        action_pred = torch.argmax(action_logits, dim=1)
        acc = (action_pred == demo).sum() / action_pred.numel()
        self.log("train/acc", acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, demo, xyzrpy_gt, optical_flow, start = batch
        action_logits, xyzrpy_pred, weights = self.actor(inputs, start)  # , idx)
        loss, immi_loss, aux_loss = self.compute_loss(
            demo, action_logits, xyzrpy_gt, xyzrpy_pred
        )
        self.log_dict(
            {"val/immi_loss": immi_loss, "val/aux_loss": aux_loss}, prog_bar=True
        )
        action_pred = torch.argmax(action_logits, dim=1)
        # number of corrects and total number
        return ((action_pred == demo).sum(), action_pred.numel())

    def validation_epoch_end(self, validation_step_outputs):
        numerator = 0
        divider = 0
        for cor, total in validation_step_outputs:
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
