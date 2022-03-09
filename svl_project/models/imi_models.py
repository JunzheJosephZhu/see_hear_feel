from torch.nn.modules.activation import MultiheadAttention
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor
import torch
from torch import nn
# from engines.imi_engine import Future_Prediction
import cv2
import numpy as np
import time


class Imitation_Baseline_Actor_Tuning(torch.nn.Module):
    def __init__(self, v_encoder, args):
        super().__init__()
        self.v_encoder = v_encoder
        self.mlp = None
        self.embed_dim = args.embed_dim * args.num_stack * args.num_camera
        # print('\n'.join(['*' * 50 + 'imi_models', 'embed_dim:', f'{args.embed_dim} * {args.num_stack} = {embed_dim}']))
        if args.loss_type == 'cce':
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(self.embed_dim, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 9 * args.action_dim)
            )
        elif args.loss_type == 'mse':
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(self.embed_dim, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, args.action_dim),
                torch.nn.Tanh()
            )

    def forward(self, v_inp, freeze): #, idx):
        # debugging dataloader
        # print(f"\nFORWARD, idx shape: {len(idx), idx[0].shape}")
        # print(idx[0].cpu().numpy())
        # print(f"{v_inp.shape[0]} imgs found with shape {v_inp[0].shape}")
        # for i in range(v_inp.shape[0]):
        #     img = v_inp[i]
        #     print(img.permute(1, 2, 0).cpu().numpy().shape)
        #     cv2.imshow('input'+ str(i), img.cpu().permute(1, 2, 0).numpy())
        #     cv2.waitKey(100)
        if freeze:
            with torch.no_grad():
                v_embeds = self.v_encoder(v_inp).detach()
        else:
            v_embeds = self.v_encoder(v_inp)
        mlp_inp = torch.reshape(v_embeds, (-1, self.embed_dim))
        action_logits = self.mlp(mlp_inp)
        # print(action_logits)
        return action_logits

class Imitation_Actor_Ablation(torch.nn.Module):
    def __init__(self, v_encoder, t_encoder, a_encoder, args):
        super().__init__()
        self.v_encoder = v_encoder
        self.t_encoder = t_encoder
        self.a_encoder = a_encoder
        self.mlp = None
        self.v_embeds_shape = args.embed_dim_v * args.num_stack * args.num_camera
        self.t_embeds_shape = args.embed_dim_t * args.num_stack
        self.a_embeds_shape = args.embed_dim_a
        self.ablation = args.ablation
        if self.ablation == 'v_t':
            self.embed_dim = self.v_embeds_shape + self.t_embeds_shape
        elif self.ablation == 'v_a':
            self.embed_dim = self.v_embeds_shape + self.a_embeds_shape
        elif self.ablation == 'v_t_a':
            self.embed_dim = self.v_embeds_shape + self.t_embeds_shape + self.a_embeds_shape
        elif self.ablation == 'v':
            self.embed_dim = self.v_embeds_shape
        # print('\n'.join(['*' * 50 + 'imi_models', 'embed_dim:', f'{args.embed_dim} * {args.num_stack} = {embed_dim}']))
        if args.loss_type == 'cce':
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(self.embed_dim, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, pow(3, args.action_dim))
            )
        elif args.loss_type == 'mse':
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(self.embed_dim, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, args.action_dim),
                torch.nn.Tanh()
            )

    def forward(self, v_inp, t_inp, a_inp, freeze): #, idx):
        # debugging dataloader
        # print(f"\nFORWARD, idx shape: {len(idx), idx[0].shape}")
        # print(idx[0].cpu().numpy())
        # print(f"{v_inp.shape[0]} imgs found with shape {v_inp[0].shape}")
        # for i in range(v_inp.shape[0]):
        #     img = v_inp[i]
        #     print(img.permute(1, 2, 0).cpu().numpy().shape)
        #     cv2.imshow('input'+ str(i), img.cpu().permute(1, 2, 0).numpy())
        #     cv2.waitKey(100)
        if freeze:
            with torch.no_grad():
                v_embeds = self.v_encoder(v_inp).detach()
                t_embeds = self.t_encoder(t_inp).detach()
                a_embeds = self.a_encoder(a_inp).detach()
        else:
            v_embeds = self.v_encoder(v_inp)
            t_embeds = self.t_encoder(t_inp)
            a_embeds = self.a_encoder(a_inp)
        v_embeds = torch.reshape(v_embeds, (-1, self.v_embeds_shape))
        t_embeds = torch.reshape(t_embeds,(-1, self.t_embeds_shape))
        a_embeds = torch.reshape(a_embeds,(-1, self.a_embeds_shape))
        if self.ablation == 'v_t':
            mlp_inp = torch.concat((v_embeds,t_embeds), dim=-1)
        elif self.ablation == 'v_a':
            mlp_inp = torch.concat((v_embeds, a_embeds), dim=-1)
        elif self.ablation == 'v_t_a':
            mlp_inp = torch.concat((v_embeds, t_embeds, a_embeds), dim=-1)
        elif self.ablation == 'v':
            mlp_inp = v_embeds
        action_logits = self.mlp(mlp_inp)
        # print(action_logits)
        return action_logits

class Imitation_Pose_Baseline_Actor(torch.nn.Module):
    def __init__(self, embed_dim, action_dim):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(3, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim, 3 * action_dim),
            # torch.nn.Linear(embed_dim, action_dim),
            torch.nn.Tanh()
        )

    def forward(self, pose_inp, freeze):
        action_logits = self.mlp(pose_inp)
        # print("action_logits: {}".format(action_logits))
        # actions = self.mlp(pose_inp)
        return action_logits

if __name__ == "__main__":
    pass
    # vision_encoder = make_vision_encoder(128)
    # empty_input = torch.zeros((1, 3, 64, 101))
    # print(vision_encoder(empty_input).shape)