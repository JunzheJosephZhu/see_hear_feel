from torch.nn.modules.activation import MultiheadAttention
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor
import torch
from torch import nn
# from engines.imi_engine import Future_Prediction
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt


class Imitation_Baseline_Actor_Tuning(torch.nn.Module):
    def __init__(self, v_encoder, args):
        super().__init__()
        self.v_encoder = v_encoder
        self.mlp = None
        self.embed_dim = args.embed_dim * args.num_stack * args.num_camera
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
        return action_logits

class Imitation_Actor_Ablation(torch.nn.Module):
    def __init__(self, v_encoder, t_encoder, a_encoder, args):
        super().__init__()
        self.v_encoder = v_encoder
        self.t_encoder = t_encoder
        self.a_encoder = a_encoder
        self.mlp = None
        self.v_embeds_shape = args.embed_dim_v * args.num_camera
        self.t_embeds_shape = args.embed_dim_t
        self.use_vision = False
        self.use_tactile = False
        self.use_audio = False
        self.use_mha = args.use_mha
        self.use_layernorm = args.use_layernorm
                
        ## load models
        self.ablation = args.ablation
        self.modalities = self.ablation.split('_')
        print(f"Using modalities: {self.modalities}")
        print(f"Using tactile flow: {args.use_flow}")
        self.embed_dim = 0
        self.use_vision = 'v' in self.modalities
        self.use_tactile = 't' in self.modalities
        self.use_audio = 'a' in self.modalities
        if self.use_vision:
            self.embed_dim += self.v_embeds_shape
        if self.use_tactile:
            self.embed_dim += self.t_embeds_shape
        self.lstm = torch.nn.LSTM(input_size=self.embed_dim, hidden_size=256, num_layers=1, batch_first=True)

        if args.loss_type == 'cce':
            print("loss: cce")
            # self.mlp = torch.nn.Sequential(
            #     torch.nn.Linear(1024, 1024),
            #     # torch.nn.Linear(self.v_embeds_shape, 1024),
            #     torch.nn.ReLU(),
            #     torch.nn.Linear(1024, 1024),
            #     torch.nn.ReLU(),
            #     torch.nn.Linear(1024, pow(3, args.action_dim))
            # )
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(256, 256),
                # torch.nn.Linear(self.v_embeds_shape, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 3 * args.action_dim) #pow(3, args.action_dim))
            )
            
        elif args.loss_type == 'mse':
            print("loss: mse")
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(self.embed_dim, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, args.action_dim),
                torch.nn.Tanh()
            )

    def forward(self, v_inputs, t_input, freeze, prev_hidden=None): #, idx):
        """
            args: v_inp - [batch, seq_len, 3, H, W]
                  t_inp - [batch, seq_len, 3, H, W]
        """



        with torch.set_grad_enabled(not freeze and self.training):
            batch_size, seq_len, _, H2, W2 = t_input.shape
            t_input = t_input.view(batch_size * seq_len, 3, H2, W2)
            if self.use_vision:
                v_embeds = []
                for v_input in v_inputs:
                    batch_size, seq_len, _, H1, W1 = v_input.shape
                    v_input = v_input.view(batch_size * seq_len, 3, H1, W1)
                    v_embed = self.v_encoder(v_input)
                    v_embed = v_embed.view(batch_size, seq_len, -1)
                    v_embeds.append(v_embed)
                v_embeds = torch.cat(v_embeds, -1)
                assert v_embeds.size(-1) == self.v_embeds_shape
            if self.use_tactile:
                t_embeds = self.t_encoder(t_input)
                t_embeds = t_embeds.view(batch_size, seq_len, self.t_embeds_shape)

        embeds = []
        if self.use_vision:
            embeds.append(v_embeds)
        if self.use_tactile:
            embeds.append(t_embeds)

        # v_embeds - (batch_size, seq_len, v_embed_dim)
        # t_embeds - (batch_size, seq_len, t_embed_dim)
        lstm_inp = torch.cat(embeds, dim=-1)
        mlp_inp, next_hidden = self.lstm(lstm_inp, prev_hidden)

        action_logits = self.mlp(mlp_inp)
        # print(action_logits)
        return action_logits, next_hidden

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