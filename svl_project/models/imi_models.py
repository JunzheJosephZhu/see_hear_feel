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
        self.lstm = torch.nn.LSTM(self.embed_dim, 1024, 1, False)

        if args.loss_type == 'cce':
            print("loss: cce")
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(1024, 1024),
                # torch.nn.Linear(self.v_embeds_shape, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, pow(3, args.action_dim))
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

    def forward(self, v_input, t_input, freeze): #, idx):
        # debugging dataloader
        # print(f"\nFORWARD, idx shape: {len(idx), idx[0].shape}")
        # print(idx[0].cpu().numpy())
        # print(f"{v_inp.shape[0]} imgs found with shape {v_inp[0].shape}")
        # for i in range(v_inp.shape[0]):
        #     img = v_inp[i]
        #     print(img.permute(1, 2, 0).cpu().numpy().shape)
        #     cv2.imshow('input'+ str(i), img.cpu().permute(1, 2, 0).numpy())
        #     cv2.waitKey(100)
        """
            args: v_inp - [batch, seq_len, 3, H, W]
                  t_inp - [batch, seq_len, 3, H, W]
        """

        batch_size, seq_len, _, H, W = v_input.shape
        v_input = v_input.view(batch_size * seq_len, 3, H, W)
        t_input = t_input.view(batch_size * seq_len, 3, H, W)
        if freeze:
            with torch.no_grad():
                if self.use_vision:
                    v_embeds = self.v_encoder(v_input).detach()
                    v_embeds = torch.reshape(v_embeds, (-1, self.v_embeds_shape))
                if self.use_tactile:
                    t_embeds = self.t_encoder(t_input).detach()
                    if self.use_layernorm:
                        t_embeds = torch.reshape(t_embeds, (-1, self.v_embeds_shape))
                    else:
                        t_embeds = torch.reshape(t_embeds, (-1, self.t_embeds_shape))

        else:
            if self.use_vision:
                v_embeds = self.v_encoder(v_input)
                v_embeds = v_embeds.view(batch_size, seq_len, self.v_embeds_shape)
            if self.use_tactile:
                t_embeds = self.t_encoder(t_input)
                t_embeds = t_embeds.view(batch_size, seq_len, self.t_embeds_shape)

        embeds = []
        if self.use_vision:
            embeds.append(v_embeds)
        if self.use_tactile:
            embeds.append(t_embeds)
        # mlp_inp = torch.concat(embeds, dim=-1)
        # v_embed - (batch_size, seq_len, v_embed_dim)
        # t_embed - (batch_size, seq_len, t_embed_dim)

        lstm_inp = torch.concat(embeds, dim=-1)
        mlp_inp, _ = self.lstm(lstm_inp)
        
            # out = torch.stack(embeds, dim=0)
            # mlp_inp = self.layernorm(mlp_inp)
            # print(out.shape)
            # mlp_inp = torch.concat([out[i] for i in range(out.size(0))], 1)

        # print(embeds[0].shape)
        # plt.plot(v_embeds.cpu().detach().numpy()[0], 'b')
        # plt.plot(t_embeds.cpu().detach().numpy()[0], 'r')

        # print(f"mlp inp shape {mlp_inp.shape}")
        # mlp_temp = mlp_inp.cpu().detach().numpy()
        # # plt.figure()
        # plt.plot(mlp_temp[0])
        # plt.show()
        
        ## MHA debugging
        # plt.figure()
        # mha_temp = weights.cpu().detach().numpy()
        # print(f"mha {mha_temp[0]}")
        # plt.imshow(mha_temp[0])

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