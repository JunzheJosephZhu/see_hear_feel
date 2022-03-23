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
        self.use_vision = False
        self.use_tactile = False
        self.use_audio = False
        
        # if self.ablation == 'v_t':
        #     self.embed_dim = self.v_embeds_shape + self.t_embeds_shape
        #     self.use_vision = True
        #     self.use_tactile = True
        # elif self.ablation == 'v_a':
        #     self.embed_dim = self.v_embeds_shape + self.a_embeds_shape
        #     self.use_vision = True
        #     self.use_audio = True
        # elif self.ablation == 'v_t_a':
        #     self.embed_dim = self.v_embeds_shape + self.t_embeds_shape + self.a_embeds_shape
        #     self.use_vision = True
        #     self.use_tactile = True
        #     self.use_audio = True
        # elif self.ablation == 'v':
        #     self.embed_dim = self.v_embeds_shape
        #     self.use_vision = True
        
        ## to enable more combinations
        self.modalities = self.ablation.split('_')
        print(f"Using modalities: {self.modalities}")
        print(f"Using tactile flow: {args.use_flow}")
        self.embed_dim = 0
        self.use_vision = 'v' in self.modalities
        self.use_tactile = 't' in self.modalities
        self.use_audio =  'a' in self.modalities
        if self.use_vision:
            self.embed_dim += self.v_embeds_shape
        if self.use_tactile:
            self.embed_dim += self.t_embeds_shape
        if self.use_audio:
            self.embed_dim += self.a_embeds_shape

        self.num_heads = args.num_heads
        self.layernorm = torch.nn.LayerNorm(self.v_embeds_shape)
        self.mha = MultiheadAttention(self.v_embeds_shape, self.num_heads)

        # print('\n'.join(['*' * 50 + 'imi_models', 'embed_dim:', f'{args.embed_dim} * {args.num_stack} = {embed_dim}']))
        if args.loss_type == 'cce':
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(self.embed_dim, 1024),
                # torch.nn.Linear(self.v_embeds_shape, 1024),
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
                if self.use_vision:
                    v_embeds = self.v_encoder(v_inp).detach()
                    v_embeds = torch.reshape(v_embeds, (-1, self.v_embeds_shape))
                if self.use_tactile:
                    t_embeds = self.t_encoder(t_inp).detach()
                    t_embeds = torch.reshape(t_embeds, (-1, self.t_embeds_shape))
                if self.use_audio:
                    a_embeds = self.a_encoder(a_inp).detach()
                    a_embeds = torch.reshape(a_embeds, (-1, self.a_embeds_shape))
        else:
            if self.use_vision:
                v_embeds = self.v_encoder(v_inp)
                v_embeds = torch.reshape(v_embeds, (-1, self.v_embeds_shape))
            if self.use_tactile:
                t_embeds = self.t_encoder(t_inp)
                t_embeds = torch.reshape(t_embeds, (-1, self.t_embeds_shape))
            if self.use_audio:
                a_embeds = self.a_encoder(a_inp)
                a_embeds = torch.reshape(a_embeds, (-1, self.a_embeds_shape))
        
        # if self.ablation == 'v_t':
        #     mlp_inp = torch.concat((v_embeds,t_embeds), dim=-1)
        # elif self.ablation == 'v_a':
        #     mlp_inp = torch.concat((v_embeds, a_embeds), dim=-1)
        # elif self.ablation == 'v_t_a':
        #     mlp_inp = torch.concat((v_embeds, t_embeds, a_embeds), dim=-1)
        # elif self.ablation == 'v':
        #     mlp_inp = v_embeds
        
        ## to enable more combinations
        embeds = []
        if self.use_vision:
            embeds.append(v_embeds)
        if self.use_tactile:
            embeds.append(t_embeds)
        if self.use_audio:
            embeds.append(a_embeds)
        # mlp_inp = torch.concat(embeds, dim=-1)

        # print(embeds[0].shape)
        # plt.plot(v_embeds.cpu().detach().numpy()[0], 'b')
        # plt.plot(t_embeds.cpu().detach().numpy()[0], 'r')
        
        mlp_inp = torch.stack(embeds, dim=0)
        sublayer_out, weights = self.mha(mlp_inp, mlp_inp, mlp_inp)
        out = self.layernorm(sublayer_out + mlp_inp)
        # ## option 1: average
        # # mlp_inp = torch.mean(out, dim=0)
        # ## option 2: concat
        mlp_inp = torch.concat([out[i] for i in range(out.size(0))], 1)
        
        # print(f"mlp inp shape {mlp_inp.shape}")
        # mlp_temp = mlp_inp.cpu().detach().numpy()
        # plt.figure()
        # plt.plot(mlp_temp[0])
        # # plt.figure()
        # mha_temp = weights.cpu().detach().numpy()
        # print(f"mha {mha_temp[0]}")
        # plt.show()
        
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