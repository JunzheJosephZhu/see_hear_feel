from torch.nn.modules.activation import MultiheadAttention
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor,get_graph_node_names
import torch
from torch import nn
from engines.imi_engine import Future_Prediction
import cv2
import numpy as np



class Immitation_Actor(torch.nn.Module):
    def __init__(self, v_encoder, a_encoder, t_encoder, embed_dim, num_heads, action_dim):
        super().__init__()
        self.v_encoder = v_encoder
        self.a_encoder = a_encoder
        self.t_encoder = t_encoder
        self.fusion = Attention_Fusion(embed_dim, num_heads)
        #for classification
        # self.mlp = torch.nn.Sequential(torch.nms.MelSpectrogram(
            # sample_rate=sr, n_fft=int(n.Linear(embed_dim * 3, embed_dim), torch.nn.ReLU(), torch.nn.Linear(embed_dim, 3 * action_dim))
        #for regression
        self.mlp = torch.nn.Sequential(torch.nn.Linear(embed_dim * 3, 1024),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(1024, 1024),
                                       torch.nn.ReLU(), torch.nn.Linear(1024, action_dim), torch.nn.Tanh())

    def forward(self, v_inp, a_inp, t_inp, freeze):
        if freeze:
            # print("freeze")
            with torch.no_grad():
                v_embed = self.v_encoder(v_inp)
                a_embed = self.a_encoder(a_inp)
                t_embed = self.t_encoder(t_inp)
            v_embed, a_embed, t_embed = v_embed.detach(), a_embed.detach(), t_embed.detach()
        else:
            v_embed = self.v_encoder(v_inp)
            a_embed = self.a_encoder(a_inp)
            t_embed = self.t_encoder(t_inp)
        v_out, a_out, t_out = self.fusion(v_embed, a_embed, t_embed)
        mlp_inp = torch.cat([v_out, a_out, t_out], dim=1)
        action_logits = self.mlp(mlp_inp)
        return action_logits


class Immitation_Baseline_Actor(torch.nn.Module):
    def __init__(self, v_gripper_encoder, v_fixed_encoder, embed_dim, action_dim):
        super().__init__()
        self.v_gripper_encoder = v_gripper_encoder
        self.v_fixed_encoder = v_fixed_encoder
        self.mlp = torch.nn.Sequential(torch.nn.Linear(embed_dim * 2, 1024),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(1024, 1024),
                                       torch.nn.ReLU(), torch.nn.Linear(1024, action_dim),torch.nn.Tanh())

    def forward(self, v_gripper_inp, v_fixed_inp, freeze):
        if freeze:
            with torch.no_grad():
                v_gripper_embed = self.v_gripper_encoder(v_gripper_inp)
                v_fixed_embed = self.v_fixed_encoder(v_fixed_inp)
            v_gripper_embed, v_fixed_embed = v_gripper_embed.detach(), v_fixed_embed.detach()
        else:
            v_gripper_embed = self.v_gripper_encoder(v_gripper_inp)
            v_fixed_embed = self.v_fixed_encoder(v_fixed_inp)
        mlp_inp = torch.cat([v_gripper_embed, v_fixed_embed], dim=1)
        action_logits = self.mlp(mlp_inp)
        return action_logits

class Immitation_Baseline_Actor_Tuning(torch.nn.Module):
    def __init__(self, v_encoder, embed_dim, action_dim):
        super().__init__()
        self.v_encoder = v_encoder
        self.mlp = torch.nn.Sequential(torch.nn.Linear(embed_dim, 1024),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(1024, 1024),
                                       torch.nn.ReLU(), torch.nn.Linear(1024, action_dim),torch.nn.Tanh())

    def forward(self, v_inp, freeze):
        if freeze:
            with torch.no_grad():
                v_embed = self.v_encoder(v_inp)
            v_embed = v_embed.detach()
        else:
            v_embed = self.v_encoder(v_inp)
        mlp_inp = v_embed
        action_logits = self.mlp(mlp_inp)
        return action_logits

class Immitation_Baseline_Actor_Tuning(torch.nn.Module):
    def __init__(self, args, v_encoder, embed_dim, action_dim):
        super().__init__()
        self.v_encoder = v_encoder
        self.mlp = None
        if args.loss_type == 'cce':
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(embed_dim, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 3 * action_dim)
            )
        elif args.loss_type == 'mse':
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(embed_dim, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, action_dim),
                torch.nn.Tanh()
            )

    def forward(self, v_inp, freeze, idx):
        print(f"\nFORWARD, idx shape: {idx.shape}")
        print(idx.cpu().numpy())
        print(v_inp.shape)
        for i in range(v_inp.shape[1] // 3):
            img = v_inp[0, 3*i : 3*i+3, :, :]
            print(img.permute(1, 2, 0).cpu().numpy().shape)
            cv2.imshow('input'+ str(i), img.cpu().permute(1, 2, 0).numpy())
            cv2.waitKey(100)
        if freeze:
            with torch.no_grad():
                v_embed = self.v_encoder(v_inp)
            v_embed = v_embed.detach()
        else:
            v_embed = self.v_encoder(v_inp)
        mlp_inp = v_embed
        action_logits = self.mlp(mlp_inp)
        return action_logits

class Immitation_Pose_Baseline_Actor(torch.nn.Module):
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


class Immitation_Baseline_Actor_Classify(torch.nn.Module):
    def __init__(self, v_gripper_encoder, v_fixed_encoder, embed_dim, action_dim):
        super().__init__()
        self.v_gripper_encoder = v_gripper_encoder
        self.v_fixed_encoder = v_fixed_encoder
        self.mlp = torch.nn.Sequential(torch.nn.Linear(embed_dim * 2, embed_dim),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(embed_dim, 3 * action_dim))

    def forward(self, v_gripper_inp, v_fixed_inp, freeze):
        print(v_gripper_inp.shape)
        for i in range(8):
            cv2.imshow('input', v_gripper_inp[0,3*i:3*i+3,:,:])
            cv2.waitKey(1)
        if freeze:
            with torch.no_grad():
                v_gripper_embed = self.v_gripper_encoder(v_gripper_inp)
                v_fixed_embed = self.v_fixed_encoder(v_fixed_inp)
            v_gripper_embed, v_fixed_embed = v_gripper_embed.detach(), v_fixed_embed.detach()
        else:
            v_gripper_embed = self.v_gripper_encoder(v_gripper_inp)
            v_fixed_embed = self.v_fixed_encoder(v_fixed_inp)
        mlp_inp = torch.cat([v_gripper_embed, v_fixed_embed], dim=1)
        action_logits = self.mlp(mlp_inp)
        return action_logits

if __name__ == "__main__":
    vision_encoder = make_vision_encoder(3)
    vision_decoder = make_vision_decoder()
    # empty_input = torch.zeros((1, 2, 64, 101))
    empty_input = torch.zeros((1, 3, 640, 480))
    bottleneck = vision_encoder(empty_input)
    print(bottleneck.shape)
    recon = vision_decoder(bottleneck)
    print(recon.shape)


    fusion = Attention_Fusion(128, 4)
    forward_model = Forward_Model(128, 3)
    v_inp = torch.rand((8, 128))
    a_inp = torch.rand((8, 128))
    t_inp = torch.rand((8, 128))
    action = torch.rand((8, 3))
    v_out, a_out, t_out = fusion(v_inp, a_inp, t_inp)
    v_pred, a_pred, t_pred = forward_model(v_out, a_out, t_out, action)