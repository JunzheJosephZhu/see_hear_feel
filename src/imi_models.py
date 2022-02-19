from torch.nn.modules.activation import MultiheadAttention
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor
import torch
from torch import nn
from engine import Future_Prediction
import cv2
import numpy as np

class Encoder(torch.nn.Module):
    def __init__(self, feature_extractor, out_dim):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.projection = nn.Linear(512, out_dim)

    def forward(self, x):
        feats = self.feature_extractor(x)["avgpool"]
        feats = feats.squeeze(3).squeeze(2)
        return self.projection(feats)

# def make_vision_encoder(out_dim, channel):
def make_vision_encoder(out_dim):
    vision_extractor = resnet18(pretrained=True)
    # # change the first conv layer to fit 30 channels
    # vision_extractor.conv1 = nn.Conv2d(
    #     channel, 64, kernel_size=7, stride=2, padding=3, bias=False
    # )
    vision_extractor = create_feature_extractor(vision_extractor, ["avgpool"])
    return Encoder(vision_extractor, out_dim)

class Imitation_Baseline_Actor_Tuning(torch.nn.Module):
    def __init__(self, v_encoder, args):
        super().__init__()
        self.v_encoder = v_encoder
        self.mlp = None
        embed_dim = args.embed_dim * args.num_stack * 2
        print('\n'.join(['*' * 50 + 'imi_models', 'embed_dim:', f'{args.embed_dim} * {args.num_stack} = {embed_dim}']))
        if args.loss_type == 'cce':
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(embed_dim, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 3 * args.action_dim)
            )
        elif args.loss_type == 'mse':
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(embed_dim, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, args.action_dim),
                torch.nn.Tanh()
            )

    def forward(self, v_inp, freeze, idx):
        # debugging dataloader
        # print(f"\nFORWARD, idx shape: {len(idx), idx[0].shape}")
        # print(idx[0].cpu().numpy())
        # print(len(v_inp), v_inp[0].shape)
        # for i in range(v_inp.shape[1] // 3):
        #     img = v_inp[0, 3*i : 3*i+3, :, :]
        #     print(img.permute(1, 2, 0).cpu().numpy().shape)
        #     cv2.imshow('input'+ str(i), img.cpu().permute(1, 2, 0).numpy())
        #     cv2.waitKey(100)
        if freeze:
            with torch.no_grad():
                v_embeds = [self.v_encoder(v_inp_i).detach() for v_inp_i in v_inp]
            # print('\n'.join(('*' * 50 + 'imi_models', 'v_embeds:', f'{len(v_embeds)}')))
            # v_embeds = v_embeds.detach()
        else:
            v_embeds = [self.v_encoder(v_inp_i) for v_inp_i in v_inp]
        mlp_inp = torch.concat(v_embeds, dim=-1)
        print('\n'.join(['*' * 50 + 'imi_models', 'v_embeds:', f'len = {len(v_embeds)}']))
        action_logits = self.mlp(mlp_inp)
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
    vision_encoder = make_vision_encoder(128)
    empty_input = torch.zeros((1, 3, 64, 101))
    print(vision_encoder(empty_input).shape)