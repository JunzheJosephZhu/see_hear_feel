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
task2actiondim = {"pouring": 2, "insertion": 3}

class Imitation_Actor_Ablation(torch.nn.Module):
    def __init__(self, v_encoder, t_encoder, a_encoder, args):
        super().__init__()
        self.v_encoder = v_encoder
        self.t_encoder = t_encoder
        self.a_encoder = a_encoder
        self.mlp = None
        self.layernorm_embed_shape = args.encoder_dim * (2 * args.num_stack - 1)
        self.encoder_dim = args.encoder_dim
        self.ablation = args.ablation
        self.use_vision = False
        self.use_tactile = False
        self.use_audio = False
        self.use_mha = args.use_mha
        self.query = nn.Parameter(torch.randn(1, 1, self.layernorm_embed_shape))
            
        ## load models
        self.modalities = self.ablation.split('_')
        print(f"Using modalities: {self.modalities}")
        self.embed_dim = self.layernorm_embed_shape * len(self.modalities)
        self.layernorm = nn.LayerNorm(self.layernorm_embed_shape)
        self.mha = MultiheadAttention(self.layernorm_embed_shape, args.num_heads)
        self.bottleneck = nn.Linear(self.embed_dim, self.layernorm_embed_shape) # if we dont use mha

        # action_dim = 3 ** task2actiondim[args.task]

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.layernorm_embed_shape, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 3 ** args.action_dim)
        )
        self.aux_mlp = torch.nn.Linear(self.layernorm_embed_shape, 6)


    def forward(self, inputs):
        '''
        Args:
            cam_fixed_framestack, cam_gripper_framestack, tactile_framestack, audio_clip_g, audio_clip_h
            vf_inp: [batch, num_stack, 3, H, W]
            vg_inp: [batch, num_stack, 3, H, W]
            t_inp: [batch, num_stack, 3, H, W]
            a_inp: [batch, 1, T]
        
        '''

        vf_inp, vg_inp, t_inp, audio_g, audio_h = inputs
        embeds = []
        if "vf" in self.modalities:
            batch, num_stack, _, Hv, Wv = vf_inp.shape
            vf_inp = vf_inp.view(batch * num_stack, 3, Hv, Wv) 
            vf_embeds = self.v_encoder(vf_inp) # [batch * num_stack, encoder_dim]
            vf_embeds = vf_embeds.view(-1, num_stack, self.encoder_dim) # [batch, num_stack, encoder_dim]
            # concat the embedding and its diff with the next frame
            vf_embeds_plus_diff = torch.cat([torch.cat((vf_embeds[:, i] - vf_embeds[:, i-1], vf_embeds[:, i]), 1) for i in range(1, num_stack)], 1)
            # add the first frame embed
            vf_embeds_plus_diff = torch.cat((vf_embeds[:, 0], vf_embeds_plus_diff), 1)
            vf_embeds_plus_diff = vf_embeds_plus_diff.view(-1, self.layernorm_embed_shape) # [batch, encoder_dim * (num_stack + num_stack - 1)]
            embeds.append(vf_embeds_plus_diff)
        if "vg" in self.modalities:
            batch, num_stack, _, Hv, Wv = vg_inp.shape
            vg_inp = vg_inp.view(batch * num_stack, 3, Hv, Wv) 
            vg_embeds = self.v_encoder(vg_inp) # [batch * num_stack, encoder_dim]
            vg_embeds = vg_embeds.view(-1, num_stack, self.encoder_dim)
            # concat the embedding and its diff with the next frame
            vg_embeds_plus_diff = torch.cat([torch.cat((vg_embeds[:, i] - vg_embeds[:, i-1], vg_embeds[:, i]), 1) for i in range(1, num_stack)], 1)
            # add the first frame embed
            vg_embeds_plus_diff = torch.cat((vg_embeds[:, 0], vg_embeds_plus_diff), 1)
            vg_embeds_plus_diff = vg_embeds_plus_diff.view(-1, self.layernorm_embed_shape) # [batch, encoder_dim * (num_stack + num_stack - 1)]
            embeds.append(vg_embeds_plus_diff)
        if "t" in self.modalities:
            batch, num_stack, Ct, Ht, Wt = t_inp.shape
            t_inp = t_inp.view(batch * num_stack, Ct, Ht, Wt)
            t_embeds = self.t_encoder(t_inp)
            t_embeds = t_embeds.view(-1, num_stack, self.encoder_dim)
            # concat the embedding and its diff with the next frame
            t_embeds_plus_diff = torch.cat([torch.cat((t_embeds[:, i] - t_embeds[:, i-1], t_embeds[:, i]), 1) for i in range(1, num_stack)], 1)
            # add the first frame embed
            t_embeds_plus_diff = torch.cat((t_embeds[:, 0], t_embeds_plus_diff), 1)
            t_embeds_plus_diff = t_embeds_plus_diff.view(-1, self.layernorm_embed_shape) # [batch, encoder_dim * (num_stack + num_stack - 1)]
            embeds.append(t_embeds_plus_diff)
        if "ah" in self.modalities:
            batch, _, _ = audio_h.shape
            ah_embeds = self.a_encoder(audio_h)
            ah_embeds = ah_embeds.view(-1, self.layernorm_embed_shape)
            embeds.append(ah_embeds)
        if "ag" in self.modalities:
            batch, _, _ = audio_g.shape
            ag_embeds = self.a_encoder(audio_g)
            ag_embeds = ag_embeds.view(-1, self.layernorm_embed_shape)
            embeds.append(ag_embeds)
        
        if self.use_mha:
            mlp_inp = torch.stack(embeds, dim=0) # [3, batch, D]
            # batch first=False, (L, N, E)
            # query = self.query.repeat(1, batch, 1) # [1, 1, D] -> [1, batch, D]
            # change back to 3*3
            mha_out, weights = self.mha(mlp_inp, mlp_inp, mlp_inp) # [1, batch, D]
            mha_out += mlp_inp
            mlp_inp = torch.concat([mha_out[i] for i in range(mha_out.shape[0])], 1)
            mlp_inp = self.bottleneck(mlp_inp)        
            # mlp_inp = mha_out.squeeze(0) # [batch, D]
        else:
            mlp_inp = torch.cat(embeds, dim=-1)
            mlp_inp = self.bottleneck(mlp_inp)
            weights = None

        action_logits = self.mlp(mlp_inp)
        xyzrpy = self.aux_mlp(mlp_inp)
        return action_logits, xyzrpy, weights

if __name__ == "__main__":
    pass
    # vision_encoder = make_vision_encoder(128)
    # empty_input = torch.zeros((1, 3, 64, 101))
    # print(vision_encoder(empty_input).shape)