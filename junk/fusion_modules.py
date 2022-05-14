from torch.nn.modules.activation import MultiheadAttention
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor,get_graph_node_names
import torch
from torch import nn
from engines.imi_engine import Future_Prediction
import cv2
import numpy as np

class Attention_Fusion(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.layernorm = torch.nn.LayerNorm(embed_dim)
        self.mha = MultiheadAttention(embed_dim, num_heads)

    def forward(self, v_inp, a_inp, t_inp):
        inp = torch.stack([v_inp, a_inp, t_inp], dim=0)
        sublayer_out, weights = self.mha(inp, inp, inp)
        out = self.layernorm(sublayer_out + inp)
        v_out, a_out, t_out = out[:]
        return v_out, a_out, t_out

