from torch.nn.modules.activation import MultiheadAttention
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor
import torch
from torch import nn


class Encoder(torch.nn.Module):
    def __init__(self, feature_extractor, out_dim):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.projection = nn.Linear(512, out_dim)

    def forward(self, x):
        feats = self.feature_extractor(x)["avgpool"]
        feats = feats.squeeze(3).squeeze(2)
        return self.projection(feats)

def make_vision_encoder(out_dim):
    vision_extractor = resnet18(pretrained=True)
    vision_extractor = create_feature_extractor(vision_extractor, ["avgpool"])
    return Encoder(vision_extractor, out_dim)

def make_tactile_encoder(out_dim):
    tactile_extractor = resnet18()
    tactile_extractor = create_feature_extractor(tactile_extractor, ["avgpool"])
    return Encoder(tactile_extractor, out_dim)

def make_audio_encoder(out_dim):
    audio_extractor = resnet18()
    audio_extractor.conv1 = nn.Conv2d(
        2, 64, kernel_size=7, stride=1, padding=3, bias=False
    )
    audio_extractor = create_feature_extractor(audio_extractor, ["avgpool"])
    return Encoder(audio_extractor, out_dim)

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

class Forward_Model(torch.nn.Module):
    def __init__(self, embed_dim, action_dim):
        super().__init__()
        self.action_encoder = torch.nn.Sequential(torch.nn.Linear(action_dim, embed_dim), torch.nn.ReLU(), torch.nn.Linear(embed_dim, embed_dim))
        self.mlp_v = torch.nn.Sequential(torch.nn.Linear(embed_dim * 4, embed_dim), torch.nn.ReLU(), torch.nn.Linear(embed_dim, embed_dim), torch.nn.ReLU(), torch.nn.Linear(embed_dim, embed_dim))
        self.mlp_a = torch.nn.Sequential(torch.nn.Linear(embed_dim * 4 , embed_dim), torch.nn.ReLU(), torch.nn.Linear(embed_dim, embed_dim), torch.nn.ReLU(), torch.nn.Linear(embed_dim, embed_dim))
        self.mlp_t = torch.nn.Sequential(torch.nn.Linear(embed_dim * 4, embed_dim), torch.nn.ReLU(), torch.nn.Linear(embed_dim, embed_dim), torch.nn.ReLU(), torch.nn.Linear(embed_dim, embed_dim))

    def forward(self, v_out, a_out, t_out, actions):
        action_embed = self.action_encoder(actions)
        fused = torch.cat([v_out, a_out, t_out, action_embed], dim=1)
        v_pred = self.mlp_v(fused)
        a_pred = self.mlp_a(fused)
        t_pred = self.mlp_t(fused)
        return v_pred, a_pred, t_pred

class Immitation_Actor(torch.nn.Module):
    def __init__(self, v_encoder, a_encoder, t_encoder, embed_dim, num_heads, action_dim):
        super().__init__()
        self.v_encoder = v_encoder
        self.a_encoder = a_encoder
        self.t_encoder = t_encoder
        self.fusion = Attention_Fusion(embed_dim, num_heads)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(embed_dim * 3, embed_dim), torch.nn.ReLU(), torch.nn.Linear(embed_dim, embed_dim), torch.nn.ReLU(), torch.nn.Linear(embed_dim, 3 * action_dim))

    def forward(self, v_inp, a_inp, t_inp, freeze):
        if freeze:
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

class Immitation_Actor_Reg(torch.nn.Module):
    def __init__(self, v_encoder, a_encoder, t_encoder, embed_dim, num_heads, action_dim):
        super().__init__()
        self.v_encoder = v_encoder
        self.a_encoder = a_encoder
        self.t_encoder = t_encoder
        self.fusion = Attention_Fusion(embed_dim, num_heads)
        self.mlp = torch.nn.Sequential(
                torch.nn.Linear(embed_dim * 3, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, action_dim),
                torch.nn.Tanh()
        )
        
    def forward(self, v_inp, a_inp, t_inp, freeze):
        if freeze:
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
        # self.mlp = torch.nn.Sequential(torch.nn.Linear(embed_dim * 2, embed_dim), torch.nn.ReLU(), torch.nn.Linear(embed_dim, 3 * action_dim))
        self.mlp = torch.nn.Sequential(
                torch.nn.Linear(embed_dim * 2, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 1024),
                torch.nn.Tanh(),
                torch.nn.Linear(1024, action_dim),
                torch.nn.Tanh()
        )
            
    def forward(self, v_gripper_inp, v_fixed_inp):
        v_gripper_embed = self.v_gripper_encoder(v_gripper_inp)
        v_fixed_embed = self.v_fixed_encoder(v_fixed_inp)
        mlp_inp = torch.cat([v_gripper_embed, v_fixed_embed], dim=1)
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
            # torch.nn.Linear(embed_dim, 3 * action_dim)
            torch.nn.Linear(embed_dim, action_dim),
            torch.nn.Tanh()
        )

    def forward(self, pose_inp, freeze):
        action_logits = self.mlp(pose_inp)
        print("action_logits: {}".format(action_logits))
        # actions = self.mlp(pose_inp)
        return action_logits



if __name__ == "__main__":
    vision_encoder = make_vision_encoder(128)
    empty_input = torch.zeros((1, 3, 64, 101))
    print(vision_encoder(empty_input).shape)

    fusion = Attention_Fusion(128, 4)
    forward_model = Forward_Model(128, 3)
    v_inp = torch.rand((8, 128))
    a_inp = torch.rand((8, 128))
    t_inp = torch.rand((8, 128))
    action = torch.rand((8, 3))
    v_out, a_out, t_out = fusion(v_inp, a_inp, t_inp)
    v_pred, a_pred, t_pred = forward_model(v_out, a_out, t_out, action)