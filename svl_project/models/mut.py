
from turtle import forward
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import numpy as np
# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)




class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, qkv_bias, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = qkv_bias)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        '''
        x: [batch, num_tokens, dim]
        '''
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Block(nn.Module):

    def __init__(self, dim, heads, mlp_dim, qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0.1, norm_layer=nn.LayerNorm, attention_type='divided_space_time'):
        super().__init__()
        self.attention_type = attention_type
        assert(attention_type in ['divided_space_time', 'space_only','joint_space_time'])

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
           dim, heads=heads, qkv_bias=qkv_bias, dropout=attn_drop, dim_head=dim)

        ## Temporal Attention Parameters
        if self.attention_type == 'divided_space_time':
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(
              dim, heads=heads, qkv_bias=qkv_bias, dropout=attn_drop, dim_head=dim)
            self.temporal_fc = nn.Linear(dim, dim)

        ## drop path
        self.drop_path = nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        self.mlp = FeedForward(dim=dim, hidden_dim=mlp_dim, dropout=drop)


    def forward(self, x, T):
        '''
        x: [batch, 1 + num_frames * total_patches, dim] or [b 1+(t p) m]
        B: batch
        T: num_frames
        P: total_patches(num_patch per frame)
        '''
        B = x.size(0)
        P = (x.size(1) - 1) // T

        if self.attention_type in ['space_only', 'joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == 'divided_space_time':
            ## Temporal, class token is excluded from this part
            xt = x[:,1:,:] # [batch, num_frames * total_patches, dim]
            xt = rearrange(xt, 'b (t p) m -> (b p) t m',b=B, p=P, t=T)
            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
            res_temporal = rearrange(res_temporal, '(b p) t m -> b (t p) m',b=B, p=P, t=T)
            res_temporal = self.temporal_fc(res_temporal)
            xt = x[:,1:,:] + res_temporal

            ## Spatial
            init_cls_token = x[:,0,:].unsqueeze(1) # [batch, 1, dim]
            cls_token = init_cls_token.repeat(1, T, 1) # [batch, T, dim]
            cls_token = rearrange(cls_token, 'b t m -> (b t) m',b=B,t=T).unsqueeze(1) # [(b t) 1 m]
            xs = xt
            xs = rearrange(xs, 'b (t p) m -> (b t) p m',b=B, t=T, p=P)
            xs = torch.cat((cls_token, xs), 1) # [(b t) p+1 m]
            res_spatial = self.drop_path(self.attn(self.norm1(xs)))

            ### Taking care of CLS token
            cls_token = res_spatial[:,0,:] # [(b t) m]
            cls_token = rearrange(cls_token, '(b t) m -> b t m', b=B, t=T)
            cls_token = torch.mean(cls_token,1,True) ## averaging for every frame
            res_spatial = res_spatial[:,1:,:] # [(b t) p m]
            res_spatial = rearrange(res_spatial, '(b t) p m -> b (t p) m',b=B, p=P, t=T)
            res = res_spatial
            x = xt

            ## Mlp
            x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class Audio_Encoder(nn.Module):
    def __init__(self, input_channels, encoding_dim=512, strides=[5,2,2,2,2,2,2], kernel_widths=[10,3,3,3,3,2,2]):
        super().__init__()
        sr = 16000
        self.output_freq = sr / np.prod(strides)
        self.layers = nn.ModuleList()
        in_channel = input_channels
        self.encoding_dim = encoding_dim
        for stride, kernel_width in zip(strides, kernel_widths):
            self.layers.append(nn.Conv1d(in_channels=in_channel, out_channels=self.encoding_dim, kernel_size=kernel_width, stride=stride, padding=kernel_width // 2))
            self.layers.append(nn.ReLU())
            in_channel = self.encoding_dim
    
    def forward(self, x):
        """
        Args:
            x: [batch, input_channels, T]
        Return:
            encoding: [batch, T // prod(strides), encoding_dim]
        """
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(1, 2)
        return x

class MuT(nn.Module):
    def __init__(self, *, image_size, tactile_size, patch_size, num_stack, frameskip, fps, num_classes, dim, depth, qkv_bias, heads, mlp_dim, modal_enc_dim, time_enc_dim, ablation, channels = 3, audio_channels = 2, dim_head = 64, dropout = 0., emb_dropout = 0., drop_path=0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        tactile_height, tactile_width = pair(tactile_size)
        patch_height, patch_width = pair(patch_size)
        patch_dim = channels * patch_height * patch_width


        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert tactile_height % patch_height == 0 and tactile_width % patch_width == 0
        self.modalities = ablation.split('_')



        self.to_patch_embedding_v = nn.Sequential(
            Rearrange('b l c (h p1) (w p2) -> b l (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        ) # vision encoder
        self.to_patch_embedding_t = nn.Sequential(
            Rearrange('b l c (h p1) (w p2) -> b l (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        ) # tactile encoder
        self.audio_encoder = Audio_Encoder(audio_channels)
        self.to_patch_embedding_a = nn.Sequential(
            self.audio_encoder,
            nn.Linear(self.audio_encoder.encoding_dim, dim),
        ) # audio encoder

        # compute number of patches per timestep for each modality
        num_patches_v = (image_height // patch_height) * (image_width // patch_width)
        num_patches_a = (1 / fps) * frameskip / (1 / self.audio_encoder.output_freq)
        assert num_patches_a.is_integer()
        num_patches_a = int(num_patches_a)
        self.num_patches_a = num_patches_a
        num_patches_t = (tactile_height // patch_height) * (tactile_width // patch_width)

        # within_image positional encoding
        self.pos_embed_vf = nn.Parameter(torch.randn(1, 1, num_patches_v, dim))
        self.pos_embed_vg = nn.Parameter(torch.randn(1, 1, num_patches_v, dim))
        self.pos_embed_ah = nn.Parameter(torch.randn(1, 1, num_patches_a, dim))
        self.pos_embed_ag = nn.Parameter(torch.randn(1, 1, num_patches_a, dim))
        self.pos_embed_t = nn.Parameter(torch.randn(1, 1, num_patches_t, dim))
        # modality specific encodings
        self.modal_enc_vf = nn.Parameter(torch.randn(1, 1, num_patches_v, modal_enc_dim))
        self.modal_enc_vg = nn.Parameter(torch.randn(1, 1, num_patches_v, modal_enc_dim))
        self.modal_enc_ah = nn.Parameter(torch.randn(1, 1, num_patches_a, modal_enc_dim))
        self.modal_enc_ag = nn.Parameter(torch.randn(1, 1, num_patches_a, modal_enc_dim))
        self.modal_enc_t = nn.Parameter(torch.randn(1, 1, num_patches_t, modal_enc_dim)
        )
        # time encoding
        self.time_encoding = nn.Parameter(torch.randn(1, num_stack, 1, time_enc_dim))
        # class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim + modal_enc_dim + time_enc_dim))

        self.in_linear = nn.Linear(dim + modal_enc_dim + time_enc_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)

        self.blocks = nn.ModuleList([
            Block(
                dim=dim, heads=heads, mlp_dim=mlp_dim, qkv_bias=qkv_bias,
                drop=dropout, attn_drop=dropout, drop_path=drop_path)
            for _ in range(depth)])

        self.to_latent = nn.Identity()

        self.action_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.Linear(mlp_dim, num_classes)
        )
        self.xyz_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.Linear(mlp_dim, 6)
        )

    def forward(self, inputs):
        '''
        Args:
            cam_fixed_framestack, cam_gripper_framestack, tactile_framestack, audio_clip_g, audio_clip_h
            vf_inp: [batch, num_frames, 3, H, W]
            vg_inp: [batch, num_frames, 3, H, W]
            t_inp: [batch, num_frames, 3, H, W]
            a_inp: [batch, audio_channels, T]
        
        '''
        vf_inp, vg_inp, t_inp, audio_g, audio_h = inputs
        batch_size, num_frames, _, Hv, Wv = vf_inp.shape

        embeds = []
        # visual
        if "vf" in self.modalities:
            vf_patch = self.to_patch_embedding_v(vf_inp) # [batch, num_frames, num_patches_v, dim]
            vf_patch += self.pos_embed_vf
            vf_modal_enc = repeat(self.modal_enc_vf, '1 1 n d -> b l n d', b=batch_size, l=num_frames) # [batch, num_frames, num_patches_v, modal_enc_dim]
            vf_embed = torch.cat([vf_patch, vf_modal_enc], dim=-1) # [batch, num_frames, num_patches_v, dim + modal_enc_dim]
            embeds.append(vf_embed)
        if "vg" in self.modalities:
            vg_patch = self.to_patch_embedding_v(vg_inp)
            vg_patch += self.pos_embed_vg
            vg_modal_enc = repeat(self.modal_enc_vg, '1 1 n d -> b l n d', b=batch_size, l=num_frames)
            vg_embed = torch.cat([vg_patch, vg_modal_enc], dim=-1)
            embeds.append(vg_embed)
        # tactile
        if "t" in self.modalities:
            t_patch = self.to_patch_embedding_t(t_inp)
            t_patch += self.pos_embed_t
            t_modal_enc = repeat(self.modal_enc_t, '1 1 n d -> b l n d', b=batch_size, l=num_frames)
            t_embed = torch.cat([t_patch, t_modal_enc], dim=-1)
            embeds.append(t_embed)
        # audio
        a_patches_to_keep = num_frames * self.num_patches_a
        if "ah" in self.modalities:
            ah_patch = self.to_patch_embedding_a(audio_h) # [batch, T // prod(strides), dim]
            ah_patch = ah_patch[:, -a_patches_to_keep:, :].view(batch_size, num_frames, self.num_patches_a, ah_patch.size(-1))
            ah_patch += self.pos_embed_ah
            ah_modal_enc = repeat(self.modal_enc_ah, '1 1 n d -> b l n d', b=batch_size, l=num_frames)
            ah_embed = torch.cat([ah_patch, ah_modal_enc], dim=-1)
            embeds.append(ah_embed)
        if "ag" in self.modalities:
            ag_patch = self.to_patch_embedding_a(audio_g)
            ag_patch = ag_patch[:, -a_patches_to_keep:, :].view(batch_size, num_frames, self.num_patches_a, ag_patch.size(-1))
            ag_patch += self.pos_embed_ag
            ag_modal_enc = repeat(self.modal_enc_ag, '1 1 n d -> b l n d', b=batch_size, l=num_frames)
            ag_embed = torch.cat([ag_patch, ag_modal_enc], dim=-1)
            embeds.append(ag_embed)
        embeds = torch.cat(embeds, dim=2) # [batch, num_frames, total_patches, dim + modal_enc_dim]
        total_patches = embeds.size(2) # total number of patches for one frameskip
        time_enc = repeat(self.time_encoding, '1 l 1 d -> b l n d', b=batch_size, n=total_patches) # [batch, num_frames, total_patches, time_enc_dim]
        embeds = torch.cat([embeds, time_enc], dim=-1) # [batch, num_frames, total_patches, dim + modal_enc_dim + time_enc_dim]
        embeds = embeds.view(batch_size, num_frames * total_patches, -1) # [batch, num_frames * total_patches, dim + modal_enc_dim + time_enc_dim]
        

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=batch_size) # [batch, 1, dim + modal_enc_dim + time_enc_dim]
        x = torch.cat((cls_tokens, embeds), dim=1)
        x = self.in_linear(x) # [batch, 1 + num_frames * total_patches, dim]
        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x, num_frames)

        x =  x[:, 0]

        x = self.to_latent(x)
        return self.action_head(x), self.xyz_head(x)

if __name__ == "__main__":
    '''
    Base model from the paper: Hidden size=768, mlp_size=3072, heads=12, depth=16
    '''
    model = MuT(image_size=(224, 224), tactile_size=(96, 96), patch_size=16, num_stack=10, frameskip=5, fps=10, num_classes=27, dim=20, depth=5, heads=8, qkv_bias=False, mlp_dim=20, modal_enc_dim=100, time_enc_dim=100, ablation="vf_t_ah").cuda()
    vf_inp = torch.zeros((4, 10, 3, 224, 224)).float().cuda()
    vg_inp = None
    t_inp = torch.zeros((4, 10, 3, 96, 96)).float().cuda()
    audio_g = None
    audio_h = torch.zeros((4, 2, 16000 * 100)).float().cuda()
    inputs = (vf_inp, vg_inp, t_inp, audio_g, audio_h)
    a, b = model(inputs)
    print(a.shape, b.shape)