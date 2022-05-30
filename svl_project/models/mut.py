
from turtle import forward
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, heads, mlp_ratio, qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0.1, norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
           dim, num_heads=heads, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop, proj_drop=drop)

        ## Temporal Attention Parameters
        self.temporal_norm1 = norm_layer(dim)
        self.temporal_attn = Attention(
            dim, num_heads=heads, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop, proj_drop=drop)
        self.temporal_fc = nn.Linear(dim, dim)

        ## drop path
        self.drop_path = nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(dim=dim, hidden_dim=mlp_hidden_dim, dropout=drop)


    def forward(self, x, T):
        '''
        x: [batch, 1 + num_frames * total_patches, dim] or [b 1+(t p) m]
        B: batch
        T: num_frames
        P: total_patches(num_patch per frame)
        '''
        B = x.size(0)
        P = (x.size(1) - 1) // T

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
    def __init__(self, input_channels, last_layer_stride, encoding_dim=512, strides=[5,2,2,2,2,2,-1], kernel_widths=[10,3,3,3,3,2,2]):
        super().__init__()
        sr = 16000
        strides[-1]=last_layer_stride
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

class TimeEncoding(nn.Module):

    def __init__(self, d_model: int, num_stack, frameskip, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.offset = num_stack * frameskip
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, 1, d_model)
        pe[0, :, 0, 0::2] = torch.sin(position * div_term)
        pe[0, :, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, start):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, total_patches, dim]
        """
        for i in range(x.size(0)):
            x[i] = x[i] + self.pe[0, start[i] + self.offset: start[i] + x.size(1) + self.offset]
        return self.dropout(x)

class MuT(nn.Module):
    def __init__(self, *, image_size, tactile_size, patch_size, num_stack, frameskip, fps, last_layer_stride, num_classes, dim, depth, qkv_bias, heads, mlp_ratio, ablation, channels, audio_channels, drop_rate = 0., attn_drop_rate = 0., drop_path_rate=0.1):
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
        self.audio_encoder = Audio_Encoder(audio_channels, last_layer_stride)
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
        print("# of audio patches", num_patches_a, "; # of vision patches", num_patches_v, "; # of tactile patches", num_patches_t)
        # within_image positional encoding
        self.pos_embed_v = nn.Parameter(torch.randn(1, 1, num_patches_v, dim))
        self.pos_embed_a = nn.Parameter(torch.randn(1, 1, num_patches_a, dim))
        self.pos_embed_t = nn.Parameter(torch.randn(1, 1, num_patches_t, dim))
        # stream specific encodings
        self.modal_enc_vf = nn.Parameter(torch.randn(1, 1, 1, dim))
        self.modal_enc_vg = nn.Parameter(torch.randn(1, 1, 1, dim))
        self.modal_enc_ah = nn.Parameter(torch.randn(1, 1, 1, dim))
        self.modal_enc_ag = nn.Parameter(torch.randn(1, 1, 1, dim))
        self.modal_enc_t = nn.Parameter(torch.randn(1, 1, 1, dim)
        )
        # time encoding
        self.time_embed = TimeEncoding(dim, num_stack, frameskip, dropout=drop_rate, max_len=1000)
        # class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=dim, heads=heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
            for i in range(depth)])

        self.to_latent = nn.Identity()

        mlp_hidden_dim = mlp_ratio * dim
        self.action_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_hidden_dim),
            nn.Linear(mlp_hidden_dim, num_classes)
        )
        self.xyz_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_hidden_dim),
            nn.Linear(mlp_hidden_dim, 6)
        )

    def forward(self, inputs, start):
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
            vf_patch += self.pos_embed_v + self.modal_enc_vf
            embeds.append(vf_patch)
        if "vg" in self.modalities:
            vg_patch = self.to_patch_embedding_v(vg_inp)
            vg_patch += self.pos_embed_v + self.modal_enc_vg
            embeds.append(vg_patch)
        # tactile
        if "t" in self.modalities:
            t_patch = self.to_patch_embedding_t(t_inp)
            t_patch += self.pos_embed_t + self.modal_enc_t
            embeds.append(t_patch)
        # audio
        a_patches_to_keep = num_frames * self.num_patches_a
        if "ah" in self.modalities:
            ah_patch = self.to_patch_embedding_a(audio_h) # [batch, T // prod(strides), dim]
            ah_patch = ah_patch[:, -a_patches_to_keep:, :].view(batch_size, num_frames, self.num_patches_a, ah_patch.size(-1))
            ah_patch += self.pos_embed_a + self.modal_enc_ah
            embeds.append(ah_patch)
        if "ag" in self.modalities:
            ag_patch = self.to_patch_embedding_a(audio_g)
            ag_patch = ag_patch[:, -a_patches_to_keep:, :].view(batch_size, num_frames, self.num_patches_a, ag_patch.size(-1))
            ag_patch += self.pos_embed_a + self.modal_enc_ag
            embeds.append(ag_patch)
        embeds = torch.cat(embeds, dim=2) # [batch, num_frames, total_patches, dim + modal_enc_dim]
        total_patches = embeds.size(2) # total number of patches for one frameskip
        embeds = self.time_embed(embeds, start)
        embeds = embeds.view(batch_size, num_frames * total_patches, -1) # [batch, num_frames * total_patches, dim]
        

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=batch_size) # [batch, 1, dim]
        x = torch.cat((cls_tokens, embeds), dim=1)

        print(torch.any(torch.isnan(x)))
        for blk in self.blocks:
            x = blk(x, num_frames)

        x =  x[:, 0]

        x = self.to_latent(x)
        return self.action_head(x), self.xyz_head(x), None

if __name__ == "__main__":
    '''
    Base model from the paper: Hidden size=768, mlp_size=3072, heads=12, depth=16
    '''
    import configargparse
    p = configargparse.ArgParser()
    import time
    p.add("-c", "--config", is_config_file=True, default="conf/imi/transformer.yaml")
    p.add("--batch_size", default=32)
    p.add("--lr", default=1e-4, type=float)
    p.add("--gamma", default=0.9, type=float)
    p.add("--period", default=3)
    p.add("--epochs", default=65, type=int)
    p.add("--resume", default=None)
    p.add("--num_workers", default=8, type=int)
    # imi_stuff
    p.add("--exp_name", required=True, type=str)
    p.add("--action_dim", default=3, type=int)
    p.add("--num_stack", required=True, type=int)
    p.add("--frameskip", required=True, type=int)
    p.add("--use_mha", default=False, action="store_true")
    # data
    p.add("--train_csv", default="train.csv")
    p.add("--val_csv", default="val.csv")
    p.add("--data_folder", default="data/data_0502/test_recordings")
    p.add("--resized_height_v", required=True, type=int)
    p.add("--resized_width_v", required=True, type=int)
    p.add("--resized_height_t", required=True, type=int)
    p.add("--resized_width_t", required=True, type=int)
    p.add("--patch_size", default=16, type=int)
    p.add("--dim", default=768, type=int)
    p.add("--depth", default=12, type=int)
    p.add("--heads", default=12, type=int)
    p.add("--mlp_ratio", default=4, type=int)
    p.add("--qkv_bias", action="store_false", default=True)
    p.add("--last_layer_stride", default=1, type=int)

    p.add("--num_episode", default=None, type=int)
    p.add("--crop_percent", required=True, type=float)
    p.add("--ablation", required=True)
    p.add("--use_flow", default=False, action="store_true")
    p.add("--use_holebase", default=False, action="store_true")
    p.add("--task", type=str)
    p.add("--norm_audio", default=False, action="store_true")
    p.add("--aux_multiplier", type=float)
    args = p.parse_args()
    model = MuT(image_size=(args.resized_height_v, args.resized_width_t), tactile_size=(args.resized_height_t, args.resized_width_t), patch_size=args.patch_size, num_stack=args.num_stack, frameskip=args.frameskip, fps=10, last_layer_stride=args.last_layer_stride, num_classes=3 ** args.action_dim, dim=args.dim, depth=args.depth, qkv_bias=args.qkv_bias, heads=args.heads, mlp_ratio=args.mlp_ratio, ablation=args.ablation, channels=3, audio_channels=2).cuda()
    vf_inp = torch.zeros((3, 30, 3, 128, 96)).float().cuda()
    vg_inp = None
    t_inp = torch.zeros((3, 30, 3, 128, 96)).float().cuda()
    audio_h = torch.zeros((3, 2, 16000 * 150)).float().cuda()
    audio_g = None
    start = -3
    inputs = (vf_inp, vg_inp, t_inp, audio_g, audio_h)
    start = time.time()
    a, b = model(inputs, -150)
    print(time.time() - start)
    print(a.shape, b.shape)