import torch
from torch import nn
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

def independent_multivariate_normal(mean, stddev):
    # Create a normal distribution, which by default will assume all dimensions but one are a batch dimension
    dist = torch.distributions.Normal(mean, stddev, validate_args=True)
    # Wrap the distribution in an Independent wrapper, which reclassifies all but one dimension as part of the actual
    # sample shape, but keeps variances defined only on the diagonal elements of what would be the MultivariateNormal
    multivariate_mimicking_dist = torch.distributions.Independent(dist, len(mean.shape) - 1)
    return multivariate_mimicking_dist

class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder, out_dim, latent_dim):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mean_layer = nn.Linear(out_dim, latent_dim)
        # log variance
        self.scale_layer = nn.Linear(out_dim, latent_dim)

    def forward(self, inp):
        shared_repr = self.encoder(inp)
        mean = self.mean_layer(shared_repr)
        scale = torch.exp(0.5 * self.scale_layer(shared_repr))
        z_dist = independent_multivariate_normal(mean=mean,
                                               stddev=scale)
        z = z_dist.rsample()
        pixels = self.decoder(z)
        return pixels, z_dist

class VAE_FuturePred(torch.nn.Module):
    def __init__(self, v_encoder, v_decoder, t_encoder, t_decoder, fusion_module, out_dim, latent_dim, action_dim):
        super().__init__()
        self.v_encoder = v_encoder
        self.t_encoder = t_encoder
        self.v_decoder = v_decoder
        self.t_decoder = t_decoder
        self.mean_layer = nn.Linear(out_dim, latent_dim)
        # log variance
        self.scale_layer = nn.Linear(out_dim, latent_dim)
        self.action_encoder = nn.Sequential(nn.Linear(action_dim, out_dim),
                                            nn.Sigmoid(),
                                            nn.Linear(out_dim, out_dim),
                                            nn.Sigmoid(),
                                            nn.Linear(out_dim, out_dim))
        self.future_model = nn.Sequential(nn.Linear(out_dim + 256, out_dim),
                                            nn.Sigmoid(),
                                            nn.Linear(out_dim, out_dim),
                                            nn.Sigmoid(),
                                            nn.Linear(out_dim, out_dim))
        self.fusion_module = fusion_module
        self.future_mean_layer = nn.Linear(out_dim, latent_dim)
        # log variance
        self.future_scale_layer = nn.Linear(out_dim, latent_dim)

    def common_steps(self, inp, encoder, decoder):
        shared_repr = encoder(inp)
        mean = self.mean_layer(shared_repr)
        scale = torch.exp(0.5 * self.scale_layer(shared_repr))
        z_dist = independent_multivariate_normal(mean=mean,
                                                 stddev=scale)
        z = z_dist.rsample()
        pixels = decoder(z)
        return shared_repr, pixels, z_dist, mean, scale

    def forward(self, inp_v, inp_t, action):
        repr_v, pixels_v, z_dist_v, mean_v, scale_v = self.common_steps(inp_v, self.v_encoder, self.v_decoder)
        repr_t, pixels_t, z_dist_t, mean_t, scale_t = self.common_steps(inp_t, self.t_encoder, self.t_decoder)
        shared_repr = self.fusion_module(torch.cat([repr_v, repr_t],-1))
        # predict future representation
        action_enc = self.action_encoder(action)
        future_repr = self.future_model(torch.cat([shared_repr, action_enc], -1))
        future_mean = self.future_mean_layer(future_repr)
        future_scale = torch.exp(0.5 * self.future_scale_layer(future_repr))
        future_z_dist = independent_multivariate_normal(mean=future_mean,
                                        stddev=future_scale)

        z_dist_v_t = independent_multivariate_normal(mean=mean_v + mean_t,
                                        stddev=scale_v + scale_t)

        return pixels_v, z_dist_v,  pixels_t, z_dist_t, z_dist_v_t, future_z_dist