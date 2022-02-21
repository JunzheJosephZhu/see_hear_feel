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
        self.encoder = encoder
        self.decoder = decoder
        self.mean_layer = nn.Linear(out_dim, latent_dim)
        self.scale_layer = nn.Linear(out_dim, latent_dim)
        self.decoder_initial = nn.Linear(latent_dim, out_dim)

    def forward(self, inp):
        shared_repr = self.encoder(inp)
        mean = self.mean_layer(shared_repr)
        scale = torch.exp(self.scale_layer(shared_repr))
        z_dist = independent_multivariate_normal(mean=mean,
                                               stddev=scale)
        z = self.get_vector(z_dist)
        decoder_inp = self.decoder_initial(z)
        pixels = self.decoder(decoder_inp)
        return pixels, z_dist