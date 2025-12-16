import torch.nn as nn
import torch
from .actor_critic import get_activation
from torch.distributions import Normal
from torch.nn import functional as F

class VAE(nn.Module):
    """Variational Auto-Encoder with MLP encoder and decoder."""
    def __init__(self, 
                 num_history_input,
                 num_latent_dims,
                 num_explicit_dims,
                 num_decoder_output,
                 activation = 'elu',
                 encoder_hidden_dims = [256, 128],
                 decoder_hidden_dims = [256, 128],):
        super(VAE, self).__init__()
        self.num_history_input = num_history_input
        self.num_latent_dims = num_latent_dims
        self.num_explicit_dims = num_explicit_dims
        
        activation = get_activation(activation)

        # MLP Encoder
        encoder_layers = []
        encoder_layers.append(
            nn.Linear(num_history_input, encoder_hidden_dims[0]))
        encoder_layers.append(activation)
        for l in range(len(encoder_hidden_dims)):
            if l == len(encoder_hidden_dims) - 1:
                encoder_layers.append(
                    nn.Linear(encoder_hidden_dims[l], num_latent_dims*2 + num_explicit_dims*2))  # output latent + base_lin_vel(estimated)
                encoder_layers.append(activation)
            else:
                encoder_layers.append(
                    nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l + 1]))
                encoder_layers.append(activation)
        self.encoder = nn.Sequential(*encoder_layers)

        self.latent_mu = nn.Linear(num_latent_dims * 2 + num_explicit_dims * 2, num_latent_dims)
        self.latent_var = nn.Sequential(
            nn.Linear(num_latent_dims * 2 + num_explicit_dims * 2, num_latent_dims),
            nn.Hardtanh(min_val=-5., max_val=5.) # to avoid numerical issues
            )

        self.vel_mu = nn.Linear(num_latent_dims * 2 + num_explicit_dims * 2, num_explicit_dims)
        self.vel_var = nn.Sequential(
            nn.Linear(num_latent_dims * 2 + num_explicit_dims * 2, num_explicit_dims),
            nn.Hardtanh(min_val=-5., max_val=5.) # to avoid numerical issues
            )

        # MLP Decoder
        decoder_layers = []
        decoder_input_dim = num_latent_dims + num_explicit_dims
        decoder_layers.append(nn.Linear(decoder_input_dim, decoder_hidden_dims[0]))
        decoder_layers.append(activation)
        for l in range(len(decoder_hidden_dims)):
            if l == len(decoder_hidden_dims) - 1:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[l], num_decoder_output))
            else:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l + 1]))
                decoder_layers.append(activation)
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self,obs_history):
        encoded = self.encoder(obs_history)
        latent_mu = self.latent_mu(encoded)
        latent_var = self.latent_var(encoded)
        vel_mu = self.vel_mu(encoded)
        vel_var = self.vel_var(encoded)
        return latent_mu, latent_var, vel_mu, vel_var

    def decode(self,z,v):
        decoder_in = torch.cat([z,v], dim = 1)
        output = self.decoder(decoder_in)
        return output

    def forward(self,obs_history):
        latent_mu, latent_var, vel_mu, vel_var = self.encode(obs_history)
        z = self.reparameterize(latent_mu, latent_var)
        vel = self.reparameterize(vel_mu, vel_var)
        return (z,vel), (latent_mu, latent_var, vel_mu, vel_var)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def sample(self,obs_history):
        sampled_out, distribution_params = self.forward(obs_history)
        return sampled_out, distribution_params

    def inference(self, obs_history):
        _, distribution_params = self.forward(obs_history)
        latent_mu, latent_var, vel_mu, vel_var = distribution_params
        return torch.cat((latent_mu, vel_mu), dim=-1)
