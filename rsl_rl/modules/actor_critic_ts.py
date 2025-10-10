import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from .actor_critic import get_activation

'''
Actor-Critic for Teacher-Student architecture.
'''

class ActorCriticTS(nn.Module):
    is_recurrent = False

    def __init__(self,  
                 num_actor_obs,
                 num_actions,
                 num_privilege_encoder_input,
                 num_history_encoder_input,
                 num_latent_dims,
                 num_critic_obs,
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 privilege_encoder_hidden_dims=[256, 128],
                 history_encoder_type="MLP", # "MLP" or "TCN"
                 history_encoder_hidden_dims=[256, 128],
                 history_encoder_channel_dims=[30, 30, 30, 30, 30, 30], # for TCN
                 history_encoder_dilation=[1, 1, 2, 1, 4, 1], # for TCN
                 history_encoder_stride=[1, 2, 1, 2, 1, 2], # for TCN
                 history_encoder_final_layer_dim=128, # for TCN
                 kernel_size=5,
                 activation='elu',
                 init_noise_std=1.0,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " +
                  str([key for key in kwargs.keys()]))
        super(ActorCriticTS, self).__init__()

        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs + num_latent_dims
        # input of the critic is the concatenation of actor observation and the latent from the privilege encoder
        mlp_input_dim_c = num_critic_obs

        # Privilege encoder
        privilege_encoder_layers = []
        privilege_encoder_layers.append(
            nn.Linear(num_privilege_encoder_input, privilege_encoder_hidden_dims[0]))
        privilege_encoder_layers.append(activation)
        for l in range(len(privilege_encoder_hidden_dims)):
            if l == len(privilege_encoder_hidden_dims) - 1:
                privilege_encoder_layers.append(
                    nn.Linear(privilege_encoder_hidden_dims[l], num_latent_dims))
            else:
                privilege_encoder_layers.append(nn.Linear(
                    privilege_encoder_hidden_dims[l], privilege_encoder_hidden_dims[l + 1]))
                privilege_encoder_layers.append(activation)
        self.privilege_encoder = nn.Sequential(*privilege_encoder_layers)

        # History encoder
        self.history_encoder_type = history_encoder_type
        history_encoder_layers = []
        if history_encoder_type == "MLP":
            history_encoder_layers.append(
                nn.Linear(num_history_encoder_input, history_encoder_hidden_dims[0]))
            history_encoder_layers.append(activation)
            for l in range(len(history_encoder_hidden_dims)):
                if l == len(history_encoder_hidden_dims) - 1:
                    history_encoder_layers.append(
                        nn.Linear(history_encoder_hidden_dims[l], num_latent_dims))
                else:
                    history_encoder_layers.append(
                        nn.Linear(history_encoder_hidden_dims[l], history_encoder_hidden_dims[l + 1]))
                    history_encoder_layers.append(activation)
            self.history_encoder = nn.Sequential(*history_encoder_layers)
        elif history_encoder_type == "TCN":
            in_channels = 1
            for l in range(len(history_encoder_channel_dims)):
                out_channels = history_encoder_channel_dims[l]
                padding = history_encoder_dilation[l]*(kernel_size-1)// 2
                history_encoder_layers.append(
                    nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=history_encoder_stride[l],
                              padding=padding,
                              dilation=history_encoder_dilation[l])
                )
                in_channels = out_channels
                num_history_encoder_input = (num_history_encoder_input - 1) // history_encoder_stride[l] + 1
            history_encoder_output_layer = nn.Linear(
                num_history_encoder_input * history_encoder_channel_dims[-1], history_encoder_final_layer_dim)
            history_encoder_output_activation = activation
            history_encoder_latent_layer = nn.Linear(
                history_encoder_final_layer_dim, num_latent_dims)
            history_encoder_layers.append(nn.Flatten())
            history_encoder_layers.append(history_encoder_output_layer)
            history_encoder_layers.append(history_encoder_output_activation)
            history_encoder_layers.append(history_encoder_latent_layer)
            self.history_encoder = nn.Sequential(*history_encoder_layers)

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(
                    nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(
                    nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(
                    nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Privilege Encoder MLP: {self.privilege_encoder}")
        print(f"History Encoder: {self.history_encoder}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, privilege_observations):
        latent = self.privilege_encoder(privilege_observations)
        mean = self.actor(torch.cat(
            (
            observations, latent
            ), dim=-1))
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, privilege_observations, **kwargs):
        self.update_distribution(observations, privilege_observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_teacher(self, observations, privilege_observations, **kwargs):
        latent = self.privilege_encoder(privilege_observations)
        actions_mean = self.actor(torch.cat(
            (
            observations, latent
            ), dim=-1))
        return actions_mean

    def act_student(self, observations, observation_history, **kwargs):
        if self.history_encoder_type == "TCN":
            # input shape (batch_size, obs_history_len) -> (batch_size, 1, obs_history_len)
            observation_history = observation_history.unsqueeze(1)
        latent = self.history_encoder(observation_history)
        actions_mean = self.actor(torch.cat(
            (
            observations, latent
            ), dim=-1))
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
