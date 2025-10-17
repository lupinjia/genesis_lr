# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.actor_critic_ts import ActorCriticTS
from .actor_critic import get_activation, init_orhtogonal, init_normal, init_constant, init_xavier
from .vae import VAE

'''
Actor-Critic for Hybrid Implicit-Explicit architecture using VAE
'''

class ActorCriticDreamWaQ(nn.Module):
    is_recurrent = False

    def __init__(self,  
                 num_actor_obs,
                 num_actions,
                 num_privileged_obs, 
                 num_history_input,
                 num_latent_dims,
                 num_explicit_dims,
                 num_decoder_output,
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 encoder_hidden_dims=[256, 128],
                 decoder_hidden_dims=[256, 128],
                 activation='elu',
                 init_noise_std=1.0,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " +
                  str([key for key in kwargs.keys()]))
        super(ActorCriticDreamWaQ, self).__init__()

        # Input dimension of actor (proprioceptive obs + base_lin_vel(estimated) + latent)
        mlp_input_dim_a = num_actor_obs + num_explicit_dims + num_latent_dims
        # Input dimension of actor (proprioceptive obs + base_lin_vel(true) + latent)
        mlp_input_dim_c = num_privileged_obs
        
        self.vae = VAE(num_history_input = num_history_input,
                       num_latent_dims=num_latent_dims,
                       num_explicit_dims=num_explicit_dims,
                       num_decoder_output=num_decoder_output,
                       activation=activation,
                       encoder_hidden_dims=encoder_hidden_dims,
                       decoder_hidden_dims=decoder_hidden_dims)
        
        activation = get_activation(activation)

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

        print(f"Encoder MLP: {self.vae.encoder}")
        print(f"Decoder MLP: {self.vae.decoder}")
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

    def update_distribution(self, observations, obs_history):
        """When inferring the actor, use the current observation and latent"""
        sampled_out, distribution_params = self.vae.sample(obs_history)
        z, vel = sampled_out
        latent_mu, latent_var, vel_mu, vel_var = distribution_params
        sampled_out = torch.cat((z, vel), dim=-1)
        mean = self.actor(torch.cat(
            (
            observations, sampled_out
            ), dim=-1))
        # # Check if NaN exists in mean
        # if torch.isnan(mean).any():
        #     print("NaN detected in action mean")
        #     import ipdb; ipdb.set_trace()
            
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, obs_history, **kwargs):
        self.update_distribution(observations, obs_history)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, observation_history, **kwargs):
        mean_out = self.vae.inference(observation_history)
        actions_mean = self.actor(torch.cat(
            (
            observations, mean_out
            ), dim=-1))
        return actions_mean

    def evaluate(self, critic_obs, **kwargs):
        value = self.critic(critic_obs)
        return value
