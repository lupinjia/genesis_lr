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

import torch
import torch.nn as nn
import torch.optim as optim
import math

from rsl_rl.modules import ActorCriticDreamWaQ
from rsl_rl.storage import RolloutStorageDreamWaQ

'''
PPO for MLP-decoder version DreamWaQ
'''

class PPO_DreamWaQ:
    actor_critic: ActorCriticDreamWaQ

    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 encoder_lr=1e-3,      # learning rate for hybrid encoder
                 num_encoder_epochs=1, # number of epochs for hybrid encoder via supervised learning
                 vae_kld_weight=1.0,   # weight of KL divergence loss in VAE
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.encoder_lr = encoder_lr
        self.vae_kld_weight = vae_kld_weight

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.rl_parameters = list(self.actor_critic.actor.parameters()) + \
                             list(self.actor_critic.critic.parameters()) + \
                            [self.actor_critic.std]
        self.optimizer = optim.Adam(
            self.rl_parameters, lr=learning_rate)
        self.vae_optimizer = optim.Adam(
            self.actor_critic.vae.parameters(), lr=encoder_lr)
        self.transition = RolloutStorageDreamWaQ.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_encoder_epochs = num_encoder_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, 
                     privileged_obs_shape, obs_history_shape, explicit_info_shape, next_states_shape, action_shape):
        self.storage = RolloutStorageDreamWaQ(
            num_envs, num_transitions_per_env, actor_obs_shape, 
            privileged_obs_shape, obs_history_shape, explicit_info_shape, next_states_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, privileged_obs, obs_history, explicit_info_labels):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs, obs_history).detach()
        self.transition.values = self.actor_critic.evaluate(privileged_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.privileged_observations = privileged_obs
        self.transition.observation_histories = obs_history
        self.transition.explicit_info_labels = explicit_info_labels
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, next_state):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.next_states = next_state.clone()
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * \
                torch.squeeze(self.transition.values *
                              infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_privileged_obs):
        # 'last' here means 'final' in the rollout storage
        last_values = self.actor_critic.evaluate(last_privileged_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_explicit_estimation_loss = 0
        mean_reconstruction_loss = 0
        mean_kld_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, privileged_obs_batch, obs_histories_batch, explicit_info_labels_batch, next_state_batch, \
            terminated_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
                old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

            self.actor_critic.act(
                obs_batch, obs_histories_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                actions_batch)
            value_batch = self.actor_critic.evaluate(
                privileged_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.desired_kl != None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + 
                        (torch.square(old_sigma_batch) + 
                         torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(
                            1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(
                            1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch -
                              torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                               1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(
                    value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * \
                value_loss - self.entropy_coef * entropy_batch.mean()

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.rl_parameters, self.max_grad_norm)
            self.optimizer.step()
            
            # vae gradient step
            for _ in range(self.num_encoder_epochs):
                sampled_out, distribution_params = self.actor_critic.vae.forward(obs_histories_batch)
                z,v = sampled_out
                latent_mu, latent_var, _, _ = distribution_params
                reconstructed_out = self.actor_critic.vae.decode(z, v)

                explicit_estimation_loss = nn.functional.mse_loss(v * terminated_batch,
                                                             explicit_info_labels_batch * terminated_batch)
                # Ignore the explicit_estimation and reconstruction loss for terminated episodes
                reconstruction_loss = nn.functional.mse_loss(reconstructed_out * terminated_batch,
                                                             next_state_batch * terminated_batch)
                # KL Divergence loss of VAE
                kld_loss = -0.5 * torch.mean(torch.sum(1 + latent_var - latent_mu ** 2 - latent_var.exp(), dim = 1) * terminated_batch)
                vae_loss = explicit_estimation_loss + reconstruction_loss + self.vae_kld_weight * kld_loss
                self.vae_optimizer.zero_grad()
                vae_loss.backward()
                nn.utils.clip_grad_norm_(
                self.actor_critic.vae.parameters(), self.max_grad_norm)
                self.vae_optimizer.step()
                
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_explicit_estimation_loss += explicit_estimation_loss.item()
            mean_reconstruction_loss += reconstruction_loss.item()
            mean_kld_loss += kld_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_explicit_estimation_loss /= (num_updates * self.num_encoder_epochs)
        mean_reconstruction_loss /= (num_updates * self.num_encoder_epochs)
        mean_kld_loss /= (num_updates * self.num_encoder_epochs)
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_explicit_estimation_loss, \
            mean_reconstruction_loss, mean_kld_loss
