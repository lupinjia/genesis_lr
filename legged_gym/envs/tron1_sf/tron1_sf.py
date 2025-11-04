from legged_gym import *
from time import time
import numpy as np

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math_utils import torch_rand_float
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.gs_utils import *
from .tron1_sf_config import TRON1SFCfg
from collections import deque

class TRON1SF(LeggedRobot):
    def compute_observations(self):
        obs_buf = torch.cat((
                                self.commands[:, :3] * self.commands_scale,                   # 3
                                self.simulator.projected_gravity,                             # 3
                                self.simulator.base_ang_vel * self.obs_scales.ang_vel,        # 3
                                (self.simulator.dof_pos - self.simulator.default_dof_pos) 
                                    * self.obs_scales.dof_pos,                                # num_dofs
                                self.simulator.dof_vel * self.obs_scales.dof_vel,             # num_dofs
                                self.actions                                                  # num_actions
                                ), dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.simulator.base_pos[:, 2].unsqueeze(
                1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            obs_buf = torch.cat((obs_buf, heights), dim=-1)

        if self.num_privileged_obs is not None:
            privileged_obs_buf = torch.cat(
                (
                    self.simulator.base_lin_vel * self.obs_scales.lin_vel, # 3
                    obs_buf,
                    self.last_actions,                      # num_actions
                    (self.simulator._friction_values - 
                    self.friction_value_offset),            # 1
                    self.simulator._added_base_mass,        # 1
                    self.simulator._base_com_bias,          # 3
                    self.simulator._rand_push_vels[:, :2],  # 2
                    self.feet_air_time,                     # 2
                ),
                dim=-1,
            )
            # add perceptive inputs if not blind
            if self.cfg.terrain.measure_heights:
                heights = torch.clip(self.simulator.base_pos[:, 2].unsqueeze(
                    1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
                privileged_obs_buf = torch.cat((privileged_obs_buf, heights), dim=-1)
            
            self.critic_obs_deque.append(privileged_obs_buf)
            self.privileged_obs_buf = torch.cat(
                [self.critic_obs_deque[i]
                    for i in range(self.critic_obs_deque.maxlen)],
                dim=-1,
            )
                
        # add noise if needed
        if self.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - \
                             1) * self.noise_scale_vec
        
        # check if NaN exists in obs_buf or privileged_obs_buf
        if torch.isnan(obs_buf).any():
            print("NaN detected in obs_buf")
            import ipdb; ipdb.set_trace()
        
        self.obs_history_deque.append(obs_buf)
        # construct stacked observations
        self.obs_buf = torch.cat(
            [self.obs_history_deque[i]
                for i in range(self.obs_history_deque.maxlen)],
            dim=-1,
        )
        
    
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # clear obs history for the envs that are reset
        for i in range(self.obs_history_deque.maxlen):
            self.obs_history_deque[i][env_ids] *= 0
        for i in range(self.critic_obs_deque.maxlen):
            self.critic_obs_deque[i][env_ids] *= 0
    
    def _init_buffers(self):
        super()._init_buffers()
        # obs_history
        self.obs_history_deque = deque(maxlen=self.cfg.env.frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history_deque.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.num_single_obs,
                    dtype=torch.float,
                    device=self.device,
                )
            )
        # critic observation buffer
        self.critic_obs_deque = deque(maxlen=self.cfg.env.c_frame_stack)
        for _ in range(self.cfg.env.c_frame_stack):
            self.critic_obs_deque.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.num_single_privileged_obs,
                    dtype=torch.float,
                    device=self.device,
                )
            )
        # init stand joint angles
        self.init_stand_dof_pos = torch.zeros(
            1,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        for i in range(self.num_actions):
            name = self.cfg.asset.dof_names[i]
            stand_angle = self.cfg.init_state.init_stand_joint_angles[name]
            self.init_stand_dof_pos[:, i] = stand_angle
    
    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        indices = torch.randperm(n)

        half_size = n // 2
        half_indices = indices[:half_size]
        remaining_indices = indices[half_size:]
        dof_pos = torch.zeros((len(env_ids), self.num_actions), dtype=torch.float, 
                              device=self.device, requires_grad=False)
        dof_vel = torch.zeros((len(env_ids), self.num_actions), dtype=torch.float, 
                              device=self.device, requires_grad=False)
        dof_pos[half_indices] = self.simulator.default_dof_pos[:] + torch_rand_float(
            -0.3, 0.3, (len(half_indices), self.num_actions), device=self.device
        )
        dof_pos[remaining_indices] = self.init_stand_dof_pos[:] + torch_rand_float(
            -0.3, 0.3, (len(remaining_indices), self.num_actions), device=self.device
        )

        self.simulator.reset_dofs(env_ids, dof_pos, dof_vel)
    
    def _get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = 0.  # commands
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = noise_scales.ang_vel * \
            noise_level * self.obs_scales.ang_vel
        noise_vec[9:17] = noise_scales.dof_pos * \
            noise_level * self.obs_scales.dof_pos
        noise_vec[17:25] = noise_scales.dof_vel * \
            noise_level * self.obs_scales.dof_vel
        noise_vec[25:33] = 0.  # previous actions
        # if self.cfg.terrain.measure_heights:
        #     noise_vec[27:214] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        return noise_vec
    
    def _reward_feet_air_time(self):
        # Reward long steps
        contact = self.simulator.link_contact_forces[:, self.simulator.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.25) * first_contact, dim=1)  # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime