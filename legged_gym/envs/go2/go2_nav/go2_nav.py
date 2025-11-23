from legged_gym import *
from time import time
import numpy as np
import os

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot_nav import LeggedRobotNav
from legged_gym.utils.math_utils import torch_rand_float, quat_from_euler_xyz
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.gs_utils import *
from .go2_nav_config import GO2NavCfg

class GO2Nav(LeggedRobotNav):
    
    # Override functions for deployment
    def compute_observations(self):
        self.obs_buf = torch.cat((self.simulator.base_lin_vel * self.obs_scales.lin_vel,                    # 3
                                    self.simulator.base_ang_vel * self.obs_scales.ang_vel,                   # 3
                                    self.simulator.projected_gravity,                                         # 3
                                    self.commands * self.commands_scale,                                      # 5
                                    (self.simulator.dof_pos - self.simulator.default_dof_pos) 
                                      * self.obs_scales.dof_pos, # num_dofs
                                    self.simulator.dof_vel * self.obs_scales.dof_vel,                         # num_dofs
                                    self.actions                                                    # num_actions
                                    ), dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.simulator.base_pos[:, 2].unsqueeze(
                1) - 0.3 - self.simulator.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        
        # critic observations
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.cat(
                (
                    self.obs_buf,
                    self.last_actions,
                    self.simulator._friction_values,        # 1
                    self.simulator._added_base_mass,        # 1
                    self.simulator._base_com_bias,          # 3
                    self.simulator._rand_push_vels[:, :2],  # 2
                ),
                dim=-1,
            )

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - \
                             1) * self.noise_scale_vec
    
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self.simulator.draw_debug_vis() if needed
        """
        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.simulator.post_physics_step()
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        if self.cfg.sensor.add_depth:
            
            self.simulator.update_depth_images()
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.llast_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.simulator.last_dof_vel[:] = self.simulator.dof_vel[:]
        self.simulator.last_base_lin_vel[:] = self.simulator.base_lin_vel[:]
        self.simulator.last_base_ang_vel[:] = self.simulator.base_ang_vel[:]
        
        if self.debug:
            # self.simulator.draw_debug_vis()
            self.simulator.draw_debug_boxes(self.target_pos_world, quat_from_euler_xyz(torch.zeros_like(self.target_orientation_world), 
                                                                                       torch.zeros_like(self.target_orientation_world), 
                                                                                       self.target_orientation_world))

    def _init_buffers(self):
        super()._init_buffers()
        self.commands_scale = torch.tensor([self.obs_scales.base_pos, self.obs_scales.base_pos,
                                            self.obs_scales.base_orientation, self.obs_scales.time_to_target],
                                           device=self.device, dtype=torch.float,
                                           requires_grad=False)
    
    def _get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * \
            noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * \
            noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:13] = 0.  # commands
        noise_vec[13:25] = noise_scales.dof_pos * \
            noise_level * self.obs_scales.dof_pos
        noise_vec[25:37] = noise_scales.dof_vel * \
            noise_level * self.obs_scales.dof_vel
        noise_vec[37:49] = 0.  # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[49:236] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        return noise_vec
    
    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        
        dof_pos = torch.zeros((len(env_ids), self.num_actions), dtype=torch.float, 
                              device=self.device, requires_grad=False)
        dof_vel = torch.zeros((len(env_ids), self.num_actions), dtype=torch.float, 
                              device=self.device, requires_grad=False)
        dof_pos[:, [0, 3, 6, 9]] = self.simulator.default_dof_pos[:, [0, 3, 6, 9]] + \
            torch_rand_float(-0.2, 0.2, (len(env_ids), 4), self.device)
        dof_pos[:, [1, 4, 7, 10]] = self.simulator.default_dof_pos[:, [1, 4, 7, 10]] + \
            torch_rand_float(-0.4, 0.4, (len(env_ids), 4), self.device)
        dof_pos[:, [2, 5, 8, 11]] = self.simulator.default_dof_pos[:, [2, 5, 8, 11]] + \
            torch_rand_float(-0.4, 0.4, (len(env_ids), 4), self.device)

        self.simulator.reset_dofs(env_ids, dof_pos, dof_vel)
    
    def _reward_move_in_direction(self):
        if self.common_step_counter <= 24 * 200:
            # Reward moving in the commanded direction
            command_dir = self.commands[:, :2] / (torch.norm(self.commands[:, :2], dim=1, keepdim=True) + 1e-6)
            vel_xy = self.simulator.base_lin_vel[:, :2] / torch.norm(self.simulator.base_lin_vel[:, :2], dim=1, keepdim=True)
            # get cosine value between command direction and velocity
            cos_angle = torch.sum(command_dir * vel_xy, dim=1)
        else:
            cos_angle = torch.zeros(self.num_envs, device=self.device)
        return cos_angle