import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
import numpy as np
import os

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math_utils import wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.gs_utils import *
from .go2_ts_config import Go2TSCfg
from collections import deque


class Go2TS(LeggedRobot):
    def get_observations(self):
        return self.obs_buf, self.privileged_obs_buf, self.obs_history

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(
            actions, -clip_actions, clip_actions).to(self.device)
        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
            self.action_queue[:, 0] = self.actions.clone()
            self.actions = self.action_queue[torch.arange(
                self.num_envs), self.action_delay].clone()
        # use self-implemented pd controller
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions)
            if self.num_build_envs == 0:
                torques = self.torques.squeeze()
                self.robot.control_dofs_force(torques, self.motors_dof_idx)
            else:
                self.robot.control_dofs_force(
                    self.torques, self.motors_dof_idx)
            self.scene.step()
            self.dof_pos[:] = self.robot.get_dofs_position(
                self.motors_dof_idx)
            self.dof_vel[:] = self.robot.get_dofs_velocity(
                self.motors_dof_idx)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(
                self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.obs_history, \
            self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, obs_history, _, _, _ = self.step(torch.zeros(
            self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs, obs_history

    def compute_observations(self):
        self.last_obs_buf = self.obs_buf.clone().detach()
        self.obs_buf = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,                   # 3
            self.projected_gravity,                                         # 3
            self.commands[:, :3] * self.commands_scale,                     # 3
            (self.dof_pos - self.default_dof_pos) *
            self.obs_scales.dof_pos,  # num_dofs
            self.dof_vel * self.obs_scales.dof_vel,                         # num_dofs
            self.actions                                                    # num_actions
        ), dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.base_pos[:, 2].unsqueeze(
                1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) -
                             1) * self.noise_scale_vec

        # push last_obs_buf to obs_history
        self.obs_history_deque.append(self.last_obs_buf)
        self.obs_history = torch.cat(
            [self.obs_history_deque[i]
                for i in range(self.obs_history_deque.maxlen)],
            dim=-1,
        )

        # privileged observations
        if self.cfg.domain_rand.randomize_ctrl_delay:
            # normalize to [0, 1]
            ctrl_delay = (self.action_delay /
                          self.cfg.domain_rand.ctrl_delay_step_range[1]).unsqueeze(1)

        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.cat(
                (
                    self.base_lin_vel * self.obs_scales.lin_vel,  # 3
                    (self._friction_values - 
                     self.friction_value_offset), # 1
                    self._added_base_mass,        # 1
                    self._base_com_bias,          # 3
                    self._rand_push_vels[:, :2],  # 2
                    (self._kp_scale - 
                     self.kp_scale_offset),       # num_actions
                    (self._kd_scale - 
                     self.kd_scale_offset),       # num_actions
                    self._joint_armature,         # 1
                    self._joint_stiffness,        # 1
                    self._joint_damping,          # 1
                ),
                dim=-1,
            )

    def _init_buffers(self):
        super()._init_buffers()
        # obs_history
        self.last_obs_buf = torch.zeros(
            (self.num_envs, self.cfg.env.num_observations),
            dtype=torch.float,
            device=self.device,
        )
        self.obs_history_deque = deque(maxlen=self.cfg.env.frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history_deque.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.num_observations,
                    dtype=torch.float,
                    device=self.device,
                )
            )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # clear obs history for the envs that are reset
        for i in range(self.obs_history_deque.maxlen):
            self.obs_history_deque[i][env_ids] *= 0
        # resample domain randomization parameters
        self._episodic_domain_randomization(env_ids)

    def _compute_torques(self, actions):
        # control_type = 'P'
        actions_scaled = actions * self.cfg.control.action_scale
        torques = (
            self._kp_scale * self.p_gains *
            (actions_scaled + self.default_dof_pos - self.dof_pos)
            - self._kd_scale * self.d_gains * self.dof_vel
        )
        return torques

    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.num_history_obs = self.cfg.env.num_history_obs
        self.num_latent_dims = self.cfg.env.num_latent_dims
        # determine privileged observation offset to normalize privileged observations
        self.friction_value_offset = (self.cfg.domain_rand.friction_range[0] + 
                                      self.cfg.domain_rand.friction_range[1]) / 2  # mean value
        self.kp_scale_offset = (self.cfg.domain_rand.kp_range[0] +
                                self.cfg.domain_rand.kp_range[1]) / 2  # mean value
        self.kd_scale_offset = (self.cfg.domain_rand.kd_range[0] +
                                self.cfg.domain_rand.kd_range[1]) / 2  # mean value

    def _init_domain_params(self):
        super()._init_domain_params()
        self._kp_scale = torch.ones(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self._kd_scale = torch.ones(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device)

    def _episodic_domain_randomization(self, env_ids):
        """ Update scale of Kp, Kd, rfi lim"""
        if len(env_ids) == 0:
            return

        if self.cfg.domain_rand.randomize_pd_gain:

            self._kp_scale[env_ids] = gs_rand_float(
                self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (len(env_ids), self.num_actions), device=self.device)
            self._kd_scale[env_ids] = gs_rand_float(
                self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (len(env_ids), self.num_actions), device=self.device)
