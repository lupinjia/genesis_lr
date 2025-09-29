from legged_gym import *

import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math_utils import wrap_to_pi, quat_apply, torch_rand_float
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.gs_utils import *
from collections import deque

class Go2EE(LeggedRobot):
    def get_observations(self):
        return self.estimator_features_buf, self.estimator_labels_buf, self.privileged_obs_buf

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        actions = torch.clip(
            actions, -clip_actions, clip_actions).to(self.device)
        self.actions[:] = actions[:]
        self.simulator.step(actions)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.estimator_features_buf = torch.clip(self.estimator_features_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(
                self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.estimator_features_buf, self.estimator_labels_buf, self.privileged_obs_buf, \
            self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        estimator_features, estimator_labels, privileged_obs, _, _, _ = self.step(torch.zeros(
            self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return estimator_features, estimator_labels, privileged_obs

    def compute_observations(self):
        obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,                     # 3
            self.simulator.projected_gravity,                                         # 3
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,                   # 3
            (self.simulator.dof_pos - self.simulator.default_dof_pos) *
            self.obs_scales.dof_pos,  # num_dofs
            self.simulator.dof_vel * self.obs_scales.dof_vel,                         # num_dofs
            self.actions                                                    # num_actions
        ), dim=-1)
        
        domain_randomization_info = torch.cat((
                    (self.simulator._friction_values - 
                    self.friction_value_offset),            # 1
                    self.simulator._added_base_mass,        # 1
                    self.simulator._base_com_bias,          # 3
                    self.simulator._rand_push_vels[:, :2],  # 2
                    (self.simulator._kp_scale - 
                     self.kp_scale_offset),                 # num_actions
                    (self.simulator._kd_scale - 
                     self.kd_scale_offset),                 # num_actions
                    self.simulator._joint_armature,         # 1
                    self.simulator._joint_stiffness,        # 1
                    self.simulator._joint_damping,          # 1
            ), dim=-1)
        
        # Critic observation
        single_critic_obs = torch.cat((
            obs_buf,                 # num_observations
            domain_randomization_info,    # 34
        ), dim=-1)
        if self.cfg.asset.obtain_link_contact_states:
            single_critic_obs = torch.cat(
                (single_critic_obs, self.simulator.link_contact_states,  # contact states of thighs, calfs and feet (4+4+4)=12
                    ),
                dim=-1,
            )
        if self.cfg.terrain.measure_heights: # 81
            heights = torch.clip(self.simulator.base_pos[:, 2].unsqueeze(
                1) - 0.5 - self.simulator.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            single_critic_obs = torch.cat((single_critic_obs, heights), dim=-1)
        self.critic_obs_deque.append(single_critic_obs)
        self.privileged_obs_buf = torch.cat(
            [self.critic_obs_deque[i]
                for i in range(self.critic_obs_deque.maxlen)],
            dim=-1,
        )
        
        # add noise if needed
        if self.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) -
                             1) * self.noise_scale_vec

        # push obs_buf to obs_history
        self.obs_history_deque.append(obs_buf)
        self.estimator_features_buf = torch.cat(
            [self.obs_history_deque[i]
                for i in range(self.obs_history_deque.maxlen)],
            dim=-1,
        )
        
        # Estimator labels
        self.estimator_labels_buf = torch.cat((
            self.simulator.base_lin_vel * self.obs_scales.lin_vel,         # 3
            1.0 * (torch.norm(
                self.simulator.link_contact_forces[:, self.simulator.feet_indices, :], dim=-1) 
                   > 1.0),  # 4
            (self.simulator.feet_pos[:, :, 2] -
            torch.mean(self.simulator.height_around_feet, dim=-1) -
            self.cfg.rewards.foot_height_offset),  # 4
        ), dim=-1)

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
                    self.cfg.env.single_critic_obs_len,
                    dtype=torch.float,
                    device=self.device,
                )
            )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # clear obs history for the envs that are reset
        for i in range(self.obs_history_deque.maxlen):
            self.obs_history_deque[i][env_ids] *= 0
        for i in range(self.critic_obs_deque.maxlen):
            self.critic_obs_deque[i][env_ids] *= 0
    
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

    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.num_estimator_features = cfg.env.num_estimator_features
        self.num_estimator_labels = cfg.env.num_estimator_labels
        # determine privileged observation offset to normalize privileged observations
        self.friction_value_offset = (self.cfg.domain_rand.friction_range[0] + 
                                      self.cfg.domain_rand.friction_range[1]) / 2  # mean value
        self.kp_scale_offset = (self.cfg.domain_rand.kp_range[0] +
                                self.cfg.domain_rand.kp_range[1]) / 2  # mean value
        self.kd_scale_offset = (self.cfg.domain_rand.kd_range[0] +
                                self.cfg.domain_rand.kd_range[1]) / 2  # mean value

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
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.llast_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.simulator.last_dof_vel[:] = self.simulator.dof_vel[:]
        
        if self.debug:
            self.simulator.draw_debug_vis()
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.simulator.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(
                0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0)

        if self.cfg.terrain.measure_heights:
            self.simulator.get_heights()
            if self.cfg.terrain.obtain_terrain_info_around_feet:
                self.simulator.calc_terrain_info_around_feet()
        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self.simulator.push_robots()
    
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
        noise_vec[9:21] = noise_scales.dof_pos * \
            noise_level * self.obs_scales.dof_pos
        noise_vec[21:33] = noise_scales.dof_vel * \
            noise_level * self.obs_scales.dof_vel
        noise_vec[33:45] = 0.  # previous actions
        # if self.cfg.terrain.measure_heights:
        #     noise_vec[48:235] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
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
    
    def _reward_foot_clearance(self):
        """
        Encourage feet to be close to desired height while swinging
        
        Attention: using torch.max(self.simulator.height_around_feet) will cause reward value jumping, bad for learning
        """
        foot_vel_xy_norm = torch.norm(self.simulator.feet_vel[:, :, :2], dim=-1)
        clearance_error = torch.sum(
            foot_vel_xy_norm * torch.square(
                self.simulator.feet_pos[:, :, 2] - torch.mean(self.simulator.height_around_feet, dim=-1) -
                self.cfg.rewards.foot_clearance_target -
                self.cfg.rewards.foot_height_offset
            ), dim=-1
        )
        return torch.exp(-clearance_error / self.cfg.rewards.foot_clearance_tracking_sigma)
    
    def _reward_hip_pos(self):
        """ Reward for the hip joint position close to default position
        """
        hip_joint_indices = [0, 3, 6, 9]
        dof_pos_error = torch.sum(torch.square(
            self.simulator.dof_pos[:, hip_joint_indices] - 
            self.simulator.default_dof_pos[:, hip_joint_indices]), dim=-1)
        return dof_pos_error