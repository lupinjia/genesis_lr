from legged_gym import *

import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math_utils import *
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.gs_utils import *
from collections import deque
from scipy.stats import vonmises

class TRON1PF_EE(LeggedRobot):
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
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
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
        # Periodic Reward Framework phi cycle
        # step after computing reward but before resetting the env
        self.gait_time += self.dt
        # +self.dt/2 in case of float precision errors
        is_over_limit = (self.gait_time >= (self.gait_period - self.dt / 2))
        over_limit_indices = is_over_limit.nonzero(as_tuple=False).flatten()
        self.gait_time[over_limit_indices] = 0.0
        self.phi = self.gait_time / self.gait_period
        
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self._calc_periodic_reward_obs()
        if self.cfg.sensor.add_depth:
            self.simulator.update_depth_images()
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.llast_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.simulator.last_dof_vel[:] = self.simulator.dof_vel[:]
        self.simulator.last_feet_vel[:] = self.simulator.feet_vel[:]
        
        if self.debug:
            self.simulator.draw_debug_vis()
            
    def compute_observations(self):
        obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,                     # 3
            self.simulator.projected_gravity,                               # 3
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,          # 3
            (self.simulator.dof_pos - self.simulator.default_dof_pos) *
            self.obs_scales.dof_pos,                                        # num_dofs
            self.simulator.dof_vel * self.obs_scales.dof_vel,               # num_dofs
            self.actions,                                                   # num_actions
            self.clock_input,                                               # 4
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
                    self.simulator._joint_armature,
                    self.simulator._joint_friction,
                    self.simulator._joint_damping,
            ), dim=-1)
        
        gait_info = torch.cat((
            self.exp_C_frc_left, 
            self.exp_C_frc_right,
        ), dim=-1)
        
        # Critic observation
        single_critic_obs = torch.cat((
            obs_buf,                 # num_observations
            domain_randomization_info,    # 22
            gait_info,                    # 2
        ), dim=-1)
        if self.cfg.asset.obtain_link_contact_states:
            single_critic_obs = torch.cat(
                (
                    single_critic_obs,                         # previous
                    self.simulator.link_contact_states,  # contact states of abad, hip, knee and foot (2+2+2+2)=8
                ),
                dim=-1,
            )
        if self.cfg.terrain.measure_heights: # 81
            heights = torch.clip(self.simulator.base_pos[:, 2].unsqueeze(
                1) - 0.6 - self.simulator.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            single_critic_obs = torch.cat((single_critic_obs, heights), dim=-1)
        if self.cfg.terrain.obtain_terrain_info_around_feet:
            single_critic_obs = torch.cat(
                (
                    single_critic_obs,                         # previous
                    self.simulator.normal_vector_around_feet,  # 3*number of feet = 6
                    (self.simulator.feet_pos[:, :, 2].unsqueeze(-1) - 
                        self.simulator.height_around_feet).flatten(1,2).clip(-1.0, 1.0),  # 9*number of feet = 18
                ),
                dim=-1,
            )
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
            self.simulator.link_contact_states, # contact states of hip, knee and foot (2+2+2)=6
            torch.clip(self.simulator.feet_pos[:, :, 2] -
                torch.max(self.simulator.height_around_feet, dim=-1).values -
                self.cfg.rewards.foot_height_offset, -1, 1.),  # 2
            self.simulator.normal_vector_around_feet,  # 3*number of feet = 6
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
        # buffers for sit pose params
        self.sit_pos = torch.tensor(self.cfg.init_state.sit_pos, dtype=torch.float, device=self.device, requires_grad=False)
        self.sit_joint_angles = torch.tensor(
            [self.cfg.init_state.sit_joint_angles[name]
                for name in self.simulator.dof_names],
            device=self.device,
            dtype=torch.float,
        )
        # Periodic Reward Framework
        self.theta = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.theta[:, 0] = self.cfg.rewards.periodic_reward_framework.theta_left
        self.theta[:, 1] = self.cfg.rewards.periodic_reward_framework.theta_right
        self.gait_time = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.phi = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.gait_period = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.gait_period[:] = self.cfg.rewards.periodic_reward_framework.gait_period
        self.clock_input = torch.zeros(
            self.num_envs,
            4,
            dtype=torch.float,
            device=self.device,
        )
        self.b_swing = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.b_swing[:] = self.cfg.rewards.periodic_reward_framework.b_swing * 2 * torch.pi

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length ==0):
            self.update_command_curriculum(env_ids)

        self._resample_commands(env_ids)
        _ = np.random.random() # initialize the env at sit pose randomly
        if _ < self.cfg.init_state.sit_init_percent:
            self._reset_dofs_sit_pose(env_ids)
            self._reset_root_states_sit_pose(env_ids)
        else:
            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)
        self.simulator.reset_idx(env_ids)

        # reset buffers
        self.llast_actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.actions[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.fail_buf[env_ids] = 0
        # Periodic Reward Framework buffer reset
        self.theta[env_ids, 0] = self.cfg.rewards.periodic_reward_framework.theta_left + \
            torch.rand(len(env_ids), 1, device=self.device).squeeze(1)  # add small random offset
        self.theta[env_ids, 1] = self.theta[env_ids, 0] + (self.cfg.rewards.periodic_reward_framework.theta_right -
                                                           self.cfg.rewards.periodic_reward_framework.theta_left)
        self.gait_time[env_ids] = torch.rand(len(env_ids), 1, device=self.device) * self.gait_period[env_ids]
        self.phi[env_ids] = self.gait_time[env_ids] / self.gait_period[env_ids]
        self.clock_input[env_ids, :] = 0.0

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(
                self.simulator.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        # reset action queue and delay
        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue[env_ids] *= 0.
            self.action_queue[env_ids] = 0.
            self.action_delay[env_ids] = torch.randint(self.cfg.domain_rand.ctrl_delay_step_range[0],
                                                       self.cfg.domain_rand.ctrl_delay_step_range[1]+1, (len(env_ids),), device=self.device, requires_grad=False)
        # clear obs history for the envs that are reset
        for i in range(self.obs_history_deque.maxlen):
            self.obs_history_deque[i][env_ids] *= 0
        for i in range(self.critic_obs_deque.maxlen):
            self.critic_obs_deque[i][env_ids] *= 0
    
    def _calc_periodic_reward_obs(self):
        """Calculate the periodic reward observations.
        """
        for i in range(2):
            self.clock_input[:, i] = torch.sin(2 * torch.pi * (self.phi + self.theta[:, i].unsqueeze(1))).squeeze(-1)
            self.clock_input[:, i + 2] = torch.cos(2 * torch.pi * (self.phi + self.theta[:, i].unsqueeze(1))).squeeze(-1)

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
        dof_pos[:, [0, 3]] = self.simulator.default_dof_pos[:, [0, 3]] + \
            torch_rand_float(-0.2, 0.2, (len(env_ids), 2), self.device)
        dof_pos[:, [1, 4]] = self.simulator.default_dof_pos[:, [1, 4]] + \
            torch_rand_float(-0.4, 0.4, (len(env_ids), 2), self.device)
        dof_pos[:, [2, 5]] = self.simulator.default_dof_pos[:, [2, 5]] + \
            torch_rand_float(-0.4, 0.4, (len(env_ids), 2), self.device)
        self.simulator.reset_dofs(env_ids, dof_pos, dof_vel)
    
    def _reset_dofs_sit_pose(self, env_ids):
        dof_pos = self.sit_joint_angles.unsqueeze(0).repeat(len(env_ids), 1)
        dof_vel = torch.zeros((len(env_ids), self.num_actions), dtype=torch.float, 
                              device=self.device, requires_grad=False)
        self.simulator.reset_dofs(env_ids, dof_pos, dof_vel)
    
    def _reset_root_states_sit_pose(self, env_ids):
        if self.simulator.custom_origins:
            base_pos = self.sit_pos.reshape(1, -1).repeat(len(env_ids), 1)
            base_pos += self.simulator.env_origins[env_ids]
            base_pos[:, :2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            base_pos = self.sit_pos.reshape(1, -1).repeat(len(env_ids), 1)
            base_pos += self.simulator.env_origins[env_ids]
        base_quat = quat_from_euler_xyz(
                torch_rand_float(0, 0, (len(env_ids), 1), self.device).view(-1),
                torch_rand_float(self.cfg.init_state.sit_pitch_angle, 
                                 self.cfg.init_state.sit_pitch_angle, (len(env_ids), 1), self.device).view(-1),
                torch_rand_float(0, 0, (len(env_ids), 1), self.device).view(-1)
            )
        base_lin_vel = torch.zeros((len(env_ids), 3), dtype=torch.float, 
                                  device=self.device, requires_grad=False)
        base_ang_vel = torch.zeros((len(env_ids), 3), dtype=torch.float, 
                                  device=self.device, requires_grad=False)
        self.simulator.reset_root_states(env_ids, base_pos, base_quat, base_lin_vel, base_ang_vel)

    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.num_estimator_features = cfg.env.num_estimator_features
        self.num_estimator_labels = cfg.env.num_estimator_labels
        # Periodic Reward Framework. Constants are init here.
        self.kappa = self.cfg.rewards.periodic_reward_framework.kappa
        self.gait_function_type = self.cfg.rewards.periodic_reward_framework.gait_function_type
        self.a_swing = 0.0
        self.b_stance = 2 * torch.pi # a_stance(start of stance) = b_swing
    
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
    
    def _uniped_periodic_gait(self, foot_type):
        # q_frc and q_spd
        if foot_type == "left":
            q_frc = torch.norm(
                self.simulator.link_contact_forces[:, 
                                    self.simulator.feet_indices[0], :], dim=-1).view(-1, 1)
            q_spd = torch.norm(
                self.simulator.feet_vel[:, 0, :], dim=-1).view(-1, 1) # sequence of feet_pos is FL, FR, RL, RR
            # size: num_envs; need to reshape to (num_envs, 1), or there will be error due to broadcasting
            # modulo phi over 1.0 to get cicular phi in [0, 1.0]
            phi = (self.phi + self.theta[:, 0].unsqueeze(1)) % 1.0
        elif foot_type == "right":
            q_frc = torch.norm(
                self.simulator.link_contact_forces[:, 
                                    self.simulator.feet_indices[1], :], dim=-1).view(-1, 1)
            q_spd = torch.norm(
                self.simulator.feet_vel[:, 1, :], dim=-1).view(-1, 1)
            # modulo phi over 1.0 to get cicular phi in [0, 1.0]
            phi = (self.phi + self.theta[:, 1].unsqueeze(1)) % 1.0
        
        phi *= 2 * torch.pi  # convert phi to radians
        
        if self.gait_function_type == "smooth":
            # coefficient
            c_swing_spd = 0  # speed is not penalized during swing phase
            c_swing_frc = -1  # force is penalized during swing phase
            c_stance_spd = -1  # speed is penalized during stance phase
            c_stance_frc = 0  # force is not penalized during stance phase
            
            # clip the value of phi to [0, 1.0]. The vonmises function in scipy may return cdf outside [0, 1.0]
            F_A_swing = torch.clip(torch.tensor(vonmises.cdf(loc=self.a_swing, 
                kappa=self.kappa, x=phi.cpu()), device=self.device), 0.0, 1.0)
            F_B_swing = torch.clip(torch.tensor(vonmises.cdf(loc=self.b_swing.cpu(), 
                kappa=self.kappa, x=phi.cpu()), device=self.device), 0.0, 1.0)
            F_A_stance = F_B_swing
            F_B_stance = torch.clip(torch.tensor(vonmises.cdf(loc=self.b_stance,
                kappa=self.kappa, x=phi.cpu()), device=self.device), 0.0, 1.0)

            # calc the expected C_spd and C_frc according to the formula in the paper
            exp_swing_ind = F_A_swing * (1 - F_B_swing)
            exp_stance_ind = F_A_stance * (1 - F_B_stance)
            exp_C_spd_ori = c_swing_spd * exp_swing_ind + c_stance_spd * exp_stance_ind
            exp_C_frc_ori = c_swing_frc * exp_swing_ind + c_stance_frc * exp_stance_ind

            # just the code above can't result in the same reward curve as the paper
            # a little trick is implemented to make the reward curve same as the paper
            # first let all envs get the same exp_C_frc and exp_C_spd
            exp_C_frc = -0.5 + (-0.5 - exp_C_spd_ori)
            exp_C_spd = exp_C_spd_ori
            # select the envs that are in swing phase
            is_in_swing = (phi >= self.a_swing) & (phi < self.b_swing)
            indices_in_swing = is_in_swing.nonzero(as_tuple=False).flatten()
            # update the exp_C_frc and exp_C_spd of the envs in swing phase
            exp_C_frc[indices_in_swing] = exp_C_frc_ori[indices_in_swing]
            exp_C_spd[indices_in_swing] = -0.5 + \
                (-0.5 - exp_C_frc_ori[indices_in_swing])

            # Judge if it's the standing gait
            is_standing = (self.b_swing[:] == self.a_swing).nonzero(
                as_tuple=False).flatten()
            exp_C_frc[is_standing] = 0
            exp_C_spd[is_standing] = -1
        elif self.gait_function_type == "step":
            ''' ***** Step Gait Indicator ***** '''
            exp_C_frc = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
            exp_C_spd = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
            
            swing_indices = (phi >= self.a_swing) & (phi < self.b_swing)
            swing_indices = swing_indices.nonzero(as_tuple=False).flatten()
            stance_indices = (phi >= self.b_swing) & (phi < self.b_stance)
            stance_indices = stance_indices.nonzero(as_tuple=False).flatten()
            exp_C_frc[swing_indices, :] = -1
            exp_C_spd[swing_indices, :] = 0
            exp_C_frc[stance_indices, :] = 0
            exp_C_spd[stance_indices, :] = -1

        return exp_C_spd * q_spd + exp_C_frc * q_frc, \
            exp_C_spd.type(dtype=torch.float), exp_C_frc.type(dtype=torch.float)
    
    def _reward_biped_periodic_gait(self):
        biped_reward_left, self.exp_C_spd_left, self.exp_C_frc_left = self._uniped_periodic_gait(
            "left")
        biped_reward_right, self.exp_C_spd_right, self.exp_C_frc_right = self._uniped_periodic_gait(
            "right")
        # reward for the whole body
        biped_reward = biped_reward_left.flatten() + biped_reward_right.flatten()
        return torch.exp(biped_reward)
    
    def _reward_tracking_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.simulator.base_pos[:, 2].unsqueeze(
            1) - self.simulator.measured_heights, dim=1)
        rew = torch.square(base_height - self.cfg.rewards.base_height_target)
        return torch.exp(-rew / self.cfg.rewards.base_height_tracking_sigma)
    
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
                self.simulator.feet_pos[:, :, 2] - torch.max(self.simulator.height_around_feet, dim=-1).values -
                self.cfg.rewards.foot_clearance_target -
                self.cfg.rewards.foot_height_offset
            ), dim=-1
        )
        return torch.exp(-clearance_error / self.cfg.rewards.foot_clearance_tracking_sigma)
    
    def _reward_feet_distance(self):
        '''reward for feet distance'''
        feet_xy_distance = torch.norm(
            self.simulator.feet_pos[:, 0, [0, 1]] - self.simulator.feet_pos[:, 1, [0, 1]], dim=-1)
        return torch.max(torch.zeros_like(feet_xy_distance),
                         self.cfg.rewards.foot_distance_threshold - feet_xy_distance)
    
    def _reward_no_fly(self):
        contacts = self.simulator.link_contact_forces[:, self.simulator.feet_indices, 2] > 0.1
        single_contact = torch.sum(1.*contacts, dim=1)==1
        return 1.*single_contact