from legged_gym import *
from warnings import WarningMessage

import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math_utils import *
from legged_gym.utils.helpers import class_to_dict
from collections import deque
from scipy.stats import vonmises


class GO2WTW(LeggedRobot):
    
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
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
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.llast_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.simulator.last_dof_vel[:] = self.simulator.dof_vel[:]

    def compute_observations(self):
        """ Computes observations
        """
        obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,    # cmd     3
            self.simulator.projected_gravity,                        # g       3
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,   # omega   3
            (self.simulator.dof_pos - self.simulator.default_dof_pos) *
            self.obs_scales.dof_pos,                       # p_t     12
            self.simulator.dof_vel * self.obs_scales.dof_vel,        # dp_t    12
            self.actions,                                  # a_{t-1} 12
            self.clock_input,                              # clock   4
            self.gait_period,                              # gait period 1
            self.base_height_target,                       # base height target 1
            self.foot_clearance_target,                    # foot clearance target 1
            self.pitch_target,                             # pitch target 1
            self.theta,                                    # theta, gait offset, 4
        ), dim=-1)

        if self.cfg.domain_rand.randomize_ctrl_delay:
            # normalize to [0, 1]
            ctrl_delay = (self.action_delay /
                          self.cfg.domain_rand.ctrl_delay_step_range[1]).unsqueeze(1)

        if self.num_privileged_obs is not None:  # critic_obs, no noise
            self.privileged_obs_buf = torch.cat((
                obs_buf,                                       # all above
                self.simulator.base_lin_vel * self.obs_scales.lin_vel,   # v_t     3
                # domain randomization parameters
                self.simulator._rand_push_vels[:, :2],                   # 2
                self.simulator._added_base_mass,                         # 1
                self.simulator._friction_values,                         # 1
                self.simulator._base_com_bias,                           # 3
                # ctrl_delay,                                    # 1
                self.simulator._kp_scale,                                # 12
                self.simulator._kd_scale,                                # 12
                # privileged infos
                self.exp_C_frc_fl,
                self.exp_C_frc_fr, 
                self.exp_C_frc_rl, 
                self.exp_C_frc_rr,           # 4
            ), dim=-1)

        # add noise if needed
        if self.add_noise:
            obs_now = obs_buf.clone()
            obs_now += (2 * torch.rand_like(obs_now) - 1) * \
                self.noise_scale_vec
        else:
            obs_now = obs_buf.clone()

        self.obs_history.append(obs_now)
        self.obs_buf = torch.cat(
            [self.obs_history[i] for i in range(self.obs_history.maxlen)], dim=-1
        )
        self.critic_history.append(self.privileged_obs_buf)
        self.privileged_obs_buf = torch.cat(
            [self.critic_history[i] for i in range(self.critic_history.maxlen)], dim=-1
        )

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length ==0):
            self.update_command_curriculum(env_ids)
            self._update_behavior_param_curriculum(env_ids)

        # reset robot states
        self._resample_behavior_params(env_ids)
        self._resample_commands(env_ids)
        self._reset_dofs(env_ids)
        self.simulator.reset_idx(env_ids)

        # reset buffers
        self.llast_actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.fail_buf[env_ids] = 0
        # Periodic Reward Framework buffer reset
        self.gait_time[env_ids] = 0.0
        self.phi[env_ids] = 0.0
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
                self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        # Behavior parameters
        self.extras["episode"]["gait_period_max"] = self.gait_period_range[1]
        self.extras["episode"]["base_height_target_max"] = self.base_height_target_range[1]
        self.extras["episode"]["foot_clearance_target_max"] = self.foot_clearance_target_range[1]
        self.extras["episode"]["pitch_target_max"] = self.pitch_target_range[1]
        self.extras["episode"]["num_gaits"] = self.num_gaits
        
        # reset action queue and delay
        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue[env_ids] *= 0.
            self.action_queue[env_ids] = 0.
            self.action_delay[env_ids] = torch.randint(self.cfg.domain_rand.ctrl_delay_step_range[0],
                                                       self.cfg.domain_rand.ctrl_delay_step_range[1]+1, (len(env_ids),), device=self.device, requires_grad=False)

        # clear obs and critic history for the envs that are reset
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

    def _resample_behavior_params(self, env_ids):
        if len(env_ids) == 0:
            return
        self.gait_period[env_ids, :] = torch_rand_float(
            self.gait_period_range[0],
            self.gait_period_range[1],
            (len(env_ids), 1), device=self.device
        )
        self.base_height_target[env_ids, :] = torch_rand_float(
            self.base_height_target_range[0],
            self.base_height_target_range[1],
            (len(env_ids), 1), device=self.device
        )
        self.foot_clearance_target[env_ids, :] = torch_rand_float(
            self.foot_clearance_target_range[0],
            self.foot_clearance_target_range[1],
            (len(env_ids), 1), device=self.device
        )
        self.pitch_target[env_ids, :] = torch_rand_float(
            self.pitch_target_range[0],
            self.pitch_target_range[1],
            (len(env_ids), 1), device=self.device
        )
        
        # Theta, gait offset
        selected_idx = torch.randint(0, self.num_gaits, 
                                         (1,), device=self.device)
        self.theta[env_ids, 0] = self.cfg.rewards.periodic_reward_framework.theta_fl_list[selected_idx]
        self.theta[env_ids, 1] = self.cfg.rewards.periodic_reward_framework.theta_fr_list[selected_idx]
        self.theta[env_ids, 2] = self.cfg.rewards.periodic_reward_framework.theta_rl_list[selected_idx]
        self.theta[env_ids, 3] = self.cfg.rewards.periodic_reward_framework.theta_rr_list[selected_idx]
        
        # Environments with pronk and bound gait should not give too high foot clearance target
        pronk_env_ids = ((self.theta[:, 0] == 0.0) & (self.theta[:, 1] == 0.0) & \
                        (self.theta[:, 2] == 0.0) & (self.theta[:, 3] == 0.0)).nonzero(as_tuple=False).flatten()
        bound_env_ids = ((self.theta[:, 0] == 0.0) & (self.theta[:, 1] == 0.0) & \
                        (self.theta[:, 2] == 0.5) & (self.theta[:, 3] == 0.5)).nonzero(as_tuple=False).flatten()
        self.foot_clearance_target[pronk_env_ids, :] = self.foot_clearance_target_range[0]
        self.foot_clearance_target[bound_env_ids, :] = self.foot_clearance_target_range[0]

    def _update_behavior_param_curriculum(self, env_ids):
        if len(env_ids) == 0:
            return
        # Widen the behavior param range according to reward values
        if torch.mean(self.episode_sums["quad_periodic_gait"][env_ids]) / \
            self.max_episode_length > 0.8 * self.reward_scales["quad_periodic_gait"]: # 0.8 for step gait, 0.5 for smooth gait
            # gait period
            self.gait_period_range[0] = max(self.gait_period_range[0] - 0.05, self.gait_period_min)
            self.gait_period_range[1] = min(self.gait_period_range[1] + 0.05, self.gait_period_max)
            # gait number
            self.num_gaits = min(self.num_gaits + 1, self.num_gait_max)

        if torch.mean(self.episode_sums["tracking_base_height"][env_ids]) / \
            self.max_episode_length > 0.9 * self.reward_scales["tracking_base_height"]:
            self.base_height_target_range[0] = max(self.base_height_target_range[0] - 0.02, self.base_height_target_min)
            self.base_height_target_range[1] = min(self.base_height_target_range[1] + 0.02, self.base_height_target_max)

        if torch.mean(self.episode_sums["tracking_foot_clearance"][env_ids]) / \
            self.max_episode_length > 0.8 * self.reward_scales["tracking_foot_clearance"]:
            self.foot_clearance_target_range[0] = max(self.foot_clearance_target_range[0] - 0.01, 
                                                      self.foot_clearance_target_min)
            self.foot_clearance_target_range[1] = min(self.foot_clearance_target_range[1] + 0.01, 
                                                      self.foot_clearance_target_max)
        
        if torch.mean(self.episode_sums["tracking_orientation"][env_ids]) / \
            self.max_episode_length > 0.9 * self.reward_scales["tracking_orientation"]:
            self.pitch_target_range[0] = max(self.pitch_target_range[0] - 0.05, self.pitch_target_min)
            self.pitch_target_range[1] = min(self.pitch_target_range[1] + 0.05, self.pitch_target_max)

    # ------------- Callbacks --------------
    
    def _calc_periodic_reward_obs(self):
        """Calculate the periodic reward observations.
        """
        for i in range(4):
            self.clock_input[:, i] = torch.sin(2 * torch.pi * (self.phi + self.theta[:, i].unsqueeze(1))).squeeze(-1)
            self.clock_input[:, i + 4] = torch.cos(2 * torch.pi * (self.phi + self.theta[:, i].unsqueeze(1))).squeeze(-1)
    
    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        env_ids = (self.episode_length_buf % int(
            self.cfg.rewards.behavior_params_range.resampling_time / self.dt) == 0).nonzero(as_tuple=False).flatten()
        # Periodic Reward Framework. resample phase and theta
        self._resample_behavior_params(env_ids)

    def _get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(
            self.cfg.env.num_single_obs, dtype=torch.float, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = 0.  # commands
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = noise_scales.ang_vel * \
            noise_level * self.obs_scales.ang_vel
        noise_vec[9:9+1*self.num_actions] = noise_scales.dof_pos * \
            noise_level * self.obs_scales.dof_pos                    # p_t
        noise_vec[9+1*self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * \
            noise_level * self.obs_scales.dof_vel  # dp_t
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0.  # a_{t-dt}

        # if self.cfg.terrain.measure_heights:
        #     noise_vec[48:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements

        return noise_vec

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        super()._init_buffers()

        # obs_history
        self.obs_history = deque(maxlen=self.cfg.env.frame_stack)
        self.critic_history = deque(maxlen=self.cfg.env.c_frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.num_single_obs,
                    dtype=torch.float,
                    device=self.device,
                )
            )
        for _ in range(self.cfg.env.c_frame_stack):
            self.critic_history.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.single_num_privileged_obs,
                    dtype=torch.float,
                    device=self.device,
                )
            )
        # Periodic Reward Framework
        self.theta = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.theta[:, 0] = self.cfg.rewards.periodic_reward_framework.theta_fl_list[0]
        self.theta[:, 1] = self.cfg.rewards.periodic_reward_framework.theta_fr_list[0]
        self.theta[:, 2] = self.cfg.rewards.periodic_reward_framework.theta_rl_list[0]
        self.theta[:, 3] = self.cfg.rewards.periodic_reward_framework.theta_rr_list[0]
        self.gait_time = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.phi = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.gait_period = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.gait_period[:] = self.gait_period_range[0]
        self.clock_input = torch.zeros(
            self.num_envs,
            8,
            dtype=torch.float,
            device=self.device,
        )
        self.b_swing = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.b_swing[:] = self.cfg.rewards.periodic_reward_framework.b_swing * 2 * torch.pi
        # Tracking params
        self.base_height_target = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device
        )
        self.base_height_target[:, :] = self.base_height_target_range[0]
        self.foot_clearance_target = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device
        )
        self.foot_clearance_target[:, :] = self.foot_clearance_target_range[0]
        self.pitch_target = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device
        )
        self.pitch_target[:, :] = self.pitch_target_range[0]
            
    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        # Periodic Reward Framework. Constants are init here.
        self.kappa = self.cfg.rewards.periodic_reward_framework.kappa
        self.gait_function_type = self.cfg.rewards.periodic_reward_framework.gait_function_type
        self.a_swing = 0.0
        self.b_stance = 2 * torch.pi
        # Process behavior param range, specify to medium value initially
        self.gait_period_min = self.cfg.rewards.behavior_params_range.gait_period_range[0]
        self.gait_period_max = self.cfg.rewards.behavior_params_range.gait_period_range[1]
        self.gait_period_range = [(self.gait_period_min + self.gait_period_max) / 2] * 2
        self.foot_clearance_target_min = self.cfg.rewards.behavior_params_range.foot_clearance_target_range[0]
        self.foot_clearance_target_max = self.cfg.rewards.behavior_params_range.foot_clearance_target_range[1]
        self.foot_clearance_target_range = [self.foot_clearance_target_min] * 2
        self.base_height_target_min = self.cfg.rewards.behavior_params_range.base_height_target_range[0]
        self.base_height_target_max = self.cfg.rewards.behavior_params_range.base_height_target_range[1]
        self.base_height_target_range = [(self.base_height_target_min + self.base_height_target_max) / 2] * 2
        self.pitch_target_min = self.cfg.rewards.behavior_params_range.pitch_target_range[0]
        self.pitch_target_max = self.cfg.rewards.behavior_params_range.pitch_target_range[1]
        self.pitch_target_range = [(self.pitch_target_min + self.pitch_target_max) / 2] * 2
        self.num_gaits = 1     # start from one gait initially
        self.num_gait_max = len(self.cfg.rewards.periodic_reward_framework.theta_fl_list)
        
    def _uniped_periodic_gait(self, foot_type):
        # q_frc and q_spd
        if foot_type == "FL":
            q_frc = torch.norm(
                self.simulator.link_contact_forces[:, 
                                    self.simulator.feet_indices[0], :], dim=-1).view(-1, 1)
            q_spd = torch.norm(
                self.simulator.feet_vel[:, 0, :], dim=-1).view(-1, 1) # sequence of feet_pos is FL, FR, RL, RR
            # size: num_envs; need to reshape to (num_envs, 1), or there will be error due to broadcasting
            # modulo phi over 1.0 to get cicular phi in [0, 1.0]
            phi = (self.phi + self.theta[:, 0].unsqueeze(1)) % 1.0
        elif foot_type == "FR":
            q_frc = torch.norm(
                self.simulator.link_contact_forces[:, 
                                    self.simulator.feet_indices[1], :], dim=-1).view(-1, 1)
            q_spd = torch.norm(
                self.simulator.feet_vel[:, 1, :], dim=-1).view(-1, 1)
            # modulo phi over 1.0 to get cicular phi in [0, 1.0]
            phi = (self.phi + self.theta[:, 1].unsqueeze(1)) % 1.0
        elif foot_type == "RL":
            q_frc = torch.norm(
                self.simulator.link_contact_forces[:, 
                                    self.simulator.feet_indices[2], :], dim=-1).view(-1, 1)
            q_spd = torch.norm(
                self.simulator.feet_vel[:, 2, :], dim=-1).view(-1, 1)
            # modulo phi over 1.0 to get cicular phi in [0, 1.0]
            phi = (self.phi + self.theta[:, 2].unsqueeze(1)) % 1.0
        elif foot_type == "RR":
            q_frc = torch.norm(
                self.simulator.link_contact_forces[:, 
                                    self.simulator.feet_indices[3], :], dim=-1).view(-1, 1)
            q_spd = torch.norm(
                self.simulator.feet_vel[:, 3, :], dim=-1).view(-1, 1)
            # modulo phi over 1.0 to get cicular phi in [0, 1.0]
            phi = (self.phi + self.theta[:, 3].unsqueeze(1)) % 1.0
        
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
    
    def _reward_quad_periodic_gait(self):
        quad_reward_fl, self.exp_C_spd_fl, self.exp_C_frc_fl = self._uniped_periodic_gait(
            "FL")
        quad_reward_fr, self.exp_C_spd_fr, self.exp_C_frc_fr = self._uniped_periodic_gait(
            "FR")
        quad_reward_rl, self.exp_C_spd_rl, self.exp_C_frc_rl = self._uniped_periodic_gait(
            "RL")
        quad_reward_rr, self.exp_C_spd_rr, self.exp_C_frc_rr = self._uniped_periodic_gait(
            "RR")
        # reward for the whole body
        quad_reward = quad_reward_fl.flatten() + quad_reward_fr.flatten() + \
            quad_reward_rl.flatten() + quad_reward_rr.flatten()
        return torch.exp(quad_reward)
    
    def _reward_hip_pos(self):
        """ Reward for the hip joint position close to default position
        """
        hip_joint_indices = [0, 3, 6, 9]
        dof_pos_error = torch.sum(torch.square(
            self.simulator.dof_pos[:, hip_joint_indices] - 
            self.simulator.default_dof_pos[:, hip_joint_indices]), dim=-1)
        return dof_pos_error
    
    def _reward_tracking_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.simulator.base_pos[:, 2].unsqueeze(
            1) - self.simulator.measured_heights, dim=1)
        rew = torch.square(base_height - self.base_height_target.squeeze(1))
        return torch.exp(-rew / self.cfg.rewards.base_height_tracking_sigma)

    def _reward_tracking_orientation(self):
        roll_error = torch.square(self.simulator.base_euler[:, 0])
        pitch_error = torch.square(self.simulator.base_euler[:, 1] - self.pitch_target.squeeze(1))
        return torch.exp(-(roll_error + pitch_error) / self.cfg.rewards.euler_tracking_sigma)
    
    def _reward_tracking_foot_clearance(self):
        """
        Encourage feet to be close to desired height while swinging
        """
        foot_vel_xy_norm = torch.norm(self.simulator.feet_vel[:, :, :2], dim=-1)
        clearance_error = torch.sum(
            foot_vel_xy_norm * torch.square(
                self.simulator.feet_pos[:, :, 2] -
                self.foot_clearance_target -
                self.cfg.rewards.foot_height_offset
            ), dim=-1
        )
        return torch.exp(-clearance_error / self.cfg.rewards.foot_clearance_tracking_sigma)