import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import numpy as np
import os
import sys

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math import wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.gs_utils import *
from .go2_gait_config import GO2GaitCfg
from scipy.stats import vonmises

class GO2Gait(LeggedRobot):
        
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        base_quat_rel = gs_quat_mul(self.base_quat, gs_inv_quat(self.base_init_quat.reshape(1, -1).repeat(self.num_envs, 1)))
        self.base_euler = gs_quat2euler(base_quat_rel)
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat) # trasform to base frame
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        self.links_vel[:] = self.robot.get_links_vel()
        self.link_contact_forces[:] = torch.tensor(
            self.robot.get_links_net_contact_force(),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.refresh_force_vel_components()
        
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_base_pos_out_of_bound()
        self.check_termination()
        self.compute_reward()
        
        #------ Periodic Reward Framework ------#
        # step after computing reward but before resetting the env
        self.gait_time += self.dt
        is_over_limit = (self.gait_time > self.gait_period + self.dt / 2)  # +self.dt/2 in case of float precision errors
        over_limit_indices = is_over_limit.nonzero(as_tuple=False).flatten()
        self.gait_time[over_limit_indices] = self.dt
        self.phi = self.gait_time / self.gait_period
        #------ Periodic Reward Framework ------#
        
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if self.num_build_envs > 0:
            self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        if self.debug_viz:
            self._draw_debug_vis()
    
    def refresh_force_vel_components(self):
        #------ Periodic Reward Framework ------#
        self.foot_vel_fl = self.links_vel[:, self.foot_index_fl, :]
        self.foot_vel_fr = self.links_vel[:, self.foot_index_fr, :]
        self.foot_vel_rl = self.links_vel[:, self.foot_index_rl, :]
        self.foot_vel_rr = self.links_vel[:, self.foot_index_rr, :]
        self.foot_contact_force_fl = self.link_contact_forces[:, self.foot_index_fl, :]
        self.foot_contact_force_fr = self.link_contact_forces[:, self.foot_index_fr, :]
        self.foot_contact_force_rl = self.link_contact_forces[:, self.foot_index_rl, :]
        self.foot_contact_force_rr = self.link_contact_forces[:, self.foot_index_rr, :]
        #------ Periodic Reward Framework ------#
    
    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.link_contact_forces[:, self.termination_indices, :], dim=-1)> 1.0, dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        # self.check_base_euler_termination()
        # self.check_dof_limit_termination()
    
    def check_base_euler_termination(self):
        """ Check if environments need to be reset
        """
        self.base_euler_reset_buf = torch.abs(self.base_euler[:, 0]) > torch.pi / 4
        self.reset_buf |= self.base_euler_reset_buf
    
    def check_dof_limit_termination(self):
        """ Check if environments need to be reset
        """
        lower_limit_reset_buf = torch.any(self.dof_pos < self.dof_pos_limits[:, 0], dim=1)
        upper_limit_reset_buf = torch.any(self.dof_pos > self.dof_pos_limits[:, 1], dim=1)
        dof_pos_limit_reset_buf = torch.logical_or(lower_limit_reset_buf, upper_limit_reset_buf)
        self.reset_buf |= dof_pos_limit_reset_buf

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        #------ Periodic Reward Framework ------#
        env_ids = (self.episode_length_buf % int(self.cfg.rewards.periodic_reward_framework.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self.resample_phase_and_theta(env_ids)
        #------ Periodic Reward Framework ------#
        
        if self.cfg.commands.heading_command:
            forward = gs_transform_by_quat(self.forward_vec, self.base_quat)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots:
            self._push_robots()
    
    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)
        #------ Periodic Reward Framework ------#
        self.resample_phase_and_theta(env_ids)
        #------ Periodic Reward Framework ------#
        
        # domain randomization
        if self.cfg.domain_rand.randomize_friction:
            self._randomize_friction(env_ids)
        if self.cfg.domain_rand.randomize_base_mass:
            self._randomize_base_mass(env_ids)
        if self.cfg.domain_rand.randomize_com_displacement:
            self._randomize_com_displacement(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        #------ Periodic Reward Framework ------#
        self.gait_time[env_ids] = 0.
        self.phi[env_ids] = 0
        #------ Periodic Reward Framework ------#
        
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    
    def compute_observations(self):
        """ Computes observations
        """
        #------ Periodic Reward Framework ------#
        self.calc_periodic_reward_obs()
        #------ Periodic Reward Framework ------#
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,                    # 3
                                    self.base_ang_vel  * self.obs_scales.ang_vel,                   # 3
                                    self.projected_gravity,                                         # 3
                                    self.commands[:, :3] * self.commands_scale,                     # 3
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,# num_dofs
                                    self.dof_vel * self.obs_scales.dof_vel,                         # num_dofs
                                    self.actions,                                                    # num_actions
                                    # self.clock_input,                                               # 4
                                    # self.phase_ratio,                                               # 2
                                    ),dim=-1)
        if torch.isnan(self.obs_buf).any():
            print("nan in obs_buf")
            nan_ids = torch.isnan(self.obs_buf).any(dim=1).nonzero(as_tuple=False).flatten()
            print(f"nan_ids: {nan_ids}")
            sys.exit()
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.base_pos[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.cat(
                (
                    self.base_lin_vel * self.obs_scales.lin_vel,
                    self.base_ang_vel  * self.obs_scales.ang_vel,
                    self.projected_gravity,
                    self.commands[:, :3] * self.commands_scale,
                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                    self.dof_vel * self.obs_scales.dof_vel,
                    self.actions,
                    self.last_actions,
                ),
                dim=-1,
            )
    
    #----------------------------------------
    def _get_noise_scale_vec(self, cfg):
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
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[36:48] = 0. # previous actions
        # noise_vec[48:52] = 0. # clock input
        # noise_vec[52:54] = 0. # phase ratio
        if self.cfg.terrain.measure_heights:
            noise_vec[54:241] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec
    
    def _reset_dofs(self, envs_idx):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        
        dof_pos = torch.zeros((len(envs_idx), self.num_actions), dtype=gs.tc_float, device=self.device)
        dof_pos[:, [0, 3, 6, 9]] = self.default_dof_pos[[0, 3, 6, 9]] + gs_rand_float(-0.2, 0.2, (len(envs_idx), 4), self.device)
        dof_pos[:, [1, 4, 7, 10]] = self.default_dof_pos[[0, 1, 4, 7]] + gs_rand_float(-0.4, 0.4, (len(envs_idx), 4), self.device)
        dof_pos[:, [2, 5, 8, 11]] = self.default_dof_pos[[0, 2, 5, 8]] + gs_rand_float(-0.4, 0.4, (len(envs_idx), 4), self.device)
        self.dof_pos[envs_idx] = dof_pos
        
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        self.robot.zero_all_dofs_velocity(envs_idx)
    
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.forward_vec = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.forward_vec[:, 0] = 1.0
        self.base_init_pos = torch.tensor(
            self.cfg.init_state.pos, device=self.device
        )
        self.base_init_quat = torch.tensor(
            self.cfg.init_state.rot, device=self.device
        )
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.cfg.commands.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], 
            device=self.device,
            dtype=gs.tc_float, 
            requires_grad=False,) # TODO change this
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.feet_air_time = torch.zeros((self.num_envs, len(self.feet_indices)), device=self.device, dtype=gs.tc_float)
        self.last_contacts = torch.zeros((self.num_envs, len(self.feet_indices)), device=self.device,dtype=gs.tc_int)
        self.links_vel = torch.zeros((self.num_envs, self.robot.n_links, 3), device=self.device, dtype=gs.tc_float)
        self.link_contact_forces = torch.zeros(
            (self.num_envs, self.robot.n_links, 3), device=self.device, dtype=gs.tc_float
        )
        #------ Periodic Reward Framework ------#
        self.foot_vel_fl = self.links_vel[:, self.foot_index_fl, :]
        self.foot_vel_fr = self.links_vel[:, self.foot_index_fr, :]
        self.foot_vel_rl = self.links_vel[:, self.foot_index_rl, :]
        self.foot_vel_rr = self.links_vel[:, self.foot_index_rr, :]
        self.foot_contact_force_fl = self.link_contact_forces[:, self.foot_index_fl, :]
        self.foot_contact_force_fr = self.link_contact_forces[:, self.foot_index_fr, :]
        self.foot_contact_force_rl = self.link_contact_forces[:, self.foot_index_rl, :]
        self.foot_contact_force_rr = self.link_contact_forces[:, self.foot_index_rr, :]
        self.gait_time = torch.zeros(self.num_envs, 1, device=self.device, dtype=gs.tc_float, requires_grad=False)
        self.gait_period = torch.zeros(self.num_envs, 1, device=self.device, dtype=gs.tc_float, requires_grad=False)
        self.phi = torch.zeros(self.num_envs, 1, device=self.device, dtype=gs.tc_float, requires_grad=False)
        self.phase_ratio = torch.zeros(self.num_envs, 2, device=self.device, dtype=gs.tc_float, requires_grad=False)
        self.b_swing = torch.zeros(self.num_envs, 1, device=self.device, dtype=gs.tc_float, requires_grad=False)
        self.a_stance = torch.zeros(self.num_envs, 1, device=self.device, dtype=gs.tc_float, requires_grad=False)
        # quad periodic reward
        self.theta = torch.zeros(self.num_envs, 4, device=self.device, dtype=gs.tc_float, requires_grad=False)
        self.theta[:, 0] = self.cfg.rewards.periodic_reward_framework.theta_fl[0] # default is the first value
        self.theta[:, 1] = self.cfg.rewards.periodic_reward_framework.theta_fr[0]
        self.theta[:, 2] = self.cfg.rewards.periodic_reward_framework.theta_rl[0]
        self.theta[:, 3] = self.cfg.rewards.periodic_reward_framework.theta_rr[0]
        self.b_swing[:, :] = self.cfg.rewards.periodic_reward_framework.b_swing[0] *2*torch.pi
        self.a_stance = self.b_swing
        self.gait_period[:] = self.cfg.rewards.periodic_reward_framework.gait_period[0]
        self.phase_ratio[:, 0] = self.cfg.rewards.periodic_reward_framework.swing_phase_ratio[0] # default is the first value
        self.phase_ratio[:, 1] = self.cfg.rewards.periodic_reward_framework.stance_phase_ratio[0]
        #------ Periodic Reward Framework ------#
        self.continuous_push = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.env_identities = torch.arange(
            self.num_envs,
            device=self.device,
            dtype=gs.tc_int, 
        )
        self.terrain_heights = torch.zeros(
            (self.num_envs,),
            device=self.device,
            dtype=gs.tc_float,
        )
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        self.default_dof_pos = torch.tensor(
            [self.cfg.init_state.default_joint_angles[name] for name in self.cfg.asset.dof_names],
            device=self.device,
            dtype=gs.tc_float,
        )
        # PD control
        stiffness = self.cfg.control.stiffness
        damping = self.cfg.control.damping
        
        self.p_gains, self.d_gains = [], []
        for dof_name in self.cfg.asset.dof_names:
            for key in stiffness.keys():
                if key in dof_name:
                    self.p_gains.append(stiffness[key])
                    self.d_gains.append(damping[key])
        self.p_gains = torch.tensor(self.p_gains, device=self.device)
        self.d_gains = torch.tensor(self.d_gains, device=self.device)
        self.batched_p_gains = self.p_gains[None, :].repeat(self.num_envs, 1)
        self.batched_d_gains = self.d_gains[None, :].repeat(self.num_envs, 1)
        # PD control params
        self.robot.set_dofs_kp(self.p_gains, self.motor_dofs)
        self.robot.set_dofs_kv(self.d_gains, self.motor_dofs)
    
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.sim_dt, 
                substeps=self.sim_substeps),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(1 / self.dt * self.cfg.control.decimation),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(self.cfg.viewer.num_rendered_envs))),
            rigid_options=gs.options.RigidOptions(
                dt=self.sim_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                enable_self_collision=self.cfg.asset.self_collisions,
            ),
            show_viewer= not self.headless,
        )
        # query rigid solver
        for solver in self.scene.sim.solvers:
            if not isinstance(solver, RigidSolver):
                continue
            self.rigid_solver = solver
            
        # add camera if needed
        if self.cfg.viewer.add_camera:
            self._setup_camera()
        
        # add terrain
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type=='plane':
            self.terrain = self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        elif mesh_type=='heightfield':
            self.utils_terrain = Terrain(self.cfg.terrain)
            self._create_heightfield()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self.terrain.set_friction(self.cfg.terrain.friction)
        # specify the boundary of the heightfield
        self.terrain_x_range = torch.zeros(2, device=self.device)
        self.terrain_y_range = torch.zeros(2, device=self.device)
        if self.cfg.terrain.mesh_type=='heightfield':
            self.terrain_x_range[0] = -self.cfg.terrain.border_size + 1.0 # give a small margin(1.0m)
            self.terrain_x_range[1] = self.cfg.terrain.border_size + self.cfg.terrain.num_rows * self.cfg.terrain.terrain_length - 1.0
            self.terrain_y_range[0] = -self.cfg.terrain.border_size + 1.0
            self.terrain_y_range[1] = self.cfg.terrain.border_size + self.cfg.terrain.num_cols * self.cfg.terrain.terrain_width - 1.0
        elif self.cfg.terrain.mesh_type=='plane': # the plane used has limited size, 
                                                  # and the origin of the world is at the center of the plane
            self.terrain_x_range[0] = -self.cfg.terrain.plane_length/2+1
            self.terrain_x_range[1] = self.cfg.terrain.plane_length/2-1
            self.terrain_y_range[0] = -self.cfg.terrain.plane_length/2+1 # the plane is a square
            self.terrain_y_range[1] = self.cfg.terrain.plane_length/2-1
        self._create_envs()
    
    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset, create entity
             2. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)
        
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=os.path.join(asset_root, asset_file),
                merge_fixed_links = True,  # if merge_fixed_links is True, then one link may have multiple geometries, which will cause error in set_friction_ratio
                links_to_keep = self.cfg.asset.links_to_keep,
                pos= np.array(self.cfg.init_state.pos),
                quat=np.array(self.cfg.init_state.rot),
                fixed = self.cfg.asset.fix_base_link,
            ),
            visualize_contact=self.debug,
        )
        
        # build
        self.scene.build(n_envs=self.num_envs)
        
        self._get_env_origins()
        
        if self.cfg.terrain.mesh_type=='plane' and not self.headless:
            self.scene.viewer.set_camera_pose(
                pos=((self.env_origins[9405, 0]-2.0).cpu().numpy(), self.env_origins[9405, 1].cpu().numpy(), 2.5),
                lookat=(self.env_origins[9405, 0].cpu().numpy(), self.env_origins[9405, 1].cpu().numpy(), 0.5),
            )
        
        # name to indices
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.dof_names]
        
        # find link indices, termination links, penalized links, and feet
        def find_link_indices(names):
            link_indices = list()
            for link in self.robot.links:
                flag = False
                for name in names:
                    if name in link.name:
                        flag = True
                if flag:
                    link_indices.append(link.idx - self.robot.link_start)
            return link_indices
        self.termination_indices = find_link_indices(self.cfg.asset.terminate_after_contacts_on)
        all_link_names = [link.name for link in self.robot.links]
        # print(f"all link names: {all_link_names}")
        # print("termination link indices:", self.termination_indices)
        self.penalized_indices = find_link_indices(self.cfg.asset.penalize_contacts_on)
        # print(f"penalized link indices: {self.penalized_indices}")
        self.feet_names = [name for name in all_link_names if self.cfg.asset.foot_name[0] in name]
        self.feet_indices = find_link_indices(self.cfg.asset.foot_name)
        print(f"feet link indices: {self.feet_indices}")
        #------ Periodic Reward Framework ------#
        for i in range(len(self.feet_names)):
            if "FL" in self.feet_names[i]:
                self.foot_index_fl = self.feet_indices[i]
            elif "FR" in self.feet_names[i]:
                self.foot_index_fr = self.feet_indices[i]
            elif "RL" in self.feet_names[i]:
                self.foot_index_rl = self.feet_indices[i]
            elif "RR" in self.feet_names[i]:
                self.foot_index_rr = self.feet_indices[i]
        #------ Periodic Reward Framework ------#
        assert len(self.termination_indices) > 0
        assert len(self.feet_indices) > 0
        self.feet_link_indices_world_frame = [i+1 for i in self.feet_indices]
        
        # dof position limits
        self.dof_pos_limits = torch.stack(self.robot.get_dofs_limit(self.motor_dofs), dim=1)
        self.dof_vel_limits = torch.tensor(self.cfg.asset.dof_vel_limits, device=self.device) # genesis doesn't provide get_dofs_vel_limit api, so specify it by hand
        self.torque_limits = self.robot.get_dofs_force_range(self.motor_dofs)[1]
        print(f"torque limits: {self.torque_limits}")
        for i in range(self.dof_pos_limits.shape[0]):
            # soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = (
                m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
            )
            self.dof_pos_limits[i, 1] = (
                m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        )
        
            
        # randomize friction
        if self.cfg.domain_rand.randomize_friction:
            self._randomize_friction(np.arange(self.num_envs))
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            self._randomize_base_mass(np.arange(self.num_envs))
        # randomize COM displacement
        if self.cfg.domain_rand.randomize_com_displacement:
            self._randomize_com_displacement(np.arange(self.num_envs))
    
    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.dt
        if self.cfg.sim.use_implicit_controller: # use embedded PD controller
            self.sim_dt = self.dt
            self.sim_substeps = self.cfg.control.decimation
        else: # use explicit PD controller
            self.sim_dt = self.dt / self.cfg.control.decimation
            self.sim_substeps = 1
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.push_interval_s = self.cfg.domain_rand.push_interval_s

        self.dof_names = self.cfg.asset.dof_names
        self.simulate_action_latency = self.cfg.domain_rand.simulate_action_latency
        self.debug = self.cfg.env.debug
        #------ Periodic Reward Framework ------#
        self.a_swing = self.cfg.rewards.periodic_reward_framework.a_swing * 2 * torch.pi
        self.b_stance = self.cfg.rewards.periodic_reward_framework.b_stance * 2 * torch.pi
        self.selected_gait = self.cfg.rewards.periodic_reward_framework.selected_gait
        self.kappa = self.cfg.rewards.periodic_reward_framework.kappa
        #------ Periodic Reward Framework ------#
    
    #------------ Private Functions ----------------
    #------ Periodic Reward Framework ------#
    def calc_periodic_reward_obs(self):
        # calculate clock inputs for the policy
        num_cycle_timesteps = self.gait_period / self.dt
        clock_input_fl = torch.sin(2*torch.pi*(self.phi+self.theta[:, 0].view(-1, 1)) / num_cycle_timesteps)
        clock_input_fr = torch.sin(2*torch.pi*(self.phi+self.theta[:, 1].view(-1, 1)) / num_cycle_timesteps)
        clock_input_rl = torch.sin(2*torch.pi*(self.phi+self.theta[:, 2].view(-1, 1)) / num_cycle_timesteps)
        clock_input_rr = torch.sin(2*torch.pi*(self.phi+self.theta[:, 3].view(-1, 1)) / num_cycle_timesteps)
        self.clock_input = torch.cat((clock_input_fl, clock_input_fr, clock_input_rl, clock_input_rr), dim=-1)

    def resample_phase_and_theta(self, env_ids):
        if self.selected_gait is not None:
            gait = self.selected_gait
            if gait == "stand":
                self.gait_period[env_ids, 0] = self.cfg.rewards.periodic_reward_framework.gait_period[0]
                self.theta[env_ids, 0] = self.cfg.rewards.periodic_reward_framework.theta_fl[0]
                self.theta[env_ids, 1] = self.cfg.rewards.periodic_reward_framework.theta_fr[0]
                self.theta[env_ids, 2] = self.cfg.rewards.periodic_reward_framework.theta_rl[0]
                self.theta[env_ids, 3] = self.cfg.rewards.periodic_reward_framework.theta_rr[0]
                self.b_swing[env_ids, 0] = self.cfg.rewards.periodic_reward_framework.b_swing[0] * 2 * torch.pi
                self.a_stance = self.b_swing
                self.phase_ratio[env_ids, 0] = self.cfg.rewards.periodic_reward_framework.swing_phase_ratio[0]
                self.phase_ratio[env_ids, 1] = self.cfg.rewards.periodic_reward_framework.stance_phase_ratio[0]
            elif gait == "trot":
                self.gait_period[env_ids, 0] = self.cfg.rewards.periodic_reward_framework.gait_period[1]
                self.theta[env_ids, 0] = self.cfg.rewards.periodic_reward_framework.theta_fl[1]
                self.theta[env_ids, 1] = self.cfg.rewards.periodic_reward_framework.theta_fr[1]
                self.theta[env_ids, 2] = self.cfg.rewards.periodic_reward_framework.theta_rl[1]
                self.theta[env_ids, 3] = self.cfg.rewards.periodic_reward_framework.theta_rr[1]
                self.b_swing[env_ids, 0] = self.cfg.rewards.periodic_reward_framework.b_swing[1] * 2 * torch.pi
                self.a_stance = self.b_swing
                self.phase_ratio[env_ids, 0] = self.cfg.rewards.periodic_reward_framework.swing_phase_ratio[1]
                self.phase_ratio[env_ids, 1] = self.cfg.rewards.periodic_reward_framework.stance_phase_ratio[1]
        else:
                gait_list = [0, 1]
                gait_choice = np.random.choice(gait_list)
                # update theta
                self.theta[env_ids, 0] = self.cfg.rewards.periodic_reward_framework.theta_fl[gait_choice]
                self.theta[env_ids, 1] = self.cfg.rewards.periodic_reward_framework.theta_fr[gait_choice]
                self.theta[env_ids, 2] = self.cfg.rewards.periodic_reward_framework.theta_rl[gait_choice]
                self.theta[env_ids, 3] = self.cfg.rewards.periodic_reward_framework.theta_rr[gait_choice]
                # update b_swing, phase ratio
                self.b_swing[env_ids, 0] = self.cfg.rewards.periodic_reward_framework.b_swing[gait_choice] * 2 * torch.pi
                self.a_stance = self.b_swing
                self.phase_ratio[env_ids, 0] = self.cfg.rewards.periodic_reward_framework.swing_phase_ratio[gait_choice]
                self.phase_ratio[env_ids, 1] = self.cfg.rewards.periodic_reward_framework.stance_phase_ratio[gait_choice]
        # update state indicator
        # self.omega_walking[env_ids] = 1/(1+torch.exp(-200*(self.phase_ratio[env_ids, 0] - 0.15)))
    def uniped_periodic_reward(self, foot_type):
        # coefficient
        c_swing_spd = 0 # speed is not penalized during swing phase
        c_swing_frc = -1 # force is penalized during swing phase
        c_stance_spd = -1 # speed is penalized during stance phase
        c_stance_frc = 0 # force is not penalized during stance phase
        
        # q_frc and q_spd
        if foot_type == "FL":
            q_frc = torch.norm(self.foot_contact_force_fl, dim=-1).view(-1, 1)
            q_spd = torch.norm(self.foot_vel_fl, dim=-1).view(-1, 1)
            # size: num_envs; need to reshape to (num_envs, 1), or there will be error due to broadcasting
                
            phi = (self.phi + self.theta[:, 0].view(-1, 1)) % 1.0 # modulo phi over 1.0 to get cicular phi in [0, 1.0]
        elif foot_type == "FR":
            q_frc = torch.norm(self.foot_contact_force_fr, dim=-1).view(-1, 1)
            q_spd = torch.norm(self.foot_vel_fr, dim=-1).view(-1, 1)
            phi = (self.phi + self.theta[:, 1].view(-1, 1)) % 1.0 # modulo phi over 1.0 to get cicular phi in [0, 1.0]
        elif foot_type == "RL":
            q_frc = torch.norm(self.foot_contact_force_rl, dim=-1).view(-1, 1)
            q_spd = torch.norm(self.foot_vel_rl, dim=-1).view(-1, 1)
            phi = (self.phi + self.theta[:, 2].view(-1, 1)) % 1.0 # modulo phi over 1.0 to get cicular phi in [0, 1.0]
        elif foot_type == "RR":
            q_frc = torch.norm(self.foot_contact_force_rr, dim=-1).view(-1, 1)
            q_spd = torch.norm(self.foot_vel_rr, dim=-1).view(-1, 1)
            phi = (self.phi + self.theta[:, 3].view(-1, 1)) % 1.0 # modulo phi over 1.0 to get cicular phi in [0, 1.0]
    
        phi *= 2 * torch.pi # convert phi to radians
        # clip the value of phi to [0, 1.0]. The vonmises function in scipy may return cdf outside [0, 1.0]
        F_A_swing = torch.clip(torch.tensor(vonmises.cdf(loc=self.a_swing.cpu(), kappa=self.kappa, x=phi.cpu()), device=self.device), 0.0, 1.0)
        F_B_swing = torch.clip(torch.tensor(vonmises.cdf(loc=self.b_swing.cpu(), kappa=self.kappa, x=phi.cpu()), device=self.device), 0.0, 1.0)
        F_A_stance = torch.clip(torch.tensor(vonmises.cdf(loc=self.a_stance.cpu(), kappa=self.kappa, x=phi.cpu()), device=self.device), 0.0, 1.0)
        F_B_stance = torch.clip(torch.tensor(vonmises.cdf(loc=self.b_stance.cpu(), kappa=self.kappa, x=phi.cpu()), device=self.device), 0.0, 1.0)
        
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
        exp_C_spd[indices_in_swing] = -0.5 + (-0.5 - exp_C_frc_ori[indices_in_swing])
        
        # Judge if it's the standing gait
        is_standing = (self.b_swing[:] == self.a_swing).nonzero(as_tuple=False).flatten()
        exp_C_frc[is_standing] = 0
        exp_C_spd[is_standing] = -1
        
        return exp_C_spd * q_spd + exp_C_frc * q_frc, exp_C_spd, exp_C_frc
    
    def _reward_periodic_gait(self):
        # reward for each foot
        reward_fl, self.exp_C_spd_fl, self.exp_C_frc_fl = self.uniped_periodic_reward("FL")
        reward_fr, self.exp_C_spd_fr, self.exp_C_frc_fr = self.uniped_periodic_reward("FR")
        reward_rl, self.exp_C_spd_rl, self.exp_C_frc_rl = self.uniped_periodic_reward("RL")
        reward_rr, self.exp_C_spd_rr, self.exp_C_frc_rr = self.uniped_periodic_reward("RR")
        # reward for the whole body
        reward = reward_fl.flatten() + reward_fr.flatten() + reward_rl.flatten() + reward_rr.flatten()
        # return torch.exp(reward) # only_positive_reward=True时, 加上periodic_gait奖励会导致训练崩溃
        return reward
    #------ Periodic Reward Framework ------#
    
    def _reward_feet_air_time(self):
        # Reward long steps
        contact = self.link_contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.2) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime