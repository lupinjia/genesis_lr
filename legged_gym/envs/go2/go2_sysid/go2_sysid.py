import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
from genesis.engine.solvers.avatar_solver import AvatarSolver
import numpy as np

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.gs_utils import *
from .go2_sysid_config import GO2SysIDCfg
import pandas as pd
from tqdm import tqdm


class GO2SysID(LeggedRobot):

    def system_id_in_air(self, env_cfg):
        # load motor_data_file, csv
        motor_data_file = self.cfg.sysid_data.file.format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        motor_data = pd.read_csv(motor_data_file)
        motor_q = motor_data[["jpos0", "jpos1", "jpos2", "jpos3", "jpos4", "jpos5",
                              "jpos6", "jpos7", "jpos8", "jpos9", "jpos10", "jpos11"]].to_numpy()
        # motor_dq = motor_data[["jvel_0","jvel_1","jvel_2","jvel_3","jvel_4","jvel_5"]].to_numpy()
        motor_q_des = motor_data[["jpos0_des", "jpos1_des", "jpos2_des", "jpos3_des", "jpos4_des", "jpos5_des",
                                  "jpos6_des", "jpos7_des", "jpos8_des", "jpos9_des", "jpos10_des", "jpos11_des"]].to_numpy()
        motor_kp = np.random.uniform(
            self.cfg.sysid_param_range.kp_range[0], self.cfg.sysid_param_range.kp_range[1], (self.num_envs, self.num_actions))
        motor_kd = np.random.uniform(
            self.cfg.sysid_param_range.kd_range[0], self.cfg.sysid_param_range.kd_range[1], (self.num_envs, self.num_actions))

        q_real = torch.from_numpy(motor_q).float().to(self.device)
        qd_real = torch.zeros_like(q_real).to(self.device)
        q_des = torch.from_numpy(motor_q_des).float().to(self.device)
        kp_des = torch.from_numpy(motor_kp).float().to(self.device)
        kd_des = torch.from_numpy(motor_kd).float().to(self.device)

        # sample parameters
        joint_damping_range = self.cfg.sysid_param_range.joint_damping_range
        joint_stiffness_range = self.cfg.sysid_param_range.joint_stiffness_range
        joint_armature_range = self.cfg.sysid_param_range.joint_armature_range
        # set parameters
        sampled_dampings = np.zeros((self.num_envs,))
        sampled_stiffness = np.zeros((self.num_envs,))
        sampled_armatures = np.zeros((self.num_envs,))

        # Set joint_friction, joint_damping, joint_armature
        # Generate random values for each env, then repeat for each action
        # All joints in the same env share the same damping, stiffness and armature
        joint_dampings = gs_rand_float(
            joint_damping_range[0], joint_damping_range[1], (self.num_envs,1), device=self.device).repeat(1, self.num_actions)
        joint_stiffness = gs_rand_float(
            joint_stiffness_range[0], joint_stiffness_range[1], (self.num_envs,1), device=self.device).repeat(1, self.num_actions)
        joint_armature = gs_rand_float(
            joint_armature_range[0], joint_armature_range[1], (self.num_envs,1), device=self.device).repeat(1, self.num_actions)

        # assume all joints have the same damping and friction
        for i in range(self.num_envs):
            sampled_dampings[i] = joint_dampings[i][0]
            sampled_stiffness[i] = joint_stiffness[i][0]
            sampled_armatures[i] = joint_armature[i][0]
        self.robot.set_dofs_damping(
            joint_dampings, self.motors_dof_idx)
        self.robot.set_dofs_stiffness(
            joint_stiffness, self.motors_dof_idx)
        self.robot.set_dofs_armature(
            joint_armature, self.motors_dof_idx)

        # check
        dof_dampings = self.robot.get_dofs_damping()
        dof_stiffness = self.robot.get_dofs_stiffness()
        dof_armatures = self.robot.get_dofs_armature()
        for i in range(self.num_envs):
            assert abs(dof_dampings[i][0] - sampled_dampings[i]) < 1e-4
            assert abs(dof_stiffness[i][0] - sampled_stiffness[i]) < 1e-4
            assert abs(dof_armatures[i][0] - sampled_armatures[i]) < 1e-4
        # generating samples
        metric = 0
        sim_q = []

        # reset
        envs_idx = torch.arange(self.num_envs).to(self.device)
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_pos[envs_idx] += self.env_origins[envs_idx]
        self.robot.set_pos(
            self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_quat(
            self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.zero_all_dofs_velocity(envs_idx)

        self.dof_pos[envs_idx] = (self.default_dof_pos)
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        self.robot.zero_all_dofs_velocity(envs_idx)

        self.p_gains[:] = kp_des[:]
        self.d_gains[:] = kd_des[:]

        delay_steps = self.cfg.domain_rand.delay_steps

        for i in tqdm(range(1 + delay_steps, q_real.shape[0])):
            # apply action
            actions = ((q_des[i-1-delay_steps] - self.default_dof_pos) /
                       self.cfg.control.action_scale).tile((self.num_envs, 1))
            # step physics and render each frame
            self.torques = self._compute_torques(actions)
            self.robot.control_dofs_force(self.torques, self.motors_dof_idx)
            self.scene.step()
            self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
            self.dof_vel[:] = self.robot.get_dofs_velocity(
                self.motors_dof_idx)
            # when sampling
            metric = metric + \
                torch.norm(self.dof_pos - q_real[i].unsqueeze(dim=0), dim=-1)
            sim_q.append(self.dof_pos.cpu().numpy())

        metric = metric.detach().cpu().numpy()
        # print("Average metric", np.mean(metric))
        print("best")
        print("damping", sampled_dampings[np.argmin(metric)], "\n",
              "stiffness", sampled_stiffness[np.argmin(metric)], "\n",
              "armature", sampled_armatures[np.argmin(metric)], "\n",
              #   "limb_mass_ratios", self.sampled_link_mass_scales[np.argmin(metric)], "\n",
              #   "feet_friction", self.gym.get_actor_rigid_shape_properties(self.envs[np.argmin(metric)], self.actor_handles[np.argmin(metric)])[self.feet_indices[0]].friction,
              #   "rb_restitution", self.gym.get_actor_rigid_shape_properties(self.envs[np.argmin(metric)], self.actor_handles[np.argmin(metric)])[0].restitution,
              #   "mass", [self.gym.get_actor_rigid_body_properties(self.envs[np.argmin(metric)], self.actor_handles[np.argmin(metric)])[i].mass for i in range(self.num_bodies)],
              #   "com", self.gym.get_actor_rigid_body_properties(self.envs[np.argmin(metric)], self.actor_handles[np.argmin(metric)])[0].com,
              "kp", kp_des[np.argmin(metric)], "\n",
              "kd", kd_des[np.argmin(metric)], "\n",
              "metric", metric[np.argmin(metric)])
        # print("worst", "damping", sampled_dampings[np.argmax(metric)],
        #       "friction", sampled_frictions[np.argmax(metric)],
        #       "limb_mass_ratios", self.sampled_limb_mass_scales[np.argmax(metric)],
        #     #   "feet_friction", self.gym.get_actor_rigid_shape_properties(self.envs[np.argmax(metric)], self.actor_handles[np.argmax(metric)])[self.feet_indices[0]].friction,
        #     #   "rb_restitution", self.gym.get_actor_rigid_shape_properties(self.envs[np.argmax(metric)], self.actor_handles[np.argmax(metric)])[0].restitution,
        #     #   "mass", [self.gym.get_actor_rigid_body_properties(self.envs[np.argmax(metric)], self.actor_handles[np.argmax(metric)])[i].mass for i in range(self.num_bodies)],
        #     #   "com", self.gym.get_actor_rigid_body_properties(self.envs[np.argmax(metric)], self.actor_handles[np.argmax(metric)])[0].com,
        #        "kp", kp_des[np.argmax(metric)],
        #        "kd", kd_des[np.argmax(metric)],
        #         "armature", sampled_armatures[np.argmax(metric)],
        #        metric[np.argmax(metric)])

    def _reset_dofs(self, envs_idx):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """

        self.dof_pos[envs_idx, :] = self.default_dof_pos # reset dof_pos to default position

        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        self.robot.zero_all_dofs_velocity(envs_idx)

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
                camera_pos=np.array(self.cfg.viewer.pos),
                camera_lookat=np.array(self.cfg.viewer.lookat),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx= self.cfg.viewer.rendered_envs_idx),
            rigid_options=gs.options.RigidOptions(
                dt=self.sim_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                enable_self_collision=self.cfg.asset.self_collisions,
                batch_dofs_info=True,
                batch_joints_info=True,
                batch_links_info=True,
            ),
            show_viewer=not self.headless,
        )
        # query rigid solver
        for solver in self.scene.sim.solvers:
            if not isinstance(solver, RigidSolver):
                continue
            elif isinstance(solver, AvatarSolver):
                continue
            self.rigid_solver = solver

        # add camera if needed
        if self.cfg.viewer.add_camera:
            self._setup_camera()

        # add terrain
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type =='plane':
            self.terrain = self.scene.add_entity(
                gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        elif mesh_type =='heightfield':
            self.utils_terrain = Terrain(self.cfg.terrain)
            self._create_heightfield()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self.terrain.set_friction(self.cfg.terrain.friction)
        # specify the boundary of the heightfield
        self.terrain_x_range = torch.zeros(2, device=self.device)
        self.terrain_y_range = torch.zeros(2, device=self.device)
        if self.cfg.terrain.mesh_type =='heightfield':
            self.terrain_x_range[0] = -self.cfg.terrain.border_size + 1.0  # give a small margin(1.0m)
            self.terrain_x_range[1] = self.cfg.terrain.border_size + \
                self.cfg.terrain.num_rows * self.cfg.terrain.terrain_length - 1.0
            self.terrain_y_range[0] = -self.cfg.terrain.border_size + 1.0
            self.terrain_y_range[1] = self.cfg.terrain.border_size + \
                self.cfg.terrain.num_cols * self.cfg.terrain.terrain_width - 1.0
        elif self.cfg.terrain.mesh_type =='plane': # the plane used has limited size, 
                                                 # and the origin of the world is at the center of the plane
            self.terrain_x_range[0] = -self.cfg.terrain.plane_length/2+1
            self.terrain_x_range[1] = self.cfg.terrain.plane_length/2-1
            self.terrain_y_range[0] = -self.cfg.terrain.plane_length/2+1  # the plane is a square
            self.terrain_y_range[1] = self.cfg.terrain.plane_length/2-1
        self._create_envs()
    
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec()
        self.forward_vec = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float
        )
        self.forward_vec[:, 0] = 1.0
        self.base_init_pos = torch.tensor(
            self.cfg.init_state.pos, device=self.device
        )
        self.base_init_quat = torch.tensor(
            self.cfg.init_state.rot, device=self.device
        )
        self.base_lin_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float)
        self.base_ang_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float)
        self.projected_gravity = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float).repeat(
            self.num_envs, 1
        )
        self.commands = torch.zeros(
            (self.num_envs, self.cfg.commands.num_commands), device=self.device, dtype=torch.float)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
                                           device=self.device,
            dtype=torch.float,
                                           requires_grad=False,)
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        self.last_actions = torch.zeros_like(self.actions)
        self.llast_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # last last actions
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float)
        self.base_quat = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=torch.float)
        self.feet_air_time = torch.zeros(
            (self.num_envs, len(self.feet_indices)), device=self.device, dtype=torch.float)
        self.last_contacts = torch.zeros((self.num_envs, len(self.feet_indices)), device=self.device, dtype=gs.tc_int)
        self.link_contact_forces = torch.zeros(
            (self.num_envs, self.robot.n_links, 3), device=self.device, dtype=torch.float
        )
        self.feet_pos = torch.zeros(
            (self.num_envs, len(self.feet_indices), 3), device=self.device, dtype=torch.float
        )
        self.feet_vel = torch.zeros(
            (self.num_envs, len(self.feet_indices), 3), device=self.device, dtype=torch.float
        )
        self.continuous_push = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float
        )
        self.env_identities = torch.arange(
            self.num_envs,
            device=self.device,
            dtype=gs.tc_int,
        )
        self.terrain_heights = torch.zeros(
            (self.num_envs,),
            device=self.device,
            dtype=torch.float,
        )
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # randomize action delay
        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue = torch.zeros(
                self.num_envs, self.cfg.domain_rand.ctrl_delay_step_range[1]+1, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
            self.action_delay = torch.randint(self.cfg.domain_rand.ctrl_delay_step_range[0],
                                              self.cfg.domain_rand.ctrl_delay_step_range[1]+1, (self.num_envs,), device=self.device, requires_grad=False)

        self.default_dof_pos = torch.tensor(
            [self.cfg.init_state.default_joint_angles[name]
                for name in self.cfg.asset.dof_names],
            device=self.device,
            dtype=torch.float,
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
        self.p_gains = self.p_gains[None, :].repeat(self.num_envs, 1) # use batched gains
        self.d_gains = self.d_gains[None, :].repeat(self.num_envs, 1)
        # PD control params
        self.robot.set_dofs_kp(self.p_gains, self.motors_dof_idx)
        self.robot.set_dofs_kv(self.d_gains, self.motors_dof_idx)
    
    def _compute_torques(self, actions):
        # control_type = 'P'
        actions_scaled = actions * self.cfg.control.action_scale
        torques = (
            self.p_gains * (actions_scaled + \
                            self.default_dof_pos - self.dof_pos)
            - self.d_gains * self.dof_vel
        )
        return torques