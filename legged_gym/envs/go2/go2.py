import genesis as gs

import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.gs_utils import *
from .go2_config import GO2Cfg

class GO2(LeggedRobot):
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