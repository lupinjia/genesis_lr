from legged_gym import *
from legged_gym.envs.go2.go2_ts.go2_ts_config import Go2TSCfg, Go2TSCfgPPO

class Go2CaTCfg( Go2TSCfg ):
  
    class rewards( Go2TSCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.34
        foot_clearance_target = 0.09 # desired foot clearance above ground [m]
        foot_height_offset = 0.022   # height of the foot coordinate origin above ground [m]
        foot_clearance_tracking_sigma = 0.01
        only_positive_rewards = True
        class scales( Go2TSCfg.rewards.scales ):
            # unused rewards
            dof_pos_limits = 0.0
            collision = 0.0
            stand_still = 0.0
            
            # command tracking
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            # smooth
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -0.5
            dof_power = -2.e-4
            dof_acc = -2.e-7
            action_rate = -0.01
            action_smoothness = -0.01
            # gait
            feet_air_time = 1.0
            hip_pos = -1.0
            dof_close_to_default = -0.05
            foot_clearance = 0.2
    
    class constraints:
        enable = "cat"        # enable constraint-as-terminations method
        tau_constraint = 0.95 # decay rate for violation of constraints
        soft_p = 0.25         # maximum termination probability for soft constraints  
        
        class limits:
            action_rate = 100.0
            max_projected_gravity = -0.1
            min_base_height = 0.25

class Go2CaTCfgPPO( Go2TSCfgPPO ):
    class runner( Go2TSCfgPPO.runner ):
        run_name = 'gs_cat'
        load_run = "Oct23_00-15-48_gs_cat"