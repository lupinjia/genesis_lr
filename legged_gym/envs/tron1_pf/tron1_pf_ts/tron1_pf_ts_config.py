from legged_gym import *
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class TRON1PF_TSCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_observations = 27  # num_obs
        num_privileged_obs = 51
        frame_stack = 20    # number of frames to stack for obs_history
        num_history_obs = int(num_observations * frame_stack)
        num_latent_dims = num_privileged_obs
        c_frame_stack = 5
        single_critic_obs_len = num_observations + 19 + 81 + 8
        num_critic_obs = c_frame_stack * single_critic_obs_len
        # Privileged_obs and critic_obs are seperated here
        # privileged_obs contains information given to privileged encoder
        # critic_obs contains information given to critic, including some privileged information
        # This operation is to prevent the critic from receiving noisy input from the concatenation of current observation(noisy) and latent vector
        num_actions = 6
        env_spacing = 0.5
    
    class terrain( LeggedRobotCfg.terrain ):
        if SIMULATOR == "genesis":
            mesh_type = "heightfield" # for genesis
        else:
            mesh_type = "trimesh"  # for isaacgym
        restitution = 0.
        border_size = 20.0 # [m]
        curriculum = True
        # rough terrain only:
        obtain_terrain_info_around_feet = True
        measure_heights = True
        measured_points_x = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4] # 9x9=81
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4]
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 10  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.2, 0.1, 0.25, 0.25, 0.2]
        
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = {  # target angles when action = 0.0
            "abad_L_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "foot_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_R_Joint": 0.0,
            "foot_R_Joint": 0.0,
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        # control_type = 'P'
        stiffness = {'Joint': 42.}   # [N*m/rad]
        damping = {'Joint': 2.5}     # [N*m*s/rad]
        action_scale = 0.25 # action scale: target angle = actionScale * action + defaultAngle
        decimation = 4 # decimation: Number of control action updates @ sim DT per policy DT
        dt =  0.02  # control frequency 50Hz

    class asset( LeggedRobotCfg.asset ):
        # Common: 
        name = "tron1_pf"
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/PF_TRON1A/urdf/robot.urdf'
        obtain_link_contact_states = True
        contact_state_link_names = ["abad", "hip", "knee", "foot"]
        foot_name = "foot"
        penalize_contacts_on = ["knee", "hip", "base", "abad"]
        terminate_after_contacts_on = ["base"]
        # For Genesis
        dof_names = [           # align with the real robot
            "abad_L_Joint",
            "hip_L_Joint",
            "knee_L_Joint",
            "abad_R_Joint",
            "hip_R_Joint",
            "knee_R_Joint",
        ]
        links_to_keep = ['foot_L_Link', 'foot_R_Link']# Genesis: 
        # IsaacGym:
        flip_visual_attachments = False
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.68
        foot_clearance_target = 0.09 # desired foot clearance above ground [m]
        foot_height_offset = 0.032   # height of the foot coordinate origin above ground [m]
        foot_clearance_tracking_sigma = 0.01
        foot_distance_threshold = 0.115
        only_positive_rewards = False
        class scales( LeggedRobotCfg.rewards.scales ):
            # limitation
            keep_balance = 1.0
            dof_pos_limits = -2.0
            collision = -1.0
            feet_distance = -100.0
            # command tracking
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            # smooth
            lin_vel_z = -0.5
            base_height = -2.0
            ang_vel_xy = -0.05
            orientation = -3.0
            dof_power = -2.e-4
            dof_acc = -2.e-7
            action_rate = -0.01
            action_smoothness = -0.01
            # gait
            feet_air_time = 1.0
            foot_clearance = 0.2
            no_fly = 1.0

    class commands( LeggedRobotCfg.commands ):
        curriculum = True
        max_curriculum = 1.0
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [-0.5, 0.5] # min max [m/s]
            lin_vel_y = [-0.6, 0.6]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]
            
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.2, 1.7]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.
        randomize_com_displacement = True
        com_displacement_range = [-0.03, 0.03]
        randomize_pd_gain = True
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]

class TRON1PF_TSCfgPPO( LeggedRobotCfgPPO ):
    seed = 1
    runner_class_name = "TSRunner" # Teacher-Student Runner
    class policy( LeggedRobotCfgPPO.policy ):
        critic_hidden_dims = [1024, 256, 128]
        privilege_encoder_hidden_dims = [256, 128]
        history_encoder_type = "MLP" # "MLP" or "TCN"
        history_encoder_hidden_dims = [256, 128]       # for MLP
        history_encoder_channel_dims = [1, 1, 1, 1]    # for TCN
        history_encoder_dilation = [1, 1, 2, 1]        # for TCN
        history_encoder_stride = [1, 2, 1, 2]          # for TCN
        history_encoder_final_layer_dim = 128          # for TCN
        kernel_size = 5
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        encoder_lr = 1e-3
        num_encoder_epochs = 2
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticTS"
        algorithm_class_name = "PPO_TS"
        run_name = 'gym_ts_blind'
        experiment_name = 'tron1_pf_rough'
        save_interval = 500
        load_run = "Oct26_16-35-39_gym_ts_blind"
        checkpoint = -1
        max_iterations = 2500