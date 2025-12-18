from legged_gym import *
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class TRON1PF_EECfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_single_obs = 31  # number of elements in single step observation
        frame_stack = 20     # number of frames to stack for obs_history
        num_estimator_features = int(num_single_obs * frame_stack) # dim of input of estimator
        num_estimator_labels = 13 # dim of output of estimator
        c_frame_stack = 5         # number of frames to stack for critic input
        single_critic_obs_len = num_single_obs + 22 + 81 + 8 + 2 # number of elements in single step critic observation
        num_privileged_obs = c_frame_stack * single_critic_obs_len
        # privileged_obs here is actually critic_obs
        num_actions = 6
        env_spacing = 0.5
    
    class terrain( LeggedRobotCfg.terrain ):
        if SIMULATOR == "genesis":
            mesh_type = "heightfield" # for genesis
        else:
            mesh_type = "trimesh"  # for isaacgym
        restitution = 0.
        border_size = 10.0 # [m]
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
        # sit mode pose
        sit_pos = [0.0, 0.0, 0.55] # x,y,z [m]
        sit_joint_angles = {  # target angles when action = 0.0
            "abad_L_Joint": 0.0,
            "hip_L_Joint": 0.6,
            "knee_L_Joint": 1.36,
            "foot_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_R_Joint": -0.6,
            "knee_R_Joint": -1.36,
            "foot_R_Joint": 0.0,
        }
        sit_pitch_angle = -0.2
        sit_init_percent = 0.5  # probability of resetting env in sit pose

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        # control_type = 'P'
        stiffness = {'Joint': 42.}   # [N*m/rad]
        damping = {'Joint': 2.5}     # [N*m*s/rad]
        action_scale = 0.5 # action scale: target angle = actionScale * action + defaultAngle
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
        base_height_target = 0.7
        foot_clearance_target = 0.06 # desired foot clearance above ground [m]
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
            base_height = -1.0
            ang_vel_xy = -0.05
            orientation = -5.0
            dof_power = -2.e-4
            dof_acc = -2.e-7
            # foot_acc = -1.e-5
            action_rate = -0.01
            action_smoothness = -0.01
            # gait
            biped_periodic_gait = 0.3
            foot_clearance = 0.5
        
        class periodic_reward_framework:
            '''Periodic reward framework in OSU's paper(https://arxiv.org/abs/2011.01387)'''
            gait_function_type = "step" # can be "step" or "smooth"
            kappa = 20
            # start of swing(a_swing) is all the same
            b_swing = 0.5
            # phase offset of left and right legs
            theta_left = 0.0
            theta_right = 0.5
            gait_period = 0.5  # [s]

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
        friction_range = [0.0, 1.7]
        randomize_base_mass = True
        added_mass_range = [-1., 2.]
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 1.
        randomize_com_displacement = True
        com_pos_x_range = [-0.03, 0.03]
        com_pos_y_range = [-0.03, 0.03]
        com_pos_z_range = [-0.03, 0.03]
        randomize_pd_gain = True
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]
        randomize_joint_armature = True
        joint_armature_range = [0.11, 0.13]
        randomize_joint_friction = True
        joint_friction_range = [0.00, 0.01]
        randomize_joint_damping = True
        joint_damping_range = [1.4, 1.45]

class TRON1PF_EECfgPPO( LeggedRobotCfgPPO ):
    seed = 1
    runner_class_name = "EERunner" # Teacher-Student Runner
    class policy( LeggedRobotCfgPPO.policy ):
        init_noise_std = 0.5
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [1024, 256, 128]
        estimator_hidden_dims = [256, 128]
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        estimator_lr = 5.e-4
        num_estimator_epochs = 1
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticEE"
        algorithm_class_name = "PPO_EE"
        if SIMULATOR == "genesis":
            run_name = "gs_ee"
        else:
            run_name = 'gym_ee'
        experiment_name = 'tron1_pf_rough'
        save_interval = 500
        load_run = "Dec18_18-44-21_gym_ee"
        checkpoint = -1
        max_iterations = 4000