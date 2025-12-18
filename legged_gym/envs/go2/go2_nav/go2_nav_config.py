from legged_gym import *
from legged_gym.envs.base.legged_robot_nav_config import LeggedRobotNavCfg, LeggedRobotNavCfgPPO

class GO2NavCfg( LeggedRobotNavCfg ):
    
    class env( LeggedRobotNavCfg.env ):
        num_envs = 4096
        num_observations = 49 + 187
        num_privileged_obs = 68 + 187
        num_actions = 12
        env_spacing = 2.0
        episode_length_s = 8.0
        fail_to_terminal_time_s = 0.1

    class terrain( LeggedRobotNavCfg.terrain ):
        if SIMULATOR == "genesis":
            mesh_type = "heightfield" # for genesis
        else:
            mesh_type = "trimesh"  # for isaacgym
        border_size = 5.0 # [m]
        curriculum = True
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5] # 17x11 = 187 points
        terrain_width = 8.0
        terrain_length = 8.0
        platform_size = 3.0
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, stepping stones, gap, pit]
        terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]

    class init_state( LeggedRobotNavCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,   # [rad]
            'RL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': 0.0 ,  # [rad]
            'RR_hip_joint': 0.0,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 0.8,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 0.8,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }
        roll_random_scale = 0.1
        pitch_random_scale = 0.1
        yaw_random_scale = 3.14

    class control( LeggedRobotNavCfg.control ):
        # PD Drive parameters:
        # control_type = 'P'
        stiffness = {'joint': 20.}   # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        action_scale = 0.25 # action scale: target angle = actionScale * action + defaultAngle
        decimation = 4 # decimation: Number of control action updates @ sim DT per policy DT

    class asset( LeggedRobotNavCfg.asset ):
        # Common
        name = "go2"
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "hip"]
        terminate_after_contacts_on = ["base"]
        # For Genesis
        dof_names = [           # align with the real robot
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint"
        ]
        dof_vel_limits = [30.1, 30.1, 15.7, 
                          30.1, 30.1, 15.7, 
                          30.1, 30.1, 15.7, 
                          30.1, 30.1, 15.7] # [rad/s], corresponds to dof_names order, values from urdf
        links_to_keep = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        # For IsaacGym
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up

    class rewards( LeggedRobotNavCfg.rewards ):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        base_height_target = 0.35
        tracking_duration_pos_s = 4.0         # duration for tracking pos rewards active
        tracking_duration_orientation_s = 6.0 # duration for tracking orientation rewards active
        pos_error_threshold = 2.0 # if the error between target and robot base is lower than this threshold, orientation tracking will be activated
        stall_distance_threshold = 1.0 # [m], min distance to target for stall penalty to be active
        stall_velocity_threshold = 0.1 # [m/s], max xy velocity for stall penalty to be active
        only_positive_rewards = False
        class scales( LeggedRobotNavCfg.rewards.scales ):
            # limitation
            termination = -200.0
            dof_pos_limits = -5.0
            dof_vel_limits = -1.0
            torque_limits = -0.5
            collision = -1.0
            # command tracking
            tracking_target_pos = 10.0
            tracking_target_orientation = 5.0
            # smooth
            base_acc = -4.e-4
            lin_vel_z = -1.0
            dof_power = -4.e-4
            dof_acc = -1.e-7
            action_rate = -0.01
            action_smoothness = -0.01
            # regularization
            feet_stumble = -1.0
            stall = -1.0
            stand_still = -1.0
            move_in_direction = 0.5
    
    class commands( LeggedRobotNavCfg.commands ):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4      # default: position of the target point(x, y) in base frame, heading, time left to target
        default_pos_z = 0.35  # base height relative to the ground
        class ranges( LeggedRobotNavCfg.commands.ranges ):
            pos_x = [0.0, 1.0]      # m, relative to the terrain's border
            pos_y = [-3.0, 3.0]     # m, relative to the terrain's origin
            heading = [-3.14, 3.14] # rad

    class domain_rand( LeggedRobotNavCfg.domain_rand ):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.
        randomize_com_displacement = True
        com_pos_x_range = [-0.03, 0.03]
        com_pos_y_range = [-0.03, 0.03]
        com_pos_z_range = [-0.03, 0.03]
    
    class normalization( LeggedRobotNavCfg.normalization ):
        class obs_scales( LeggedRobotNavCfg.normalization.obs_scales ):
            lin_vel = 1.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            base_pos = 0.2
            base_orientation = 1/3.15
            time_to_target = 0.25

class GO2NavCfgPPO( LeggedRobotNavCfgPPO ):
    class policy (LeggedRobotNavCfgPPO.policy ):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
    class algorithm( LeggedRobotNavCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotNavCfgPPO.runner ):
        num_steps_per_env = 48
        policy_class_name = 'ActorCritic'
        run_name = ''
        experiment_name = 'go2_nav'
        save_interval = 500
        load_run = "Nov07_22-21-20_"
        checkpoint = -1
        max_iterations = 6000