from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2Cfg( LeggedRobotCfg ):
    
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_observations = 45 # 48 for only sim, 45 for deployment
        num_privileged_obs = None
        num_actions = 12
        env_spacing = 2.0
    
    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = "plane" # none, plane, heightfield
        
    class init_state( LeggedRobotCfg.init_state ):
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

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        # control_type = 'P'
        stiffness = {'joint': 20.}   # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        action_scale = 0.25 # action scale: target angle = actionScale * action + defaultAngle
        decimation = 4 # decimation: Number of control action updates @ sim DT per policy DT

    class asset( LeggedRobotCfg.asset ):
        # Common
        name = "go2"
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
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
        links_to_keep = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        # For IsaacGym
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.36
        foot_clearance_target = 0.05 # desired foot clearance above ground [m]
        foot_height_offset = 0.022   # height of the foot coordinate origin above ground [m]
        foot_clearance_tracking_sigma = 0.01
        only_positive_rewards = True
        class scales( LeggedRobotCfg.rewards.scales ):
            # limitation
            dof_pos_limits = -10.0
            collision = -1.0
            # command tracking
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            # smooth
            lin_vel_z = -0.5
            base_height = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            dof_vel = -5.e-4
            dof_acc = -2.e-7
            action_rate = -0.01
            action_smoothness = -0.01
            torques = -2.e-4
            # gait
            feet_air_time = 1.0
            foot_clearance = 0.5
    
    class commands( LeggedRobotCfg.commands ):
        curriculum = True
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [-0.5, 0.5] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class domain_rand( LeggedRobotCfg.domain_rand ):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.
        randomize_com_displacement = True
        com_pos_x_range = [-0.01, 0.01]
        com_pos_y_range = [-0.01, 0.01]
        com_pos_z_range = [-0.01, 0.01]

class GO2CfgPPO( LeggedRobotCfgPPO ):
    class policy (LeggedRobotCfgPPO.policy ):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCritic'
        run_name = 'with_lin'
        experiment_name = 'go2'
        save_interval = 200
        load_run = "Sep12_17-27-42_without_lin"
        checkpoint = -1
        max_iterations = 1400