from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# TRON1 Sole Foot
class TRON1SFCfg( LeggedRobotCfg ):
    
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_single_obs = 33
        frame_stack = 5    # Policy frame stack number
        c_frame_stack = 5  # Critic frame stack number
        num_observations = int( num_single_obs * frame_stack )
        num_single_privileged_obs = num_single_obs + 20
        num_privileged_obs = int( num_single_privileged_obs * c_frame_stack )
        num_actions = 8
        env_spacing = 2.0
    
    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = "plane" # none, plane, heightfield
        
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8]     # x,y,z [m]
        default_joint_angles = {  # target angles when action = 0.0
            "abad_L_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "ankle_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_R_Joint": 0.0,
            "ankle_R_Joint": 0.0
        }
        init_stand_joint_angles = {
            "abad_L_Joint": 0.0,
            "hip_L_Joint": 0.58,
            "knee_L_Joint": 1.35,
            "ankle_L_Joint": -0.8,
            "abad_R_Joint": 0.0,
            "hip_R_Joint": -0.58,
            "knee_R_Joint": -1.35,
            "ankle_R_Joint": 0.8
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        # control_type = 'P'
        stiffness = {
            "abad_L_Joint": 45,
            "hip_L_Joint": 45,
            "knee_L_Joint": 45,
            "abad_R_Joint": 45,
            "hip_R_Joint": 45,
            "knee_R_Joint": 45,

            "ankle_L_Joint": 45,
            "ankle_R_Joint": 45,
        }  # [N*m/rad]
        damping = {
            "abad_L_Joint": 1.5,
            "hip_L_Joint": 1.5,
            "knee_L_Joint": 1.5,
            "abad_R_Joint": 1.5,
            "hip_R_Joint": 1.5,
            "knee_R_Joint": 1.5,

            "ankle_L_Joint": 0.8,
            "ankle_R_Joint": 0.8,
        }  # [N*m*s/rad]
        action_scale = 0.25 # action scale: target angle = actionScale * action + defaultAngle
        decimation = 4      # decimation: Number of control action updates @ sim DT per policy DT

    class asset( LeggedRobotCfg.asset ):
        # Common
        name = "tron1_sf"
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/SF_TRON1A/urdf/robot.urdf'
        foot_name = "ankle"
        penalize_contacts_on = ["knee", "hip"]
        terminate_after_contacts_on = ["base", "abad"]
        # For Genesis
        dof_names = [           # align with the real robot
            "abad_L_Joint",
            "hip_L_Joint",
            "knee_L_Joint",
            "ankle_L_Joint",
            "abad_R_Joint",
            "hip_R_Joint",
            "knee_R_Joint",
            "ankle_R_Joint",
        ]
        links_to_keep = []
        # For IsaacGym
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.68
        foot_clearance_target = 0.07 # desired foot clearance above ground [m]
        foot_height_offset = 0.06   # height of the foot coordinate origin above ground [m]
        foot_clearance_tracking_sigma = 0.01
        foot_distance_threshold = 0.115
        about_landing_threshold = 0.1
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
            orientation = -1.0
            dof_vel = -5.e-4
            dof_acc = -2.e-7
            action_rate = -0.01
            action_smoothness = -0.01
            torques = -2.e-5
            # gait
            feet_air_time = 1.0
            foot_clearance = 0.5
            no_fly = 0.5
            foot_landing_vel = -0.15
    
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
        com_pos_x_range = [-0.03, 0.03]
        com_pos_y_range = [-0.03, 0.03]
        com_pos_z_range = [-0.03, 0.03]

class TRON1SFCfgPPO( LeggedRobotCfgPPO ):
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
        run_name = 'gs_flat'
        experiment_name = 'tron1_sf'
        save_interval = 500
        load_run = "Nov04_09-44-36_gs_flat"
        max_iterations = 2500