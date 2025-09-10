from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class GO2WTWCfg(LeggedRobotCfg):

    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_actions = 12
        # observation history
        frame_stack = 5   # policy frame stack
        c_frame_stack = 5  # critic frame stack
        num_single_obs = 57
        num_observations = int(num_single_obs * frame_stack)
        single_num_privileged_obs = 102
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        env_spacing = 1.0

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'  # "heightfield" # none, plane, heightfield or trimesh

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,   # [rad]
            'RL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': 0.0,  # [rad]
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

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        # control_type = 'P'
        stiffness = {'joint': 20.}   # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        action_scale = 0.25  # action scale: target angle = actionScale * action + defaultAngle
        decimation = 4  # decimation: Number of control action updates @ sim DT per policy DT

    class asset(LeggedRobotCfg.asset):
        # Common
        name = "go2" # name of the robot
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        # For Genesis
        dof_names = [        # specify the sequence of actions, keep consistent with IsaacGym
            'FL_hip_joint',
            'FL_thigh_joint',
            'FL_calf_joint',
            'FR_hip_joint',
            'FR_thigh_joint',
            'FR_calf_joint',
            'RL_hip_joint',
            'RL_thigh_joint',
            'RL_calf_joint',
            'RR_hip_joint',
            'RR_thigh_joint',
            'RR_calf_joint',]
        links_to_keep = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        self_collisions_gs = True
        # For IsaacGym
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_tracking_sigma = 0.01
        foot_height_offset = 0.022 # height of the foot coordinate origin above ground [m]
        foot_clearance_tracking_sigma = 0.01
        euler_tracking_sigma = 0.1
        about_landing_threshold = 0.03
        only_positive_rewards = True

        class scales(LeggedRobotCfg.rewards.scales):
            # limitation
            dof_pos_limits = -10.0
            collision = -1.0
            # command tracking
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            tracking_base_height = 0.6
            orientation = 0.6
            foot_clearance = 0.6
            quad_periodic_gait = 1.0
            # smooth
            lin_vel_z = -0.5
            ang_vel_xy = -0.05
            dof_vel = -5.e-4
            dof_acc = -2.e-7
            action_rate = -0.01
            action_smoothness = -0.01
            torques = -2.e-4
            foot_landing_vel = -0.15
            hip_pos = -1.0
            
        class periodic_reward_framework:
            '''Periodic reward framework in OSU's paper(https://arxiv.org/abs/2011.01387)'''
            gait_function_type = "step" # can be "step" or "smooth"
            kappa = 20
            # start of swing is all the same
            b_swing = 0.5
            theta_fl_list = [0.0, 0.0, 0.5]  # front left leg
            theta_fr_list = [0.5, 0.0, 0.0]
            theta_rl_list = [0.5, 0.5, 0.5]
            theta_rr_list = [0.0, 0.5, 0.0]
        
        class behavior_params_range:
            resampling_time = 6.0
            gait_period_range = [0.3, 0.6]
            foot_clearance_target_range = [0.03, 0.12]
            base_height_target_range = [0.2, 0.36]
            pitch_target_range = [-0.3, 0.3]
            
    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 1.
        # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.5, 0.5]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class domain_rand(LeggedRobotCfg.domain_rand):
        enable = True
        randomize_friction = enable
        friction_range = [0.2, 1.7]
        randomize_base_mass = enable
        added_mass_range = [-1., 1.]
        push_robots = enable
        push_interval_s = 15
        max_push_vel_xy = 1.
        randomize_com_displacement = enable
        com_displacement_range = [-0.03, 0.03]
        randomize_ctrl_delay = False
        ctrl_delay_step_range = [0, 1]
        randomize_pd_gain = enable
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]
        randomize_joint_armature = enable
        joint_armature_range = [0.015, 0.025]  # [N*m*s/rad]
        randomize_joint_stiffness = enable
        joint_stiffness_range = [0.01, 0.02]
        randomize_joint_damping = enable
        joint_damping_range = [0.25, 0.3]

    class normalization(LeggedRobotCfg.normalization):
        clip_observations = 20.
        clip_actions = 10.
    
    class noise(LeggedRobotCfg.noise):
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.03
            dof_vel = 0.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

class GO2WTWCfgPPO(LeggedRobotCfgPPO):
    seed = 0
    runner_class_name = "OnPolicyRunner"

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'step_gait'
        experiment_name = 'go2_wtw'
        save_interval = 500
        load_run = "Aug30_12-04-17_smooth_gait"
        checkpoint = -1
        max_iterations = 4000
