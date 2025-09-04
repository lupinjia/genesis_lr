from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class BipedalWalkerCfg( LeggedRobotCfg ):
    
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_observations = 42
        num_actions = 10
        env_spacing = 3.  # not used with heightfields/trimeshes
    
    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        restitution = 0.
        
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 1.08] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'hip_joint_saggital_right': -0., #-0
            'hip_joint_frontal_right': 0.,
            'hip_joint_transversal_right': 0.,
            'knee_joint_right': -0.,
            'ankle_joint_right': 0.,

            'hip_joint_saggital_left': -0.,
            'hip_joint_frontal_left': 0.,
            'hip_joint_transversal_left': 0.,
            'knee_joint_left': -0.,
            'ankle_joint_left': 0.,
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        # control_type = 'P'
        stiffness = {'hip_joint_saggital': 100.0, 'hip_joint_frontal': 100.0,
                     'hip_joint_transversal': 200., 'knee_joint': 200., 'ankle_joint': 200.}   # [N*m/rad]
        damping = { 'hip_joint_saggital': 3.0, 'hip_joint_frontal': 3.0,
                    'hip_joint_transversal': 6., 'knee_joint': 6., 'ankle_joint': 10.}     # [N*m*s/rad]
        action_scale = 0.25 # action scale: target angle = actionScale * action + defaultAngle
        dt =  0.02  # control frequency 50Hz
        decimation = 4 # decimation: Number of control action updates @ sim DT per policy DT

    class asset( LeggedRobotCfg.asset ):
        name = "bipedal_walker" # consistent with folder name
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/bipedal_walker/urdf/walker3d_hip3d.urdf'
        dof_names = [        # specify yhe sequence of actions
            'hip_joint_saggital_right', #-0
            'hip_joint_frontal_right',
            'hip_joint_transversal_right',
            'knee_joint_right',
            'ankle_joint_right',

            'hip_joint_saggital_left',
            'hip_joint_frontal_left',
            'hip_joint_transversal_left',
            'knee_joint_left',
            'ankle_joint_left']
        foot_name = ["foot"]
        penalize_contacts_on = []
        terminate_after_contacts_on = ["torso", 'thigh','shank']
        links_to_keep = []
        self_collisions = True
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 1.08
        only_positive_rewards = False
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -200.0
            # limitation
            dof_pos_limits = -5.0
            collision = -0.0
            # command tracking
            tracking_lin_vel = 1.0
            tracking_ang_vel = 1.0
            # smooth
            lin_vel_z = -2.0
            base_height = -1.0
            ang_vel_xy = -0.05
            orientation = -0.0
            dof_vel = -0.0
            dof_acc = -2.e-7
            action_rate = -0.01
            torques = -1.e-5
            # gait
            feet_air_time = 1.0
            no_fly = 0.25
            # dof_close_to_default = -0.1
    
    class commands( LeggedRobotCfg.commands ):
        curriculum = True
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [-0.5, 0.5] # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

class BipedalWalkerCfgPPO( LeggedRobotCfgPPO ):
    runner_class_name = "OnPolicyRunner"
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'bipedal_walker'
        save_interval = 100
        load_run = "Dec24_21-23-47_"
        checkpoint = -1
        max_iterations = 2000