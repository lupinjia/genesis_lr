from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

class GO2SysIDCfg( LeggedRobotCfg ):
    
    class env( LeggedRobotCfg.env ):
        num_envs = 6000
        num_actions = 12
        env_spacing = 0.5
    
    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = "plane" # none, plane, heightfield
        restitution = 0.
        
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
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
        dt =  0.02  # control frequency 50Hz
        decimation = 4 # decimation: Number of control action updates @ sim DT per policy DT

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        dof_names = [        # specify the sequence of actions, align with the order in data
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
        foot_name = ["foot"]
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        links_to_keep = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        self_collisions = True
        fix_base_link = True
    
    class domain_rand( LeggedRobotCfg.domain_rand ):
        randomize_friction = False
        friction_range = [0.2, 1.7]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 1.
        randomize_com_displacement = False
        com_displacement_range = [-0.03, 0.03]
        randomize_ctrl_delay = False
        randomize_joint_armature = False
        joint_armature_range = [0.0, 0.05]  # [N*m*s/rad]
        randomize_joint_stiffness = False
        joint_stiffness_range = [0.0, 0.1]
        randomize_joint_damping = False
        joint_damping_range = [0.0, 1.0]
        delay_steps = 0
    
    class sysid_data:
        file = "{LEGGED_GYM_ROOT_DIR}/resources/sysid/20250617_motor_response_real_200Hz.csv"
    
    class sysid_param_range:
        
        # without sysid params
        # joint_stiffness_range = [0.0, 0.0]
        # joint_damping_range = [0.0, 0.0]
        # joint_armature_range = [0.0, 0.0]
        
        # with sysid params
        joint_stiffness_range = [0.0, 0.1]
        joint_damping_range = [0.0, 1.0]
        joint_armature_range = [0.0, 0.1]
        
        kp_range = [20.0, 20.0]
        kd_range = [0.5, 0.5]