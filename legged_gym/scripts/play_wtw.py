import genesis as gs
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, QuadLogger

import numpy as np
import torch


def play(args):
    gs.init(
        backend=gs.cpu if args.cpu else gs.gpu,
        logging_level='warning',
    )
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    env_cfg.viewer.rendered_envs_idx = list(range(env_cfg.env.num_envs))
    for i in range(2):
        env_cfg.viewer.pos[i] = env_cfg.viewer.pos[i] - env_cfg.terrain.plane_length / 4
        env_cfg.viewer.lookat[i] = env_cfg.viewer.lookat[i] - env_cfg.terrain.plane_length / 4
    if FOLLOW_ROBOT:
        env_cfg.viewer.add_camera = True  # use a extra camera for moving
    env_cfg.terrain.border_size = 5
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.selected = False
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.enable = True
    env_cfg.asset.fix_base_link = False
    env_cfg.rewards.behavior_params_range.gait_period_range = [0.5, 0.5]
    env_cfg.rewards.behavior_params_range.base_height_target_range = [0.35, 0.35]
    env_cfg.rewards.behavior_params_range.foot_clearance_target_range = [0.03, 0.03]
    env_cfg.rewards.behavior_params_range.pitch_target_range = [0.0, 0.0]
    env_cfg.rewards.periodic_reward_framework.theta_fl_list = [0.0]
    env_cfg.rewards.periodic_reward_framework.theta_fr_list = [0.0]
    env_cfg.rewards.periodic_reward_framework.theta_rl_list = [0.5]
    env_cfg.rewards.periodic_reward_framework.theta_rr_list = [0.5]
    # velocity range
    env_cfg.commands.ranges.lin_vel_x = [-1.0, 1.0]

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, train_cfg.runner.load_run, 'exported')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path, train_cfg.runner.load_run)
        print('Exported policy as jit script to: ', path)

    logger = QuadLogger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 2 # which joint is used for logging
    stop_state_log = 300 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    
    # for FOLLOW_ROBOT
    camera_lookat_follow = np.array(env_cfg.viewer.lookat)
    camera_deviation_follow = np.array([0., 3., -1.])
    camera_position_follow = camera_lookat_follow - camera_deviation_follow

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        # print(f"base height: {env.simulator.base_pos[robot_index, 2].item()}")
        print(f"foot height: {env.simulator.feet_pos[robot_index, :, 2]}")
        if FOLLOW_ROBOT:
            # refresh where camera looks at(robot 0 base)
            camera_lookat_follow = env.simulator.base_pos[robot_index, :].cpu().numpy()
            # refresh camera's position
            camera_position_follow = camera_lookat_follow - camera_deviation_follow
            env.set_camera(camera_position_follow, camera_lookat_follow)
            env.floating_camera.render()

        if i < stop_state_log:
            logger.log_states(
                {
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.simulator.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.simulator.base_lin_vel[robot_index, 1].item(),
                    'base_vel_yaw': env.simulator.base_ang_vel[robot_index, 2].item(),
                    'exp_C_frc_fl': env.exp_C_frc_fl[robot_index, 0].item(),
                    'exp_C_frc_fr': env.exp_C_frc_fr[robot_index, 0].item(),
                    'exp_C_frc_rl': env.exp_C_frc_rl[robot_index, 0].item(),
                    'exp_C_frc_rr': env.exp_C_frc_rr[robot_index, 0].item(),
                    'contact_forces_fl': env.simulator.link_contact_forces[robot_index, env.simulator.feet_indices[0], 2].cpu().numpy(),
                    'contact_forces_fr': env.simulator.link_contact_forces[robot_index, env.simulator.feet_indices[1], 2].cpu().numpy(),
                    'contact_forces_rl': env.simulator.link_contact_forces[robot_index, env.simulator.feet_indices[2], 2].cpu().numpy(),
                    'contact_forces_rr': env.simulator.link_contact_forces[robot_index, env.simulator.feet_indices[3], 2].cpu().numpy(),
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
            # logger.save_data_to_xlsx()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = True
    FOLLOW_ROBOT  = False
    args = get_args()
    play(args)
