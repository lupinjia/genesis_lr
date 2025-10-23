"""
Spawning parallel environments on terrain curriculums, evaluating constraint violations across 10 episodes
"""
from legged_gym import *
import os

from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import torch
from tqdm import trange, tqdm


def evaluate(args):
    if SIMULATOR == "genesis":
        gs.init(
            backend=gs.cpu if args.cpu else gs.gpu,
            logging_level='warning',
        )
    # ---------- Constrained Teacher-Student ----------
    cat_task = "go2_cat" # constrained teacher-student
    env_cfg, train_cfg = task_registry.get_cfgs(name=cat_task)
    env_cfg.env.debug_cstr_violation = True
    # velocity range
    env_cfg.commands.ranges.lin_vel_x = [-1.0, 1.0]
    # prepare environment
    env, _ = task_registry.make_env(name=cat_task, args=args, env_cfg=env_cfg)
    env.reset()
    obs_buf, privileged_obs_buf, obs_history, critic_obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=cat_task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    # interaction loop for constrained teacher-student
    print("Evaluating Constrained Teacher-Student Policy...")
    for ep in trange(10, desc="Episodes"):
        for step in trange(int(env.max_episode_length), desc="Steps", leave=False):
            actions = policy(obs_buf, obs_history)
            obs_buf, privileged_obs_buf, obs_history, critic_obs, rews, dones, infos = env.step(actions.detach())
            
            if ep == 9 and step == int(env.max_episode_length) - 1:  # last episode
                tqdm.write("")
                tqdm.write("Constrained Teacher-Student Policy")
                tqdm.write(f"----mean terrain level: {torch.mean(env.simulator.terrain_levels.float())}")
                tqdm.write("----summed constraint violation across 10 episodes:")
                for name, violations in env.cstr_violation.items():
                    tqdm.write(f"--------{name}: {violations}")
    

if __name__ == '__main__':
    args = get_args()
    evaluate(args)
