import os

from legged_gym import *
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import shutil

def train(args):
    if SIMULATOR == "genesis":
        gs.init(
            backend=gs.cpu if args.cpu else gs.gpu,
            logging_level='warning')
    # Make environment and algorithm runner
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    
    # Copy env.py and env_config.py to log_dir for backup
    log_dir = ppo_runner.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if env_cfg.asset.name == args.task:
        robot_file_path = os.path.join(LEGGED_GYM_ROOT_DIR, "legged_gym", "envs", env_cfg.asset.name, args.task+".py")
        robot_config_path = os.path.join(LEGGED_GYM_ROOT_DIR, "legged_gym", "envs", env_cfg.asset.name, args.task+"_config.py")
    else:
        robot_file_path = os.path.join(LEGGED_GYM_ROOT_DIR, "legged_gym", "envs", env_cfg.asset.name, args.task, args.task+".py")
        robot_config_path = os.path.join(LEGGED_GYM_ROOT_DIR, "legged_gym", "envs", env_cfg.asset.name, args.task, args.task+"_config.py")
    shutil.copy(robot_file_path, log_dir)
    shutil.copy(robot_config_path, log_dir)
    
    # Start training session
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    if args.debug:
        args.offline = True
        args.num_envs = 1
    train(args)
