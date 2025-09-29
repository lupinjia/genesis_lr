import os
import copy
import torch
import numpy as np
import random
import argparse

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

def class_to_dict(obj) -> dict:
    if not hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        #TODO sort by date to handle change of month
        runs.sort()
        if 'exported' in runs: runs.remove('exported')
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run==-1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint==-1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(load_run, model)
    return load_path

def get_load_path_ee(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        #TODO sort by date to handle change of month
        runs.sort()
        if 'exported' in runs: runs.remove('exported')
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run==-1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint==-1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        # estimator
        estimators = [file for file in os.listdir(load_run) if 'estimator' in file]
        estimators.sort(key=lambda m: '{0:0>15}'.format(m))
        estimator = estimators[-1]
    else:
        model = "model_{}.pt".format(checkpoint)
        estimator = "estimator_{}.pt".format(checkpoint)

    actor_load_path = os.path.join(load_run, model)
    estimator_load_path = os.path.join(load_run, estimator)
    return actor_load_path, estimator_load_path

def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
    if cfg_train is not None:
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume

    return env_cfg, cfg_train

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',           type=str, default='go2')
    parser.add_argument('--headless',       action='store_true', default=False)  # enable visualization by default
    parser.add_argument('-c', '--cpu',      action='store_true', default=False)  # use cuda by default
    parser.add_argument('-B', '--num_envs', type=int, default=None)
    parser.add_argument('--max_iterations', type=int, default=None)
    parser.add_argument('--resume',         action='store_true', default=False)
    parser.add_argument('-o', '--offline',  action='store_true', default=False)

    parser.add_argument('--debug',          action='store_true', default=False)
    parser.add_argument('--ckpt',           type=int, default=1000)

    return parser.parse_args()

def export_policy_as_jit(actor_critic, path, prefix=None, export_type=None):
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else: 
        os.makedirs(path, exist_ok=True)
        if export_type == "ts":
            model_path = os.path.join(path, prefix + "_policy.pt")
            model = copy.deepcopy(actor_critic.actor).to('cpu')
            traced_script_module = torch.jit.script(model)
            traced_script_module.save(model_path)
            encoder_path = os.path.join(path, prefix + "_encoder.pt")
            model = copy.deepcopy(actor_critic.history_encoder).to('cpu')
            traced_script_module = torch.jit.script(model)
            traced_script_module.save(encoder_path)
        elif export_type == "ee":
            model_path = os.path.join(path, prefix + "_policy.pt")
            model = copy.deepcopy(actor_critic.actor).to('cpu')
            traced_script_module = torch.jit.script(model)
            traced_script_module.save(model_path)
            estimator_path = os.path.join(path, prefix + "_estimator.pt")
            model = copy.deepcopy(actor_critic.estimator).to('cpu')
            traced_script_module = torch.jit.script(model)
            traced_script_module.save(estimator_path)
        else:
            path = os.path.join(path, prefix + '.pt')
            model = copy.deepcopy(actor_critic.actor).to('cpu')
            traced_script_module = torch.jit.script(model)
            traced_script_module.save(path)

class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

    
