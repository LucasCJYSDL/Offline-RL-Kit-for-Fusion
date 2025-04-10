'''
note: run from home directory:
python examples/run_mppi.py
to specify policy, modify lines with "set policy here"
'''

import argparse
import random
from gym.spaces import Box

import numpy as np
import torch

import sys, os, time
import ray, yaml
from ray.util import inspect_serializability
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.nets import MLP
from offlinerlkit.modules.actor_module import Actor # Specify policy here
from preparation.get_rl_data_envs import get_rl_data_envs
from preparation.process_raw_data import raw_data_dir, general_data_path, tracking_data_path, reference_shot, training_model_dir, evaluation_model_dir, change_every
from envs.base_env import SA_processor, NFBaseEnv
curent_directory = os.getcwd()

def collect_trajectory(env_id, actor, state_mean, state_std):
    offline_data, sa_processor, env, training_dyn_model_dir = get_rl_data_envs(self.env_id, self.task, self.device)
    max_epi_length = env.time_limit
    state = env.reset()
    rwd_trajectory = []

    done = False
    i = 0
    while not done and i < max_epi_length:
        state_tensor = toorch.tensor(np.array([state]), dtype=torch.float32)
        action = actor.forward((state_tensor-state_mean)/state_std)
        next_state, reward, done, _ = env.step(action[0])
        rwd_trajectory.append(reward)
        state = next_state
        i += 1
    return rwd_trajectory

def collect_trajectories(num_envs, envs, actor, state_mean, state_std, max_epi_length):
    trajs = [{'observations': [], 'actions': [], 'rewards': [], 'dones': []} for _ in range(num_envs)]

    state_mean = torch.tensor(state_mean, dtype=torch.float32)
    state_std = torch.tensor(state_std, dtype=torch.float32)

    states = [e.reset()[0] for e in envs]
    mask = np.ones(num_envs, dtype=bool)
    for l in range(max_epi_length - 1):
        states = torch.stack(states, dim=0)
        normed_states = (states-state_mean)/state_std
        actions = actor.forward(normed_states)   
        next_states = []
        rewards = []
        dones = []
        for i,e in enumerate(envs):
            if e.cur_time >= max_epi_length:
                ns = states[i]
                rw = 0
                dn = True
            else:
                ns, rw, dn, _ = e.step(actions[i].reshape((1, actions.shape[-1]))) 
                ns = ns.reshape(states.shape[-1])
            next_states.append(ns)
            rewards.append(rw)
            dones.append(dn)

        for i in range(num_envs):
            if mask[i]:
                trajs[i]['observations'].append(states[i].detach().numpy())
                trajs[i]['actions'].append(actions[i].detach().numpy())
                trajs[i]['rewards'].append(np.array(rewards[i]))
                trajs[i]['dones'].append(np.array(dones[i]))  

        mask = np.logical_and(mask, np.logical_not(dones))
        if not mask.any():
            break

        states = next_states
            
    return trajs

def get_parameter_diff(current_parameters, initial_parameters):
    return {name: param - initial_parameters[name] for name, param in current_parameters.items()}

class DummyVecEnv:
    def __init__(self, num_envs, training_model_dir, device, task, env_id='base'):
        from envs.base_env import NFBaseEnv, SA_processor
        from preparation.get_rl_data_envs import load_offline_data
        offline_data, tracking_data = load_offline_data(env_id, task)
        sa_processor = SA_processor(offline_data, tracking_data, device)
        envs = []
        for _ in range(num_envs):
            env = NFBaseEnv(training_model_dir, sa_processor, offline_data, tracking_data[reference_shot], reference_shot, device)
            envs.append(env)
        self.envs = envs

@ray.remote
class pi2_runner:
    def __init__(self):

        self.algo_name = "pi2"
        self.env_id = "base"  # one of [base, profile_control]
        self.task = "betan_EFIT01"
        self.seed = 1
        self.device = torch.device("cpu")

        # PI2 hyperparameters
        self.noise_scale = 0.1      # control noise
        self.num_samples = 2        # number of environments to parallelly process
        self.hidden_dims = [256, 256, 256]

        self.offline_data_dir = "/home/scratch/linmo/fusion_data/noshape_gas_flat_top/general_data_rl.h5"  # must run from examples folder
        self.rnn_model_dir = "/zfsauton/project/fusion/models/rpnn_noshape_gas_flat_top_step_two_logvar"
        self.yaml_file = "../../FusionControl/cfgs/control/environment/beta_tracking_env.yaml"

        # log
        log_dirs = make_log_dirs(self.task, self.algo_name, self.seed, args=None)
        # key: output file name, value: output handler type
        output_config = {
            "consoleout_backup": "stdout",
            "policy_training_progress": "csv",
            "tb": "tensorboard"
        }
        self.logger = Logger(log_dirs, output_config)
        # logger.log_hyperparameters(vars(args))

        # policy specification
        offline_data, sa_processor, env, training_dyn_model_dir = get_rl_data_envs(self.env_id, self.task, self.device)
        self.training_dyn_model_dir = training_dyn_model_dir
        self.env = env
        self.envs = DummyVecEnv(self.num_samples, training_dyn_model_dir, self.device, self.task, self.env_id).envs
        self.state_dim = offline_data['obs_dim'] + 2 # targets, difference
        self.action_dim = offline_data['actions'].shape[1]
        actor_backbone = MLP(input_dim=np.prod(self.state_dim), hidden_dims=self.hidden_dims)
        self.policy_function = Actor(actor_backbone, self.action_dim) # Specify policy here
        # self.policy_function.update_model(center_policy_function) # TODO: find out what this function does

        offline_data["observations"] = np.array(offline_data["observations"])
        self.state_mean = offline_data["observations"].mean(axis=0)
        self.state_std = offline_data["observations"].std(axis=0)

        print("actor initialized")
    
    def get_policy_parameters(self):
        return {name: param.clone() for name, param in self.policy_function.actor.named_parameters()} # set policy here
        # may need to change .actor name here
    
    def exploration(self, state_mean, state_std, noise_scheme=None):
        # Store the initial parameters of the actor network
        initial_parameters = self.get_policy_parameters()

        # Collect several trajectories
        rwd_trajectories, parameter_dicts, parameter_diff_dicts = [], [], []
        for _ in tqdm(range(self.num_samples)):
            # Perturb the actor network
            with torch.no_grad():
                # for param in self.policy_function.policy_network.parameters():
                #     param += torch.randn_like(param) * self.noise_scale  ## Example perturbation
                ## danger
                for name, param in self.policy_function.actor.named_parameters(): # set policy here
                    if noise_scheme is None:
                        param += torch.randn_like(param) * self.noise_scale
                    else:
                        param += torch.randn_like(param) * torch.sqrt(noise_scheme[name])
            # Collect a trajectory
            rwd_trajectory = collect_trajectory(self.env_id, self.policy_function, self.state_mean, self.state_std)
            rwd_trajectories.append(rwd_trajectory)
            parameter_dicts.append(self.get_policy_parameters())
            parameter_diff_dicts.append(get_parameter_diff(parameter_dicts[-1], initial_parameters))
            # Reset the actor network to its initial parameters
            set_policy_parameters(self.policy_function, initial_parameters)

        return rwd_trajectories, parameter_dicts, parameter_diff_dicts

    def run(self):
        max_epi_length = self.env.time_limit
        trajectories = collect_trajectories(self.num_samples, self.envs, self.policy_function, self.state_mean, self.state_std, max_epi_length)

        self.policy_function = None
        print(type(trajectories), type(trajectories[0]))
        return trajectories


if __name__ == "__main__":
    # import warnings
    # warnings.filterwarnings("ignore")
    ray.init()
    # args = get_args()
    # self.device = torch.device("cpu")

    test_runner = pi2_runner.remote()
    trajectories = ray.get([test_runner.run.remote()])
    print(trajectories[0][0]["rewards"])