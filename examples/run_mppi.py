# note: run from home directory:
# python examples/run_mppi.py

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
from preparation.get_rl_data_envs import get_rl_data_envs
from preparation.process_raw_data import raw_data_dir, general_data_path, tracking_data_path, reference_shot, training_model_dir, evaluation_model_dir, change_every
from envs.base_env import SA_processor, NFBaseEnv
curent_directory = os.getcwd()

def get_args():
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="base") # one of [base, profile_control]
    parser.add_argument("--task", type=str, default="betan_EFIT01")
    parser.add_argument("--seed", type=int, default=1)
    
    # parser.add_argument("--eval_episodes", type=int, default=10)
    # parser.add_argument("--batch-size", type=int, default=256)
    # parser.add_argument("--cuda_id", type=int, default=1)

    # MPPI hyperparameters 
    # TODO: confirm with Jiayu or Rohit if these are reasonable parameters, figure out param search space
    parser.add_argument("--num_samples", help="number of trajectory samples", type=int, default=10)
    parser.add_argument("--horizon", help="planning horizon steps", type=int, default=2)
    parser.add_argument("--temperature", help="weighting temperature param", type=float, default=2.0)
    parser.add_argument("--noise_var", help="control noise variance", type=int, default=1)
    parser.add_argument("--time_step", help="initial time step", type=int, default=0)
    parser.add_argument("--num_envs", help="number of environments to parallelly process", type=int, default=1) # make sure mod num_samples = 0

    parser.add_argument("--offline_data_dir", type=str, default='/home/scratch/linmo/fusion_data/noshape_gas_flat_top/general_data_rl.h5') # must run from the examples folder
    parser.add_argument('--rnn_model_dir', type=str, default='/zfsauton/project/fusion/models/rpnn_noshape_gas_flat_top_step_two_logvar') #?
    parser.add_argument("--yaml_file", type=str, default="../../FusionControl/cfgs/control/environment/beta_tracking_env.yaml")

    args, _ = parser.parse_known_args()
    return parser.parse_args()

def mppi(init_state, 
        U_init, 
        env, 
        envs, 
        num_envs, 
        time_limit, 
        num_samples, 
        horizon, 
        lam, 
        state_dim, 
        action_dim, 
        action_low, 
        action_high, 
        sigma, 
        temp,
        logger=None):
    start_time = time.time()
    num_timesteps = 0
    traj = {'observations': [], 'actions': [], 'rewards': [], 'dones': []}
    env_state = env.reset() # get initial state
    u = U_init.copy()[:horizon]

    env_return = 0.0

    # main loop
    for t_step in tqdm(range(time_limit)):
        # generate K random control sequences
        traj['observations'].append(env_state.cpu().numpy())
        noise = np.random.normal(loc=0.0, scale=sigma, size=(num_samples, horizon, action_dim))
        actions = u[None, :, :] + noise
        actions = np.clip(actions, action_low, action_high)

        
        # simulate trajectories
        num_batches = num_samples // num_envs
        cur_state = env.cur_state
        pre_action = env.pre_action
        cur_time = env.cur_time
        total_cost = np.zeros(num_samples, dtype=np.float32)
        dones = np.zeros(num_samples, dtype=np.float32)
        final_states = np.zeros((num_samples, state_dim), dtype=np.float32)

        for batch_idx in range(num_batches):
            # reset environment and set initial states
            for e in envs:
                _ = e.reset()
                e.cur_state = cur_state
                e.pre_action = pre_action
                e.cur_time = cur_time

            mask = np.ones(num_envs, dtype=bool)
            s_id, e_id = batch_idx * num_envs, (batch_idx + 1) * num_envs
            batch_actions = actions[s_id:e_id]
            for h in range(horizon):
                num_timesteps += 1
                a = batch_actions[:, h, :]
                s = np.zeros((num_envs, state_dim))
                r = np.zeros((num_envs,))
                d = np.zeros((num_envs,))
                for i, e in enumerate(envs):
                    state, reward, is_done, _ = e.step(a[i].reshape((1, action_dim)))
                    s[i] = state.cpu()
                    r[i] = reward
                    d[i] = is_done
                final_states[s_id : e_id][mask] = s[mask]
                total_cost[s_id : e_id][mask] -= r[mask]
                dones[s_id : e_id][mask] = d[mask]
                # once an env terminates, it woul dbe masked until the end of the horizon
                mask = np.logical_and(mask, np.logical_not(d))
                if not mask.any():
                    break

        # Note: can use value function here to estimate costs too
        
        # compute weights
        min_cost = np.min(total_cost)
        weights = np.exp(-(total_cost - min_cost) / temp)
        weights = weights / np.sum(weights)

        # update nominal control seq
        weighted_noise = np.einsum('i,ijk->jk', weights, noise)
        u += weighted_noise
        u = np.clip(u, action_low, action_high)

        # apply action
        action_to_take = u[0].reshape((1, action_dim))
        # assert action_to_take.shape == (1, 5)
        env_state, env_reward, env_done, _ = env.step(action_to_take)
        env_return += env_reward
        traj['actions'].append(action_to_take)
        traj['rewards'].append(env_reward)
        traj['dones'].append(env_done)

        if logger:
            logger.logkv("eval/episode_reward", env_reward)
            logger.logkv("eval/episode_return", env_return)
            logger.set_timestep(num_timesteps)
            logger.dumpkvs()

        if env_done:
            break

        u = np.roll(u, -1, axis=0) # delete first, duplicate last
        if t_step + horizon < len(U_init):
            u[-1] = U_init[t_step + horizon]
        else:
            u[-1] = 0.0 # danger danger

    if logger:
        logger.log("total time: {:.2f}s".format(time.time() - start_time))
        logger.close()

    print(f"The final return: {env_return}")

    # post process of the trajectory
    for k in traj:
        traj[k] = np.array(traj[k])
    
    return traj

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

# @ray.remote(num_gpus=1)
@ray.remote
class mppi_runner:
    def __init__(self):
        # self.device = torch.device("cuda:{}".format(self.cuda_id) if torch.cuda.is_available() else "cpu")
        
        self.algo_name = "mppi"
        self.env_id = "base"  # one of [base, profile_control]
        self.task = "betan_EFIT01"
        self.seed = 1
        self.device = torch.device("cpu")

        # MPPI hyperparameters
        self.num_samples = 10  # number of trajectory samples
        self.horizon = 2       # planning horizon steps
        self.temperature = 2.0  # weighting temperature param
        self.noise_var = 1      # control noise variance
        self.time_step = 0      # initial time step
        self.num_envs = 1       # number of environments to parallelly process

        self.offline_data_dir = "/home/scratch/linmo/fusion_data/noshape_gas_flat_top/general_data_rl.h5"  # must run from examples folder
        self.rnn_model_dir = "/zfsauton/project/fusion/models/rpnn_noshape_gas_flat_top_step_two_logvar"
        self.yaml_file = "../../FusionControl/cfgs/control/environment/beta_tracking_env.yaml"

        print("actor initialized")
        # self.value = 42
        offline_data, sa_processor, env, training_dyn_model_dir = get_rl_data_envs(self.env_id, self.task, self.device)
        self.env = env
        # evaluation model environment(s)
        assert self.num_samples % self.num_envs == 0
        self.envs = DummyVecEnv(self.num_envs, training_dyn_model_dir, self.device, self.task, env_id=self.env_id).envs
        self.next_obs_shape = (offline_data['next_observations'].shape[1], )
        self.obs_shape = (offline_data['observations'].shape[1], )
        self.action_dim = offline_data['actions'].shape[1]
        self.max_action = 1.0
        action_space = Box(low=-self.max_action, high=self.max_action, shape=(self.action_dim, ), dtype=np.float32)

         # params for MPPI
        self.num_samples = self.num_samples # num trajectory samples
        self.num_envs = self.num_envs
        self.horizon = self.horizon # planning horizon steps
        self.temp = self.temperature # temp parameter for weighting
        self.sigma = self.noise_var # control noise variance
        dt = self.time_step # time step, default = 0
        self.action_dim = self.action_dim
        self.time_limit = offline_data["tracking_ref"].shape[0]
        self.state_dim = offline_data['obs_dim'] + 2 # targets, difference
        self.action_low = offline_data['action_lower_bounds']
        self.action_high = offline_data['action_upper_bounds']

        # nominal control sequence
        U_nominal = np.zeros((self.horizon, self.action_dim))

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

    def run(self):

        try:
            print("Run started.")

            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = ""  # disables CUDA entirely (no cuDNN)

            import torch, random, numpy as np

            torch.backends.cudnn.enabled = False  # fully disable cuDNN
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True

            U_init_pad = np.zeros((self.horizon, self.action_dim))
            U_init = None #np.zeros((horizon, action_dim)) # TODO: optimize initialization method
            if U_init is not None: 
                U_init_pad[:len(U_init)] += U_init.copy()
            U_init = U_init_pad
            noise_covariance = np.array([np.eye(self.action_dim) for _ in range(self.horizon)])

            traj = mppi(init_state=None, 
                        U_init=U_init, 
                        env=self.env, 
                        envs=self.envs, 
                        num_envs=self.num_envs, 
                        time_limit=self.time_limit, 
                        num_samples=self.num_samples, 
                        horizon=self.horizon, 
                        lam=self.temp, 
                        state_dim=self.state_dim, 
                        action_dim=self.action_dim, 
                        action_low=self.action_low, 
                        action_high=self.action_high,
                        sigma=self.sigma,
                        temp=self.temp,
                        logger=self.logger)
            return traj

        except Exception as e:
            import traceback
            print("Exception:", traceback.format_exc())
            raise e


if __name__ == "__main__":
    # import warnings
    # warnings.filterwarnings("ignore")
    ray.init()
    # args = get_args()
    # self.device = torch.device("cpu")

    test_runner = mppi_runner.remote()
    new_trajectories = ray.get([test_runner.run.remote()])
    print(new_trajectories["rewards"][0])

    # debug
    # args = get_args()
    # # # self.device = torch.device("cuda:{}".format(self.cuda_id) if torch.cuda.is_available() else "cpu")
    # # # TODO: ask gpt
    # self.device = torch.device("cpu")
    # offline_data, sa_processor, env, training_dyn_model_dir = get_rl_data_envs(self.env, self.task, self.device)
    # inspect_serializability(mppi_runner.run(), name="mppi_runner")


    # no ray
    # model = mppi_runner
    # model.run()