import argparse
import random
from gym.spaces import Box

import numpy as np
import torch

import sys, os, time
import ray, yaml
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from offlinerlkit.utils.logger import Logger, make_log_dirs
from preparation.get_rl_data_envs import get_rl_data_envs
from preparation.process_raw_data import raw_data_dir, general_data_path, tracking_data_path, reference_shot, training_model_dir, evaluation_model_dir, change_every
from envs.base_env import SA_processor, NFBaseEnv
curent_directory = os.getcwd()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="cql")
    parser.add_argument("--env", type=str, default="base") # one of [base, profile_control]
    parser.add_argument("--task", type=str, default="betan_EFIT01")
    parser.add_argument("--seed", type=int, default=1)
    
    # parser.add_argument("--eval_episodes", type=int, default=10)
    # parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cuda_id", type=int, default=1)

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
        total_cost = np.zeros(num_samples, dtype=np.float32)
        dones = np.zeros(num_samples, dtype=np.float32)
        final_states = np.zeros((num_samples, state_dim), dtype=np.float32)

        for batch_idx in range(num_batches):
            # reset environment and set initial states
            for e in envs:
                _ = e.reset()
                e.cur_state = cur_state

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

# @ray.remote(num_gpus=1)
class mppi_runner():
    def __init__(self):
        args=get_args()
        args.device = torch.device("cuda:{}".format(args.cuda_id) if torch.cuda.is_available() else "cpu")
        offline_data, sa_processor, env, training_dyn_model_dir = get_rl_data_envs(args.env, args.task, args.device)
        
        # dynamics model environment
        self.env = env
        # evaluation model environment(s)
        assert args.num_samples % args.num_envs == 0
        self.envs = [get_rl_data_envs(args.env, args.task, args.device)[2] for _ in range(args.num_envs)] # TODO: implement DummyVecEnv from mppi repo?

        args.next_obs_shape = (offline_data['next_observations'].shape[1], )
        args.obs_shape = (offline_data['observations'].shape[1], )
        args.action_dim = offline_data['actions'].shape[1]
        args.max_action = 1.0
        action_space = Box(low=-args.max_action, high=args.max_action, shape=(args.action_dim, ), dtype=np.float32)

        # seed
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        self.env.seed(args.seed)
        for e in self.envs:
            e.seed(args.seed)

         # params for MPPI
        self.num_samples = args.num_samples # num trajectory samples
        self.num_envs = args.num_envs
        self.horizon = args.horizon # planning horizon steps
        self.temp = args.temperature # temp parameter for weighting
        self.sigma = args.noise_var # control noise variance
        dt = args.time_step # time step, default = 0
        self.action_dim = args.action_dim
        self.time_limit = offline_data["tracking_ref"].shape[0]
        self.state_dim = offline_data['obs_dim'] + 2 # targets, difference
        self.action_low = offline_data['action_lower_bounds']
        self.action_high = offline_data['action_upper_bounds']

        # nominal control sequence
        U_nominal = np.zeros((args.horizon, args.action_dim))

    def run(self):
        # log
        log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args))
        # key: output file name, value: output handler type
        output_config = {
            "consoleout_backup": "stdout",
            "policy_training_progress": "csv",
            "tb": "tensorboard"
        }
        self.logger = Logger(log_dirs, output_config)
        # logger.log_hyperparameters(vars(args))

        
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


if __name__ == "__main__":
    # import warnings
    # warnings.filterwarnings("ignore")
    # ray.init()
    # test_runner = mppi_runner.remote()
    # new_trajectories = ray.get([test_runner.run.remote()])
    # print(new_trajectories["rewards"][0])
    model = mppi_runner()
    model.run()