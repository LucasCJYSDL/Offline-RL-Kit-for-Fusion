"""
A model-free planning algorithm: MPPI.
Ref: https://github.com/LucasCJYSDL/LCM/blob/main/trajectory_optimizer/MPPI.py
"""

import argparse
import random
import torch
import h5py
import numpy as np

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.planner import MPPI

from envs.env_wrappers import PlanningWrapper
from rl_preparation.get_rl_data_envs import get_rl_data_envs
from rl_preparation.process_raw_data import raw_data_dir
from rl_scripts.run_gcil import GCIL


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="mppi")

    # hyperparameters for planning
    parser.add_argument("--gamma", type=float, default=1.0) # discount factor, not used for now
    parser.add_argument("--penalty-coef", type=float, default=0.0) # reward penalty
    parser.add_argument("--episodes_per_shot", type=int, default=2) # how many times to evaluate per shot id
    parser.add_argument("--num_envs", type=int, default=1000) # number of parallel environments for planning
    parser.add_argument("--horizon", type=int, default=40) # prediction horizon, i.e., length of the action sequence
    parser.add_argument("--num_samples", type=int, default=1000) # number of action sequences to sample
    parser.add_argument("--lam", type=float, default=2.0) # temperature parameter for MPPI

    # hyperparameters for policy distillation
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256, 256])
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--stochastic_actor", type=bool, default=False)

    #!!! what you need to specify
    parser.add_argument("--load_data", type=bool, default=False) # true, if you already have the planning result and want to skip the planning process
    parser.add_argument("--env", type=str, default="profile_control") # cannot be base
    parser.add_argument("--task", type=str, default="dens") # betan_EFIT01
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda_id", type=int, default=3) # -1 represents using cpus

    return parser.parse_args()


def train(args=get_args()):
    # offline rl data and env
    if args.cuda_id == -1:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:{}".format(args.cuda_id) if torch.cuda.is_available() else "cpu")
    offline_data, _, env, _ = get_rl_data_envs(args.env, args.task, args.device)
    
    args.obs_shape = (offline_data['observations'].shape[1], )
    args.action_dim = offline_data['actions'].shape[1]
    args.max_action, args.min_action = 1.0, -1.0

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)

    # log
    log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args), record_params=["horizon", "num_samples", "lam"])
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    # logger.log_hyperparameters(vars(args))

    # main entry
    if not args.load_data:
        # do MPPI
        plan_env = PlanningWrapper(env) # plan_env is for online planning
        plan_env.seed(args.seed) # TODO: maybe use a different seed here
        planner = MPPI(plan_env, args, logger, args.device)
        planning_dataset = planner.run()
        # save the planning result
        with h5py.File(raw_data_dir + '/mppi_data.h5', 'w') as hdf:
            for key, value in planning_dataset.items():
                print(key, value.shape)
                hdf.create_dataset(key, data=value)
    else:
        # load the planning result
        planning_dataset = {}
        key_list = ["observations", "next_observations", "actions", "rewards", "terminals"]
        with h5py.File(raw_data_dir + '/mppi_data.h5', 'r') as hdf:
            for k in key_list:
                planning_dataset[k] = hdf[k][:]

    # distill the policy through goal-conditioned IL
    GCIL(args, planning_dataset, env, logger)


if __name__ == "__main__":
    train()