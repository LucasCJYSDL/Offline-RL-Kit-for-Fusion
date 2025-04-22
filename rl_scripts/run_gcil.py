"""
A goal-conditioned imitation learning algorithm to imitate a list of carefully selected shots.
"""

import argparse
import random

import numpy as np
import torch

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from offlinerlkit.nets import MLP
from offlinerlkit.modules import Actor, ActorProb, TanhDiagGaussian
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import MFPolicyTrainer
from offlinerlkit.policy import BCPolicy

from rl_preparation.get_rl_data_envs import get_rl_data_envs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="gcil")
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256, 256])
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--stochastic_actor", type=bool, default=True)

    #!!! what you need to specify
    parser.add_argument("--env", type=str, default="profile_control") # cannot be base
    parser.add_argument("--task", type=str, default="dens") # betan_EFIT01
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda_id", type=int, default=3)

    return parser.parse_args()


def GCIL(args, offline_data, env, logger): # this function is shared by some other scripts
    # create buffer
    buffer = ReplayBuffer(
        buffer_size=offline_data["observations"].shape[0],
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    buffer.load_dataset(offline_data)

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims)
    if not args.stochastic_actor:
        actor = Actor(actor_backbone, args.action_dim, max_action=args.max_action, device=args.device)
    else:
        dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True,
        max_mu=args.max_action
        )
        actor = ActorProb(actor_backbone, dist, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    # create policy
    policy = BCPolicy(actor, actor_optim, args.stochastic_actor)

    # create policy trainer
    policy_trainer = MFPolicyTrainer(
        policy=policy,
        eval_env=env,
        buffer=buffer,
        logger=logger,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes
    ) # TODO: use a lr scheduler

    # train
    policy_trainer.train()


def train(args=get_args()):
    # offline rl data and env
    args.device = torch.device("cuda:{}".format(args.cuda_id) if torch.cuda.is_available() else "cpu")
    offline_data, _, env, _ = get_rl_data_envs(args.env, args.task, args.device, is_il=True)
    
    args.obs_shape = (offline_data['observations'].shape[1], )
    args.action_dim = offline_data['actions'].shape[1]
    args.max_action = 1.0

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)

    # log
    log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args))
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    # logger.log_hyperparameters(vars(args))

    GCIL(args, offline_data, env, logger)


if __name__ == "__main__":
    train()