import argparse
import random

from gym.spaces import Box

import numpy as np
import torch

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, EnsembleCritic, TanhDiagGaussian
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import MFPolicyTrainer
from offlinerlkit.policy import EDACPolicy
from envs.fusion import SA_processor, NFEnv, load_offline_data
current_directory = os.getcwd()

"""
suggested hypers

halfcheetah-medium-v2: num-critics=10, eta=1.0
hopper-medium-v2: num-critics=50, eta=1.0
walker2d-medium-v2: num-critics=10, eta=1.0
halfcheetah-medium-replay-v2: num-critics=10, eta=1.0
hopper-medium-replay-v2: num-critics=50, eta=1.0
walker2d-medium-replay-v2: num-critics=10, eta=1.0
halfcheetah-medium-expert-v2: num-critics=10, eta=5.0
hopper-medium-expert-v2: num-critics=50, eta=1.0
walker2d-medium-expert-v2: num-critics=10, eta=5.0
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="edac")
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256, 256])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", type=bool, default=True)
    parser.add_argument("--target-entropy", type=int, default=None)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)
    parser.add_argument("--num-critics", type=int, default=50)
    parser.add_argument("--max-q-backup", type=bool, default=False)
    parser.add_argument("--deterministic-backup", type=bool, default=False)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--normalize-reward", type=bool, default=False)

    parser.add_argument("--epoch", type=int, default=3000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)

    parser.add_argument("--task", type=str, default="betan_EFIT01")
    parser.add_argument("--seed", type=int, default=1)
    # parser.add_argument("--offline_data_dir", type=str, default=current_directory + '/data/nf_data.h5') # must run from the examples folder
    parser.add_argument("--raw_data_dir", type=str, default='/zfsauton/project/fusion/data/organized/noshape_gas_flat_top/') # must run from the examples folder
    parser.add_argument('--rnn_model_dir', type=str, default='/zfsauton/project/fusion/models/rpnn_noshape_gas_flat_top_step_two_logvar') #?
    parser.add_argument("--use_partial", type=bool, default=True)
    parser.add_argument("--cuda_id", type=int, default=1)

    return parser.parse_args()


def train(args=get_args()):

    ############################# Modified #################################
    args.device = torch.device("cuda:{}".format(args.cuda_id) if torch.cuda.is_available() else "cpu")
    args.offline_data_dir = args.raw_data_dir + 'processed_data_rl.h5'
    offline_data = load_offline_data(args.offline_data_dir, args.raw_data_dir, args.task, args.use_partial)
    sa_processor = SA_processor(bounds=(offline_data['action_lower_bounds'], offline_data['action_upper_bounds']), \
                                time_limit=offline_data['tracking_ref'].shape[0], device=args.device)
    env = NFEnv(args.rnn_model_dir, args.device, offline_data['tracking_ref'], offline_data['tracking_states'], \
                offline_data['tracking_pre_actions'], offline_data['tracking_actions'], offline_data['index_list'], \
                sa_processor, offline_data["state_idxs"], offline_data["action_idxs"])
    
    # collect the data for rl training
    offline_data['rewards'] = env.get_reward(offline_data['observations'], offline_data['time_step'])
    offline_data['actions'] = sa_processor.normalize_action(offline_data['actions'])
    offline_data['observations'] = sa_processor.get_rl_state(offline_data['observations'], offline_data['time_step'][:, np.newaxis])
    offline_data['next_observations'] = sa_processor.get_rl_state(offline_data['next_observations'], offline_data['time_step'][:, np.newaxis] + 1)

    print(offline_data['observations'].shape, offline_data['next_observations'].shape, offline_data['time_step'].shape, offline_data['rewards'].shape, offline_data['actions'].shape)

    args.obs_shape = (offline_data['observations'].shape[1], )
    args.action_dim = offline_data['actions'].shape[1]
    args.max_action = 1.0
    action_space = Box(low=-args.max_action, high=args.max_action, shape=(args.action_dim, ), dtype=np.float32)
    ############################# Modified End #################################

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims)
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True,
        max_mu=args.max_action
    )
    actor = ActorProb(actor_backbone, dist, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critics = EnsembleCritic(
        np.prod(args.obs_shape), args.action_dim, \
        args.hidden_dims, num_ensemble=args.num_critics, \
        device=args.device
    )
    # init as in the EDAC paper
    for layer in critics.model[::2]:
        torch.nn.init.constant_(layer.bias, 0.1)
    torch.nn.init.uniform_(critics.model[-1].weight, -3e-3, 3e-3)
    torch.nn.init.uniform_(critics.model[-1].bias, -3e-3, 3e-3)
    critics_optim = torch.optim.Adam(critics.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(action_space.shape)
        args.target_entropy = target_entropy
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    # create policy
    policy = EDACPolicy(
        actor,
        critics,
        actor_optim,
        critics_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        max_q_backup=args.max_q_backup,
        deterministic_backup=args.deterministic_backup,
        eta=args.eta
    )

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

    # log
    log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args), record_params=["num_critics", "eta"])
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    # logger.log_hyperparameters(vars(args))

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
    )
    
    policy_trainer.train()


if __name__ == "__main__":
    train()