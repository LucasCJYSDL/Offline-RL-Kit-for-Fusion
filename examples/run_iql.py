import argparse
import random
from gym.spaces import Box ## bug fix

import numpy as np
import torch

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, DiagGaussian
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import MFPolicyTrainer
from offlinerlkit.policy import IQLPolicy
from rl_preparation.get_rl_data_envs import get_rl_data_envs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="iql")
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256])
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-q-lr", type=float, default=3e-4)
    parser.add_argument("--critic-v-lr", type=float, default=3e-4)
    parser.add_argument("--dropout_rate", type=float, default=None)
    parser.add_argument("--lr-decay", type=bool, default=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)

    #!!! what you need to specify
    parser.add_argument("--env", type=str, default="profile_control") # one of [base, profile_control]
    parser.add_argument("--task", type=str, default="betan_EFIT01")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda_id", type=int, default=3)
   
    return parser.parse_args()


def normalize_rewards(dataset):
    terminals_float = np.zeros_like(dataset["rewards"])
    for i in range(len(terminals_float) - 1):
        if np.linalg.norm(dataset["observations"][i + 1] -
                            dataset["next_observations"][i]
                            ) > 1e-6 or dataset["terminals"][i] == 1.0:
            terminals_float[i] = 1
        else:
            terminals_float[i] = 0

    terminals_float[-1] = 1

    # split_into_trajectories
    trajs = [[]]
    for i in range(len(dataset["observations"])):
        trajs[-1].append((dataset["observations"][i], dataset["actions"][i], dataset["rewards"][i], 1.0-dataset["terminals"][i],
                        terminals_float[i], dataset["next_observations"][i]))
        if terminals_float[i] == 1.0 and i + 1 < len(dataset["observations"]):
            trajs.append([])
    
    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    # normalize rewards
    dataset["rewards"] /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset["rewards"] *= 1000.0

    return dataset


def train(args=get_args()):
    # offline rl data and env
    args.device = torch.device("cuda:{}".format(args.cuda_id) if torch.cuda.is_available() else "cpu")
    offline_data, sa_processor, env, training_dyn_model_dir = get_rl_data_envs(args.env, args.task, args.device) 

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
    env.seed(args.seed)

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims, dropout_rate=args.dropout_rate)
    critic_q1_backbone = MLP(input_dim=np.prod(args.obs_shape)+args.action_dim, hidden_dims=args.hidden_dims)
    critic_q2_backbone = MLP(input_dim=np.prod(args.obs_shape)+args.action_dim, hidden_dims=args.hidden_dims)
    critic_v_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims)
    dist = DiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=False,
        conditioned_sigma=False,
        max_mu=args.max_action
    )
    actor = ActorProb(actor_backbone, dist, args.device)
    critic_q1 = Critic(critic_q1_backbone, args.device)
    critic_q2 = Critic(critic_q2_backbone, args.device)
    critic_v = Critic(critic_v_backbone, args.device)
    
    for m in list(actor.modules()) + list(critic_q1.modules()) + list(critic_q2.modules()) + list(critic_v.modules()):
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_q1_optim = torch.optim.Adam(critic_q1.parameters(), lr=args.critic_q_lr)
    critic_q2_optim = torch.optim.Adam(critic_q2.parameters(), lr=args.critic_q_lr)
    critic_v_optim = torch.optim.Adam(critic_v.parameters(), lr=args.critic_v_lr)

    if args.lr_decay:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args.epoch)
    else:
        lr_scheduler = None
    
    # create IQL policy
    policy = IQLPolicy(
        actor,
        critic_q1,
        critic_q2,
        critic_v,
        actor_optim,
        critic_q1_optim,
        critic_q2_optim,
        critic_v_optim,
        action_space=action_space, ## bug fix
        tau=args.tau,
        gamma=args.gamma,
        expectile=args.expectile,
        temperature=args.temperature
    )

    # create buffer
    buffer = ReplayBuffer(
        buffer_size=offline_data["observations"].shape[0], ## bug fix
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    buffer.load_dataset(offline_data) ## bug fix

    # log
    log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args))
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    # logger.log_hyperparameters(vars(args)) ## bug fix

    # create policy trainer
    policy_trainer = MFPolicyTrainer(
        policy=policy,
        eval_env=env,
        buffer=buffer,
        logger=logger,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes,
        lr_scheduler=lr_scheduler
    )

    # train
    policy_trainer.train()


if __name__ == "__main__":
    train()