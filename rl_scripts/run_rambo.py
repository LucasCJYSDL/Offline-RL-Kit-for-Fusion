import argparse
import random

import numpy as np
import torch

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian, EnsembleDynamicsModel
from offlinerlkit.dynamics import EnsembleDynamics
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.buffer import ReplayBuffer, ModelSLReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import MBPolicyTrainer
from offlinerlkit.policy import RAMBOPolicy
from rl_preparation.get_rl_data_envs import get_rl_data_envs


"""
suggested hypers

halfcheetah-medium-v2: rollout-length=5, adv-weight=3e-4
hopper-medium-v2: rollout-length=5, adv-weight=3e-4
walker2d-medium-v2: rollout-length=5, adv-weight=0
halfcheetah-medium-replay-v2: rollout-length=5, adv-weight=3e-4
hopper-medium-replay-v2: rollout-length=5, adv-weight=3e-4
walker2d-medium-replay-v2: rollout-length=5, adv-weight=0
halfcheetah-medium-expert-v2: rollout-length=5, adv-weight=0
hopper-medium-expert-v2: rollout-length=5, adv-weight=0
walker2d-medium-expert-v2: rollout-length=2, adv-weight=3e-4
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="rambo")
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    # parser.add_argument("--dynamics-lr", type=float, default=3e-4)
    parser.add_argument("--dynamics-adv-lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", default=True)
    parser.add_argument("--target-entropy", type=int, default=None)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)

    # parser.add_argument("--dynamics-weight-decay", type=float, nargs='*', default=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4]) # TODO
    parser.add_argument("--rollout-freq", type=int, default=250)
    parser.add_argument("--dynamics-update-freq", type=int, default=1000) #??
    parser.add_argument("--adv-batch-size", type=int, default=256)
    parser.add_argument("--adv-train-steps", type=int, default=500)
    parser.add_argument("--sl-batch-size", type=int, default=8)
    parser.add_argument("--rollout-batch-size", type=int, default=50000)
    parser.add_argument("--rollout-length", type=int, default=5)
    parser.add_argument("--adv-weight", type=float, default=3e-4) #??
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.5)

    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)

    parser.add_argument("--include-ent-in-adv", type=bool, default=False)
    parser.add_argument("--load-bc-path", type=str, default=None)
    parser.add_argument("--bc-lr", type=float, default=1e-4)
    parser.add_argument("--bc-epoch", type=int, default=50)
    parser.add_argument("--bc-batch-size", type=int, default=256)

    #!!! what you need to specify
    parser.add_argument("--env", type=str, default="profile_control") # one of [base, profile_control]
    parser.add_argument("--task", type=str, default="betan_EFIT01") #?
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--update_hidden_states", type=bool, default=False) # whether to update the hidden states in the offline dataset, since the dynamics model is being updated with the rl policy
    parser.add_argument("--cuda_id", type=int, default=3)

    return parser.parse_args()


def train(args=get_args()):
    # offline rl data and env
    args.device = torch.device("cuda:{}".format(args.cuda_id) if torch.cuda.is_available() else "cpu")
    offline_data, sa_processor, env, training_dyn_model_dir = get_rl_data_envs(args.env, args.task, args.device)

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

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims)
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True,
        max_mu=args.max_action
    )
    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)
    
    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -args.action_dim

        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    # create buffer
    real_buffer = ReplayBuffer(
        buffer_size=len(offline_data["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    real_buffer.load_dataset(offline_data, hidden=True)

    # obs_mean, obs_std = real_buffer.normalize_obs() # TODO
    fake_buffer_size = args.step_per_epoch // args.rollout_freq * args.model_retain_epochs * args.rollout_batch_size * args.rollout_length
    fake_buffer = ReplayBuffer(
        buffer_size=fake_buffer_size, 
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )

    # the buffer for sl of the dynamics model
    model_sl_buffer = ModelSLReplayBuffer()
    model_sl_buffer.load_datasets(offline_data)
    
    # create dynamics
    dynamics_model = EnsembleDynamicsModel(
        model_path=training_dyn_model_dir,
        device=args.device
    )
    # dynamics_optim = torch.optim.Adam(
    #     dynamics_model.parameters(),
    #     lr=args.dynamics_lr
    # ) 
    dynamics_adv_optim = torch.optim.Adam(
        dynamics_model.parameters(), 
        lr=args.dynamics_adv_lr
    )

    # dynamics_scaler = StandardScaler()
    # termination_fn = obs_unnormalization(get_termination_fn(task=args.task), obs_mean, obs_std)
    termination_fn = env.is_done
    reward_fn = sa_processor.get_reward
    dynamics = EnsembleDynamics(
        dynamics_model,
        termination_fn,
        reward_fn
    )

    # create policy
    # policy_scaler = StandardScaler(mu=obs_mean, std=obs_std) #TODO
    policy = RAMBOPolicy(
        dynamics, 
        actor, 
        critic1, 
        critic2, 
        actor_optim, 
        critic1_optim, 
        critic2_optim, 
        dynamics_adv_optim,
        offline_data['state_idxs'],
        offline_data['action_idxs'],
        sa_processor,
        model_sl_buffer,
        args.update_hidden_states,
        tau=args.tau, 
        gamma=args.gamma, 
        alpha=alpha, 
        adv_weight=args.adv_weight, 
        adv_train_steps=args.adv_train_steps,
        adv_rollout_length=args.rollout_length, 
        adv_rollout_batch_size=args.adv_batch_size,
        sl_batch_size=args.sl_batch_size,
        include_ent_in_adv=args.include_ent_in_adv,
        # scaler=policy_scaler, #TODO
        device=args.device
    ).to(args.device)

    # log
    log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args))
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
    policy_trainer = MBPolicyTrainer(
        policy=policy,
        eval_env=env,
        real_buffer=real_buffer,
        fake_buffer=fake_buffer,
        logger=logger,
        rollout_setting=(args.rollout_freq, args.rollout_batch_size, args.rollout_length),
        dynamics_update_freq=args.dynamics_update_freq,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        real_ratio=args.real_ratio,
        eval_episodes=args.eval_episodes
    )

    # train the policy and dynamics model
    if args.load_bc_path:
        policy.load(args.load_bc_path)
        policy.to(args.device)
    else:
        policy.pretrain(real_buffer.sample_all(), args.bc_epoch, args.bc_batch_size, args.bc_lr, logger)

    # if args.load_dynamics_path:
    #     dynamics.load(args.load_dynamics_path)
    # else:
    #     dynamics.train(
    #         real_buffer.sample_all(),
    #         logger,
    #         holdout_ratio=0.1,
    #         logvar_loss_coef=0.001, # TODO
    #         max_epochs_since_update=10
    #     )

    policy_trainer.train()


if __name__ == "__main__":
    train()