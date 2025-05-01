import argparse
import random
import numpy as np
import torch

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian, EnsembleDynamicsModel
from offlinerlkit.dynamics import EnsembleDynamics
from offlinerlkit.buffer import ReplayBuffer, RobustRolloutBuffer, ModelSLReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import MBPolicyTrainer
from offlinerlkit.policy import ROMBRLPolicy
from rl_preparation.get_rl_data_envs import get_rl_data_envs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="rombrl")
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", type=bool, default=True)
    parser.add_argument("--target-entropy", type=int, default=None)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)

    parser.add_argument("--num-q-ensemble", type=int, default=2)
    parser.add_argument("--deterministic-backup", type=bool, default=True)
    parser.add_argument("--max-q-backup", type=bool, default=False)
    parser.add_argument("--norm-reward", type=bool, default=False)

    parser.add_argument("--rollout-freq", type=int, default=1000)
    parser.add_argument("--rollout-batch-size", type=int, default=50000)
    parser.add_argument("--rollout-length", type=int, default=1)
    parser.add_argument("--penalty-coef", type=float, default=1.5)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.05)

    parser.add_argument("--epoch", type=int, default=3000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr-scheduler", type=bool, default=True)

    # new hyperparameters
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--onpolicy-rollout-batch-size", type=int, default=2500) # important hyperparameter # 250
    parser.add_argument("--onpolicy-rollout-length", type=int, default=10) # 100
    parser.add_argument("--small_traj_batch", type=bool, default=False)
    parser.add_argument("--actor_training_epoch", type=int, default=10)
    parser.add_argument("--actor-dynamics-update-freq", type=int, default=1000)
    parser.add_argument("--onpolicy-batch-size", type=int, default=256)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--grad_mode", type=int, default=1) # 1 is recommended, corresponding to Eqs. (7) - (9) in the paper

    parser.add_argument("--I_coe", type=float, default=5.0) # fine-tune
    parser.add_argument("--epsilon", type=float, default=10.0) # fine-tune
    parser.add_argument("--down_sample_size", type=int, default=8) # m in the paper
    parser.add_argument("--dynamics-adv-lr", type=float, default=3e-4)
    parser.add_argument("--dynamics_training_epoch", type=int, default=10)
    parser.add_argument("--include-ent-in-adv", type=bool, default=False)

    # lagrangian-related
    parser.add_argument("--sl_weight", type=float, default=1000.0) # fine-tune
    parser.add_argument("--lambda_training_epoch", type=int, default=1) # fine-tune
    parser.add_argument("--lambda_lr", type=float, default=1e-3) # fine-tune

    #!!! what you need to specify
    parser.add_argument("--env", type=str, default="profile_control") # one of [base, profile_control]
    parser.add_argument("--task", type=str, default="dens") #?
    parser.add_argument("--update_hidden_states", type=bool, default=True) # whether to update the hidden states in the offline dataset, since the dynamics model is being updated with the rl policy
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda_id", type=int, default=2)

    return parser.parse_args()


def train(args=get_args()):
    # set args
    args.device = torch.device("cuda:{}".format(args.cuda_id) if torch.cuda.is_available() else "cpu")
    if args.grad_mode == 3 and args.onpolicy_rollout_batch_size == 2500 and args.small_traj_batch:
        args.small_traj_batch = False

    # offline rl data and env
    offline_data, sa_processor, env, training_dyn_model_dir = get_rl_data_envs(args.env, args.task, args.device)

    args.obs_shape = (offline_data['observations'].shape[1], )
    args.action_dim = offline_data['actions'].shape[1]
    args.max_action = 1.0

    if args.norm_reward:
        r_mean, r_std = offline_data["rewards"].mean(), offline_data["rewards"].std()
        offline_data["rewards"] = (offline_data["rewards"] - r_mean) / (r_std + 1e-3)

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
    critics = []
    for i in range(args.num_q_ensemble):
        critic_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
        critics.append(Critic(critic_backbone, args.device))
    critics = torch.nn.ModuleList(critics)
    critics_optim = torch.optim.Adam(critics.parameters(), lr=args.critic_lr)

    if args.lr_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args.epoch)
    else:
        lr_scheduler = None

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

    fake_buffer = ReplayBuffer(
        buffer_size=args.rollout_batch_size*args.rollout_length*args.model_retain_epochs,
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )

    ## the buffer for sl of the dynamics model
    model_sl_buffer = ModelSLReplayBuffer()
    model_sl_buffer.load_datasets(offline_data)

    ## the buffer for on-policy training
    onpolicy_buffer = RobustRolloutBuffer(args.onpolicy_rollout_length, args.obs_shape, args.action_dim, 
                                          args.device, args.gae_lambda, args.gamma, n_envs=args.onpolicy_rollout_batch_size) # short rollouts with high parallelism

    # create dynamics
    dynamics_model = EnsembleDynamicsModel(
        model_path=training_dyn_model_dir,
        device=args.device
    )
    dynamics_adv_optim = torch.optim.Adam(
        dynamics_model.parameters(), 
        lr=args.dynamics_adv_lr
    )

    termination_fn = env.is_done
    reward_fn = sa_processor.get_reward
    dynamics = EnsembleDynamics(
        dynamics_model,
        termination_fn,
        reward_fn
    )

    # create policy
    policy = ROMBRLPolicy(
        dynamics,
        actor,
        critics,
        actor_optim,
        critics_optim,
        offline_data['state_idxs'],
        offline_data['action_idxs'],
        sa_processor,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        penalty_coef=args.penalty_coef,
        num_samples=args.num_samples,
        deterministic_backup=args.deterministic_backup,
        max_q_backup=args.max_q_backup,
        # new
        small_traj_batch=args.small_traj_batch,
        dynamics_adv_optim=dynamics_adv_optim,
        model_sl_buffer=model_sl_buffer,
        onpolicy_buffer=onpolicy_buffer,
        grad_mode=args.grad_mode,
        I_coe=args.I_coe,
        epsilon=args.epsilon,
        down_sample_size=args.down_sample_size,
        sl_weight=args.sl_weight, 
        lambda_training_epoch=args.lambda_training_epoch,
        lambda_lr=args.lambda_lr,
        onpolicy_rollout_length=args.onpolicy_rollout_length, 
        onpolicy_rollout_batch_size=args.onpolicy_rollout_batch_size,
        onpolicy_batch_size = args.onpolicy_batch_size,
        clip_range = args.clip_range,
        actor_training_epoch = args.actor_training_epoch,
        dynamics_training_epoch = args.dynamics_training_epoch,
        include_ent_in_adv=args.include_ent_in_adv,
        update_hidden_states=args.update_hidden_states,
        device=args.device
    )

    # log
    log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args), record_params=["grad_mode", "sl_weight", "actor_training_epoch", "onpolicy_rollout_batch_size", "onpolicy_rollout_length", "small_traj_batch"])
    
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
        dynamics_update_freq=args.actor_dynamics_update_freq, # hacky, but to make full use of the APIs of the original codebase
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        real_ratio=args.real_ratio,
        eval_episodes=args.eval_episodes,
        lr_scheduler=lr_scheduler
    )

    # train
    policy_trainer.train()


if __name__ == "__main__":
    train()