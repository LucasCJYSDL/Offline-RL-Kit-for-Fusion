import argparse
import random

import gym
import d4rl

import json, os
import numpy as np
import torch
from tqdm import tqdm

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian, EnsembleDynamicsModel
from offlinerlkit.dynamics import BayesEnsembleDynamics
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.termination_fns import get_termination_fn
from offlinerlkit.utils.load_dataset import qlearning_dataset
from offlinerlkit.utils.scheduler import LinearParameter
from offlinerlkit.buffer import ReplayBuffer, BayesReplayBuffer, SLReplaBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import BayesMBPolicyTrainer
from offlinerlkit.policy import BAMBRLPolicy
from offlinerlkit.utils.searcher import Searcher

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="bambrl")
    parser.add_argument("--task", type=str, default="walker2d-medium-expert-v2")
    parser.add_argument("--seed", type=int, default=1)
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

    parser.add_argument("--dynamics-lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs-since-update", type=int, default=5)
    parser.add_argument("--dynamics-max-epochs", type=int, default=30)
    parser.add_argument("--dynamics-hidden-dims", type=int, nargs='*', default=[200, 200, 200, 200])
    parser.add_argument("--dynamics-weight-decay", type=float, nargs='*', default=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4])
    parser.add_argument("--n-ensemble", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    parser.add_argument("--rollout-freq", type=int, default=1000)
    parser.add_argument("--rollout-batch-size", type=int, default=50000)
    parser.add_argument("--rollout-length", type=int, default=1)
    parser.add_argument("--penalty-coef", type=float, default=1.5)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.05)
    parser.add_argument("--load-dynamics-path", type=str, default=None)

    parser.add_argument("--epoch", type=int, default=3000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr-scheduler", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cuda:3" if torch.cuda.is_available() else "cpu")

    # search related
    parser.add_argument("--use-search", type=bool, default=True) # required by ba-mcts and ba-mcts-sl
    parser.add_argument("--search-ratio", type=float, default=0.1) # 0.1
    parser.add_argument("--search-alpha", type=float, default=0.8) # 0.8
    parser.add_argument("--search-ucb-coe", type=float, default=1.0) # 1.0
    parser.add_argument("--search-root-alpha", type=float, default=0.3) # 0.3
    parser.add_argument("--search-n-actions", type=float, default=10) # 10
    parser.add_argument("--search-n-states", type=float, default=5) # 5
    parser.add_argument("--search-n-search", type=float, default=50) # 50

    parser.add_argument("--use-sl", type=bool, default=False) # required by ba-mcts-sl
    parser.add_argument("--sl-policy-only", type=bool, default=True) # only use sl to train the policy (recommended)
    parser.add_argument("--model-retain-epochs-sl", type=int, default=5)
    parser.add_argument("--use-ba", type=bool, default=True)
    parser.add_argument("--elite-only", type=bool, default=False)
    parser.add_argument("--sample-step", type=bool, default=False)
    parser.add_argument("--test_search", type=bool, default=True)

    return parser.parse_args()


def train(load_path=None, eval_path=None):
    # set the args
    args = get_args()
    if load_path is not None:
        json_file = load_path + '/hyper_param.json'
        with open(json_file, 'r') as file:
            new_args_dict = json.load(file)
        
        # update the args
        blocked_terms = ['device', 'algo_name']
        args_dict = vars(args)
        for k, v in new_args_dict.items():
            if k in blocked_terms:
                continue
            args_dict[k] = v
            
        args = argparse.Namespace(**args_dict)
        args.load_dynamics_path = load_path + '/model'
    
    if args.use_search:
        args.algo_name += '_mcts'
    if args.use_sl:
        args.algo_name += '_sl'
    
    # create env and dataset
    env = gym.make(args.task)
    # dataset = qlearning_dataset(env)
    dataset = env.get_dataset()
    if args.norm_reward:
        r_mean, r_std = dataset["rewards"].mean(), dataset["rewards"].std()
        dataset["rewards"] = (dataset["rewards"] - r_mean) / (r_std + 1e-3)

    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]
    env_vec = [gym.make(args.task) for _ in range(args.eval_episodes)]

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)
    for i in range(len(env_vec)):
        env_vec[i].seed(args.seed+i)

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
            else -np.prod(env.action_space.shape)

        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    # create dynamics
    load_dynamics_model = True if args.load_dynamics_path else False
    dynamics_model = EnsembleDynamicsModel(
        obs_dim=np.prod(args.obs_shape),
        action_dim=args.action_dim,
        hidden_dims=args.dynamics_hidden_dims,
        num_ensemble=args.n_ensemble,
        num_elites=args.n_elites,
        weight_decays=args.dynamics_weight_decay,
        device=args.device
    )
    dynamics_optim = torch.optim.Adam(
        dynamics_model.parameters(),
        lr=args.dynamics_lr
    )
    scaler = StandardScaler()
    termination_fn = get_termination_fn(task=args.task)
    dynamics = BayesEnsembleDynamics(
        args.sample_step,
        dynamics_model,
        dynamics_optim,
        scaler,
        termination_fn
    )

    if args.load_dynamics_path:
        dynamics.load(args.load_dynamics_path)
    
    # create buffer
    if args.elite_only:
        prior_dim = args.n_elites
    else:
        prior_dim = args.n_ensemble
    real_buffer = BayesReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        prior_dim=prior_dim,
        device=args.device
    )
    real_buffer.load_dataset(dataset)

    fake_buffer = BayesReplayBuffer(
        buffer_size=args.rollout_batch_size*args.rollout_length*args.model_retain_epochs,
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        prior_dim=prior_dim,
        device=args.device
    )

    if args.use_sl:
        sl_buffer_size = int(args.rollout_length * args.rollout_batch_size * args.search_ratio * args.model_retain_epochs_sl) # ipt
        sl_buffer = SLReplaBuffer(args.action_dim, np.prod(args.obs_shape), args.search_n_actions, capacity=sl_buffer_size)
        entropy_coe_scheduler = LinearParameter(start=0.01, end=0.001, num_steps=args.epoch * args.step_per_epoch)
    else:
        sl_buffer = None
        entropy_coe_scheduler = None

    # log
    log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args), record_params=["penalty_coef", "rollout_length", "real_ratio"])
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))

    # train dynamics model
    if not load_dynamics_model:
        dynamics.train(
            real_buffer.sample_all(),
            logger,
            max_epochs_since_update=args.max_epochs_since_update,
            max_epochs=args.dynamics_max_epochs
        )
    
    elite_list = dynamics_model.elites.detach().clone().cpu().numpy()
    elite_list.sort()

        # create searcher
    if args.use_search:
        searcher = Searcher(args, dynamics, elite_list)
    else:
        searcher = None

    # create policy
    policy = BAMBRLPolicy(
        args.elite_only,
        elite_list,
        args.use_ba,
        args.use_search,
        args.search_ratio,
        args.sl_policy_only,
        searcher,
        sl_buffer,
        entropy_coe_scheduler,
        dynamics,
        actor,
        critics,
        actor_optim,
        critics_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        penalty_coef=args.penalty_coef,
        num_samples=args.num_samples,
        deterministic_backup=args.deterministic_backup,
        max_q_backup=args.max_q_backup
    )

    # get scheduler for the temperature
    # prior_tem_scheduler = LinearScheduler(start_value=100.0, end_value=1.0, num_intervals=1000)
    
    # get Bayes priors associated with the offline dataset
    all_probs = dynamics.get_bayes_priors(dataset) # (7, 2000000)

    if args.elite_only:
        temp_prior = 1.0 / args.n_elites
        uniform_all_prior = np.array([temp_prior for _ in range(args.n_elites)])
        all_probs = all_probs[elite_list]
    else:
        temp_prior = 1.0 / args.n_ensemble
        uniform_all_prior = np.array([temp_prior for _ in range(args.n_ensemble)])
    
    all_priors = [uniform_all_prior]
    trans_num = all_probs.shape[1]
    for i in tqdm(range(trans_num - 1)):
        if args.use_ba:
            done = dataset['terminals'][i]
            if 'timeouts' in dataset:
                final_timestep = dataset['timeouts'][i]
                done = done or final_timestep
            if done:
                all_priors.append(uniform_all_prior)
            else:
                a_prior = all_priors[i]
                a_prob = all_probs[:, i]
                a_prod = a_prior * a_prob
                all_priors.append(a_prod / (a_prod.sum() + 1e-6))
        else:
            all_priors.append(uniform_all_prior)

    all_priors = np.array(all_priors) # (2000000, 7)
    real_buffer.load_prior(all_priors)

    # create policy trainer
    policy_trainer = BayesMBPolicyTrainer(
        policy=policy,
        eval_env=env,
        real_buffer=real_buffer,
        fake_buffer=fake_buffer,
        logger=logger,
        rollout_setting=(args.rollout_freq, args.rollout_batch_size, args.rollout_length),
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        real_ratio=args.real_ratio,
        eval_episodes=args.eval_episodes,
        lr_scheduler=lr_scheduler
    )

    # train policy   
    if eval_path is None: 
        policy_trainer.train()
    else:
        policy_trainer.batch_evaluate(env_vec, eval_path=eval_path, ensemble_size=(args.n_elites if args.elite_only else args.n_ensemble), 
                                      use_search=args.test_search, use_ba=args.use_ba)


if __name__ == "__main__":
    current_working_directory = os.getcwd()
    load_path_ls = ['/data/wk-med-exp/seed-0', '/data/wk-med-exp/seed-1', '/data/wk-med-exp/seed-2']
    load_path_id = 1 # 0-6
    # eval_path = '/log/walker2d-medium-replay-v2/bambrl_mcts&penalty_coef=0.5&rollout_length=1&real_ratio=0.05/seed_1&timestamp_24-1214-131101/checkpoint'
    train(current_working_directory + load_path_ls[load_path_id])