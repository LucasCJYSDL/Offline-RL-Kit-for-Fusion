import os
import torch
import random
import pickle as pkl
import numpy as np
from tqdm import tqdm
import argparse

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rl_preparation.get_rl_data_envs import get_rl_data_envs
from visualization.controller import Controller
from visualization.plotter import plot_tracking_quantities, plot_actions


#!!! what you need to specify
def get_args():
    parser = argparse.ArgumentParser(description="Trajectory evaluation arguments")

    # basic settings
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--cuda_id", type=int, default=1, help="CUDA device ID")
    parser.add_argument("--plot_actuators", type=bool, default=True, help="Whether to plot actuators")

    # env settings
    parser.add_argument("--env", type=str, default="profile_control") 
    parser.add_argument("--task", type=str, default="dens", help="Targets to track") 

    # controller settings, of which the core is an NN actor
    parser.add_argument("--actor_path", type=str, default="_log/dens/mopo&penalty_coef=2.5&rollout_length=5/seed_1&timestamp_25-0415-211829", help="Path to the actor checkpoint")
    parser.add_argument("--il_actor", type=bool, default=False, help="Is this an imitation learning actor?")
    parser.add_argument("--stochastic_actor", type=bool, default=True, help="Is this a stochatic actor?")
    parser.add_argument("--hidden_dims", type=int, nargs='*', default=[256, 256], help="Hidden dimensions of the actor network") # you can get this in corresponding rl scripts
    parser.add_argument("--deterministic_mode", action="store_true", help="Whether to make the actor deterministic")

    return parser.parse_args()


def run(args=get_args()) -> None:
    # register an env
    args.device = torch.device("cuda:{}".format(args.cuda_id) if torch.cuda.is_available() else "cpu")
    offline_data, sa_processor, env, _ = get_rl_data_envs(args.env, args.task, args.device, is_il=args.il_actor) # these are the data and env used to train the actor
    args.obs_dim = offline_data['observations'].shape[1]
    args.action_dim = offline_data['actions'].shape[1]
    args.max_action = 1.0

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)

    # load up the actor
    controller = Controller(args)
    
    # rollouts
    shot_list = env.get_eval_shot_list()
    quan_names, act_names = sa_processor.get_plot_names() # name of the quantities to track and actuators in control
    
    actor_info = args.actor_path.split('/') # create a folder to store the visualization results
    log_folder = os.path.join(os.path.dirname(__file__), "results", actor_info[1], actor_info[2], actor_info[3])
    os.makedirs(log_folder, exist_ok=True)

    for shot in shot_list:
        obs = env.reset(shot_id=shot)
        episode_reward, episode_length = 0, 0

        time_array = []
        target_quan_array, real_quan_array, cur_quan_array, real_act_array, cur_act_array = [], [], [], [], []

        while True:
            action = controller.act(obs)
            next_obs, reward, terminal, info = env.step(action)
            episode_reward += reward
            episode_length += 1

            # get quantities to plot
            cur_time = info["time_step"] - 1
            target_quan, real_quan, cur_quan, real_act, cur_act = sa_processor.get_plot_quantities(shot, cur_time, obs, action)
            time_array.append(cur_time)
            target_quan_array.append(target_quan)
            real_quan_array.append(real_quan)
            cur_quan_array.append(cur_quan)
            real_act_array.append(real_act)
            cur_act_array.append(cur_act)

            if terminal:
                break

            obs = next_obs
        
        # make plots
        plot_tracking_quantities(time_array, target_quan_array, real_quan_array, cur_quan_array, quan_names, shot, log_folder)
        if args.plot_actuators:
            plot_actions(time_array, real_act_array, cur_act_array, act_names, shot, log_folder)

        # summary
        print("Shot #{} with return {} and length {}".format(shot, episode_reward, episode_length))

if __name__ == "__main__":
    run()