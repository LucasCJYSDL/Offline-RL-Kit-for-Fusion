import argparse
import random
import gym
# import d4rl
import numpy as np
import torch.nn as nn
import torch
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from offlinerlkit.utils.load_dataset import FusionEnv

# from offlinerlkit.nets import MLP
# from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian
# from offlinerlkit.utils.load_dataset import qlearning_dataset, fusion_dataset, FusionEnv
# from offlinerlkit.buffer import ReplayBuffer
# from offlinerlkit.utils.logger import Logger, make_log_dirs
# from offlinerlkit.policy_trainer import MFPolicyTrainer
# from offlinerlkit.policy import CQLPolicy

def str_to_float_list(input_string):
    return [float(item) for item in input_string.split(',')]

def get_dummy_dataset():
    # consider episode of length T
    # consider 2 actuators
    T = 10
    num_actuator = 2
    state_dim = 5
    num_episodes = 2

    # we will consider all states in this dummy
    # dummy dataset has 2 episodes
    return {
        'observations': np.random.rand(2, T, state_dim), # dim: 2 x 10 x 5
        'actions': np.random.rand(2, T, num_actuator),
        'rewards': np.random.rand(2, T),
        'targets': np.random.rand(2, T, state_dim)
    }

class DumbNetwork(nn.Module):
    # custom dynamics model
    def __init__(self):
        super().__init__()
        self.T = 10
        self.num_actuator = 2
        self.state_dim = 5

        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(self.state_dim + self.num_actuator, self.state_dim) 
        )

    def predict(self, state_act):
        # input dim: (5 + 2)
        # output dim: 5
        x = torch.Tensor(state_act)
        next_obs = self.linear(x)
        return next_obs

def get_policy_action():
    # dummy policy that returns random action
    num_actuator = 2
    return np.random.rand(2)

def train():
    dataset = get_dummy_dataset()
    dynamics_model = DumbNetwork()
    env = FusionEnv(dataset, dynamics_model)

    T = 10

    a_t, s_t = env.reset(T) # initial reset, not sure if needed
    for t in range(T):
        a_t = get_policy_action()
        s_t = dataset['observations'][0][t]
        target_t = dataset['targets'][0][t]
        s_t1, reward = env.step(s_t, a_t, target_t)
    return env.reward

if __name__ == "__main__":
    reward = train()
    print('reward:', reward)