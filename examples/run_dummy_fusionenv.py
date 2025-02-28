import random
import gym
# import d4rl
import numpy as np
import torch.nn as nn
import torch
import sys, os
from tqdm import tqdm
import h5py
import pickle
import yaml
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.fusion import SA_processor, NFEnv, get_offline_data
from offlinerlkit.nets import MLP

print()
print()
print("====== above is all warnings lol =======")
print()

# device configurations
device = 'cuda:0'

class DummyPolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DummyPolicyNet, self).__init__()
        self.linear = nn.Linear(input_dim + 1, output_dim, device=device)
        # extra dimension time index appended at end

    def forward(self, x):
        x = self.linear(x)
        return x

def get_raw_data(offline_data_dir, tracking_target, reference_shot, action_bound_path, save_file): # TODO: load the raw dataset
    offline_data = {}
    # get main components
    with h5py.File(offline_data_dir + 'full.hdf5', 'r') as hdf:
        # print("Keys in the file:", list(hdf.keys()))
        offline_data['observations'] = hdf['states'][:]
        # this sum is because 'next_actuators' are a_t - a_{t-1} for 'actuators' and 'actuators' are a_{t-1}
        offline_data['pre_actions'] = hdf['actuators'][:]
        offline_data['action_deltas'] = hdf['next_actuators'][:]
        offline_data['next_observations'] = offline_data['observations'] + hdf['next_states'][:]
        offline_data['shotnum'] = hdf['shotnum'][:]
        # shape: (3793282, 27) (3793282, 14) (3793282, 27) (3793282,) 
    offline_data['obs_dim'] = np.array(offline_data['observations']).shape[1]
    offline_data['act_dim'] = np.array(offline_data['pre_actions']).shape[1]
    

    # the first 30 steps are dirty
    # print(offline_data['observations'][20:50]) 
    # print(offline_data['observations'][26:51] - offline_data['next_observations'][25:50])
    # print(offline_data['shotnum'][:51])
    # print(offline_data['observations'][26:51] - offline_data['observations'][25:50] - offline_data['next_observations'][25:50])
    mask = []
    old_shot_num = -1
    shot_num_count = -1
    for i in range(len(offline_data['shotnum'])):
        shot_num = offline_data['shotnum'][i]
        if shot_num != old_shot_num:
            old_shot_num = shot_num
            shot_num_count = 0
        else:
            shot_num_count += 1
        
        if shot_num_count >= 30:
            mask.append(i)
    
    offline_data['observations'] = offline_data['observations'][mask]
    offline_data['pre_actions'] = offline_data['pre_actions'][mask]
    offline_data['action_deltas'] = offline_data['action_deltas'][mask]
    # offline_data['next_observations'] = offline_data['next_observations'][mask]
    offline_data['shotnum'] = offline_data['shotnum'][mask]
    tot_num = offline_data['shotnum'].shape[0]
    # print(tot_num, offline_data['observations'][1001:1031] - offline_data['next_observations'][1000:1030]) # 3108605

    # get the start points for RL training
    with open(offline_data_dir + 'info.pkl', 'rb') as file:
        data_info = pickle.load(file)
        # print(data_info['state_space'], len(data_info['state_space']))
    offline_data['index_list'] = []
    keyword = tracking_target
    for i in range(offline_data['obs_dim']):
        if data_info['state_space'][i].startswith(keyword):
            offline_data['index_list'].append(i)
    
    # get the action bounds
    with open(action_bound_path, 'r') as file: # TODO: change this path
        data_dict = yaml.safe_load(file)
    
    offline_data['action_lower_bounds'], offline_data['action_upper_bounds'] = [], []
    for act in data_info['actuator_space']:
        lb, ub = data_dict[act]
        assert lb <= ub
        offline_data['action_lower_bounds'].append(lb)
        offline_data['action_upper_bounds'].append(ub)

    offline_data['action_lower_bounds'] = np.array(offline_data['action_lower_bounds'])
    offline_data['action_upper_bounds'] = np.array(offline_data['action_upper_bounds'])

    offline_data['pre_actions'] = np.clip(offline_data['pre_actions'], offline_data['action_lower_bounds'], offline_data['action_upper_bounds'])
    offline_data['action_deltas'] = np.clip(offline_data['action_deltas'], offline_data['action_lower_bounds'], offline_data['action_upper_bounds'])
    # print(offline_data['action_lower_bounds'], offline_data['action_upper_bounds'])
            
    offline_data['tracking_ref'] = []
    offline_data['tracking_states'] = []
    offline_data['tracking_actions'] = []
    found = False
    for i in range(tot_num):
        if int(offline_data['shotnum'][i]) == reference_shot:
            found = True
            offline_data['tracking_ref'].append(offline_data['observations'][i][offline_data['index_list']])
            offline_data['tracking_states'].append(offline_data['observations'][i])
            offline_data['tracking_actions'].append(offline_data['pre_actions'][i])
        else:
            if found:
                break
    offline_data['tracking_ref'] = np.array(offline_data['tracking_ref'])
    offline_data['tracking_states'] = np.array(offline_data['tracking_states'])
    offline_data['tracking_actions'] = np.array(offline_data['tracking_actions'])

    ref_start_index = defaultdict(list)
    for i in range(tot_num):
        shot_num = int(offline_data['shotnum'][i])
        if -50 <= shot_num - reference_shot <=50:
            # if len(ref_start_index[shot_num]) < offline_data['tracking_ref'].shape[0] - 1:
            if len(ref_start_index[shot_num]) < 10:
                ref_start_index[shot_num].append(i)
    # offline_data['ref_start_index'] = ref_start_index

    # for key in ref_start_index:
    #     offline_data['ref_start_index'].extend(ref_start_index[key])
    # 4082
    # print(len(offline_data['ref_start_index']), offline_data['ref_start_index'])

    # time step labels and termination labels
    offline_data['time_step'] = []
    offline_data['terminals'] = []
    ts = 0
    for i in tqdm(range(tot_num-1)):
        offline_data['time_step'].append(ts)
        if offline_data['shotnum'][i+1] != offline_data['shotnum'][i]:
            offline_data['terminals'].append(True)
            ts = 0
        else:
            offline_data['terminals'].append(False)
            ts += 1
    offline_data['time_step'].append(ts)
    offline_data['terminals'].append(True) # a litlle bit buggy
    offline_data['time_step'] = np.array(offline_data['time_step'])
    offline_data['terminals'] = np.array(offline_data['terminals'])
    # shape: (3793282,) (3793282,)

    with h5py.File(save_file, 'w') as hdf: # TODO: change the path
        for key, value in offline_data.items():
            hdf.create_dataset(key, data=value)

    return offline_data

def evaluate(policy, env):    
    T = 10

    episode_return = 0
    s_t = env.reset() # TODO: get the initial state
    for t in range(T):
        a_t = policy.forward(s_t)
        s_t1, reward, done, _ = env.step(a_t) # TODO: get the next state, current reward, and if the episode ends
        episode_return += reward

    return episode_return

if __name__ == "__main__":
    # TODO: load these from given beta file
    offline_data_dir = "/zfsauton/project/fusion/data/organized/noshape_gas_flat_top/"
    tracking_target = "betan_EFIT01"
    reference_shot = 189268
    action_bound_path = "/zfsauton2/home/linmo/FusionControl/cfgs/control/environment/actuator_bounds/noshape_gas.yaml"
    data_path = "/zfsauton2/home/linmo/Offline-RL-Kit/data/nf_data.h5"    
    model_dir = "/zfsauton/project/fusion/models/rpnn_noshape_gas_flat_top_step_two_logvar"

    if os.path.exists(data_path):
        print("data path exists, reading from processed data")
        offline_data = get_offline_data(data_path, tracking_target) # TODO: figure out if it's necessary to have both get_raw_data and get_offline data
    else:
        print("data path doesn't exist, processing raw data")
        offline_data = get_raw_data(offline_data_dir, tracking_target, reference_shot, action_bound_path, data_path) # load the raw data and convert it to nf_data.h5

    
    sa_processor = SA_processor(bounds=(offline_data['action_lower_bounds'], offline_data['action_upper_bounds']), \
                                time_limit=offline_data['tracking_ref'].shape[0], device=device)
    env = NFEnv(model_dir, device, offline_data['tracking_ref'], offline_data['tracking_states'], offline_data['tracking_actions'], offline_data['index_list'], sa_processor)
    env.seed(seed=0)
    policy = DummyPolicyNet(offline_data['obs_dim'], offline_data['act_dim']) # TODO: takes state as input and output an action
    
    # single pass through to make sure params are initiated
    dummy_input = torch.cat([torch.Tensor(offline_data['observations'][0]).to(device), torch.Tensor([1]).to(device)])
    dummy_action = policy(dummy_input)
    # print('dummy_action shape', dummy_action.shape)

    epi_return = evaluate(policy, env)
    print('episode return:', epi_return) 
    