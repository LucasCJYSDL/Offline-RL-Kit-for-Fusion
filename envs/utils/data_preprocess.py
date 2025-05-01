import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import torch
import numpy as np
import pickle
import h5py
import yaml
from collections import defaultdict
import pickle
import h5py
from tqdm import tqdm

from dynamics_toolbox.utils.storage.model_storage import load_ensemble_from_parent_dir

current_dir = os.path.dirname(os.path.abspath(__file__))


def _post_process(general_data, offline_dst):
    for k in general_data:
        general_data[k] = np.array(general_data[k])
        print(k, general_data[k].shape) # for sanity check
    
    # applying the actuator bounds, optional
    general_data['action_lower_bounds'] = offline_dst['action_lower_bounds'].copy()
    general_data['action_upper_bounds'] = offline_dst['action_upper_bounds'].copy()
    general_data['pre_actions'] = np.clip(general_data['pre_actions'], general_data['action_lower_bounds'], general_data['action_upper_bounds'])
    general_data['actions'] = np.clip(general_data['actions'], general_data['action_lower_bounds'], general_data['action_upper_bounds'])


def _dump_data(data_path, data_dict):
    with h5py.File(data_path, 'w') as hdf:
        for key, value in data_dict.items():
            hdf.create_dataset(key, data=value)


def get_raw_data(offline_data_dir, action_bound_file, shot_list, warmup_steps): 

    offline_data = {}
    # get main components
    hdf = h5py.File(offline_data_dir + '/full.hdf5', 'r')
    offline_data['observations'] = hdf['states'][:]
    offline_data['observation_deltas'] = hdf['next_states'][:]
    offline_data['pre_actions'] = hdf['actuators'][:]
    offline_data['action_deltas'] = hdf['next_actuators'][:]
    offline_data['shotnum'] = hdf['shotnum'][:]
    offline_data['time'] = hdf['time'][:]
    hdf.close()

    # DANGER, the first 30 steps are dirty, because next_obs is not the same as obs + obs_delta
    # so we filter out the first 30 time steps
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
        
        if shot_num_count >= warmup_steps:
            mask.append(i)
    
    offline_data['observations'] = offline_data['observations'][mask]
    offline_data['observation_deltas'] = offline_data['observation_deltas'][mask]
    offline_data['pre_actions'] = offline_data['pre_actions'][mask]
    offline_data['action_deltas'] = offline_data['action_deltas'][mask]
    offline_data['shotnum'] = offline_data['shotnum'][mask]
    offline_data['time'] = offline_data['time'][mask]
    tot_num = offline_data['shotnum'].shape[0] # the total number of shots

    # get the action bounds
    with open(offline_data_dir + '/info.pkl', 'rb') as file:
        data_info = pickle.load(file)

    action_bound_path = current_dir + '/actuator_bounds/' + action_bound_file
    with open(action_bound_path, 'r') as file:
        data_dict = yaml.safe_load(file)
    
    offline_data['action_lower_bounds'], offline_data['action_upper_bounds'] = [], []
    for act in data_info['actuator_space']:
        lb, ub = data_dict[act]
        assert lb <= ub
        offline_data['action_lower_bounds'].append(lb)
        offline_data['action_upper_bounds'].append(ub)

    offline_data['action_lower_bounds'] = np.array(offline_data['action_lower_bounds'])
    offline_data['action_upper_bounds'] = np.array(offline_data['action_upper_bounds'])

    # we only take shots in the shot list
    ref_start_index = defaultdict(list)
    for i in range(tot_num):
        shot_num = int(offline_data['shotnum'][i])
        if shot_num in shot_list:
            if len(ref_start_index[shot_num]) < 10: # for each shot we take, we have 10 possible starting points
                ref_start_index[shot_num].append(i)
    offline_data['ref_start_index'] = ref_start_index

    # each shot is labelled with time steps and termination signals
    offline_data['time_step'] = []
    offline_data['terminals'] = []
    ts = 0
    for i in tqdm(range(tot_num-1)):
        offline_data['time_step'].append(ts)
        if offline_data['shotnum'][i+1] != offline_data['shotnum'][i]:
        # DANGER, comes from the function sort_by_continuous_snippets
        # if offline_data['shotnum'][i+1] != offline_data['shotnum'][i] or (offline_data['time'][i+1] - offline_data['time'][i]) > 100: # TODO: adjust this threshold
            offline_data['terminals'].append(True)
            ts = 0
        else:
            offline_data['terminals'].append(False)
            ts += 1
    offline_data['time_step'].append(ts)
    offline_data['terminals'].append(True) # a litlle bit buggy
    offline_data['time_step'] = np.array(offline_data['time_step'])
    offline_data['terminals'] = np.array(offline_data['terminals'])

    return offline_data


def store_offlinerl_dataset(offline_dst, model_dir, rl_data_path, il_data_path, tracking_data_path, rl_shot_list, il_shot_list, tracking_shot_list, device):
    rl_data = {'observations': [], 'pre_actions': [], 'actions': [], 'next_observations': [], 
                'terminals': [], 'time_step': [], 'hidden_states': []}
    il_data = {'observations': [], 'pre_actions': [], 'actions': [], 'next_observations': [], 
                'terminals': [], 'time_step': []}
    tracking_data = {}
    
    # load the rnn model ensemble used for training
    ensemble = load_ensemble_from_parent_dir(parent_dir=model_dir)
    all_models = ensemble.members
    for memb in all_models:
        memb.to(device)
        memb.eval()
    
    # go over the training dataset, generate/collect the hidden states, time-costly!
    # TODO: filter out shots that are too short
    shot_num_list = list(offline_dst['ref_start_index'].keys())
    for cur_shot in tqdm(shot_num_list):
        t = offline_dst['ref_start_index'][cur_shot][0]
        cur_state = offline_dst['observations'][t]

        # initialize this target shot
        if cur_shot in tracking_shot_list:
            terminated = False
            tracking_data[cur_shot] = {'tracking_states': [], 'tracking_next_states': [], 'tracking_pre_actions': [], 'tracking_actions': []}
        
        while True:                

            # collect training data for rl and il
            pre_action = offline_dst['pre_actions'][t]
            action_delta = offline_dst['action_deltas'][t]
            state_delta = offline_dst['observation_deltas'][t]
            next_state = cur_state + state_delta
            cur_action = pre_action + action_delta

            if cur_shot in rl_shot_list:
                rl_data['observations'].append(cur_state.copy())
                rl_data['pre_actions'].append(pre_action.copy())
                rl_data['actions'].append(cur_action.copy())
                rl_data['next_observations'].append(next_state.copy())
                rl_data['terminals'].append(offline_dst['terminals'][t])
                rl_data['time_step'].append(offline_dst['time_step'][t])
            
            if cur_shot in il_shot_list:
                il_data['observations'].append(cur_state.copy())
                il_data['pre_actions'].append(pre_action.copy())
                il_data['actions'].append(cur_action.copy())
                il_data['next_observations'].append(next_state.copy())
                il_data['terminals'].append(offline_dst['terminals'][t])
                il_data['time_step'].append(offline_dst['time_step'][t])

            # collect tracking data
            if cur_shot in tracking_data and not terminated:
                tracking_data[cur_shot]['tracking_states'].append(cur_state.copy())
                tracking_data[cur_shot]['tracking_next_states'].append(next_state.copy())
                tracking_data[cur_shot]['tracking_pre_actions'].append(pre_action.copy())
                tracking_data[cur_shot]['tracking_actions'].append(cur_action.copy())

            # end of shot - time to get the hidden states
            if offline_dst['terminals'][t]:
                terminated = True
                if cur_shot in rl_shot_list:
                    # prepare the input
                    s_id = len(rl_data['hidden_states'])
                    shot_states = np.array(rl_data['observations'][s_id:])
                    shot_pre_actions = np.array(rl_data['pre_actions'][s_id:])
                    shot_cur_actions = np.array(rl_data['actions'][s_id:])
                    net_input = torch.cat([torch.FloatTensor(shot_states).to(device), 
                                        torch.FloatTensor(shot_pre_actions).to(device),
                                        torch.FloatTensor(shot_cur_actions - shot_pre_actions).to(device)], dim=-1)
                    
                    # inference with the rpnn dynamics model
                    memb_out_list = []
                    for memb in all_models:
                        memb.reset() # optional
                        net_input_n = memb.normalizer.normalize(net_input, 0)
                        memb_out = memb.get_mem_out(net_input_n).unsqueeze(1)
                        memb_out_list.append(memb_out)
                    shot_hidden_states = torch.stack(memb_out_list, dim=1).cpu().tolist()
                    rl_data['hidden_states'].extend(shot_hidden_states)
            
            # otherwise, prepare for the next time step
            t += 1
            if t >= len(offline_dst['shotnum']) or offline_dst['shotnum'][t] != cur_shot:
                break
            
            # if not offline_dst['terminals'][t-1]:
            #     cur_state = next_state # TODO: which one is better
            # else:
            cur_state = offline_dst['observations'][t]
    
    # post process
    _post_process(rl_data, offline_dst)
    _post_process(il_data, offline_dst)

    # same for the tracking data
    for shot_id in tracking_data:
        for k in tracking_data[shot_id]:
            tracking_data[shot_id][k] = np.array(tracking_data[shot_id][k])

        tracking_data[shot_id]['tracking_pre_actions'] = np.clip(tracking_data[shot_id]['tracking_pre_actions'], 
                                                                 rl_data['action_lower_bounds'], rl_data['action_upper_bounds'])
        tracking_data[shot_id]['tracking_actions'] = np.clip(tracking_data[shot_id]['tracking_actions'], 
                                                             rl_data['action_lower_bounds'], rl_data['action_upper_bounds'])
    
    # save the training data
    _dump_data(rl_data_path, rl_data)
    _dump_data(il_data_path, il_data)

    # save the tracking data
    with h5py.File(tracking_data_path, 'w') as hdf:
        for shot_id, shot_data in tracking_data.items():
            group = hdf.create_group(str(shot_id))
            for key, value in shot_data.items():
                group.create_dataset(key, data=value)


