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

def get_raw_data(offline_data_dir, reference_shot, action_bound_file, shot_range): 

    offline_data = {}
    # get main components
    hdf = h5py.File(offline_data_dir + '/full.hdf5', 'r')
    offline_data['observations'] = hdf['states'][:]
    offline_data['observations_delta'] = hdf['next_states'][:]
    offline_data['pre_actions'] = hdf['actuators'][:]
    offline_data['action_deltas'] = hdf['next_actuators'][:]
    offline_data['shotnum'] = hdf['shotnum'][:]
    hdf.close()

    # the first 30 steps are dirty, because next_obs is not the same as obs + obs_delta
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
        
        if shot_num_count >= 30:
            mask.append(i)
    
    offline_data['observations'] = offline_data['observations'][mask]
    offline_data['observations_delta'] = offline_data['observations_delta'][mask]
    offline_data['pre_actions'] = offline_data['pre_actions'][mask]
    offline_data['action_deltas'] = offline_data['action_deltas'][mask]
    offline_data['shotnum'] = offline_data['shotnum'][mask]
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

    # we only take shots around the reference shot
    ref_start_index = defaultdict(list)
    for i in range(tot_num):
        shot_num = int(offline_data['shotnum'][i])
        if abs(shot_num - reference_shot) <= shot_range: # very important hyperparameter
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


def store_offline_dataset(offline_dst, model_dir, general_data_path, ref_shot, tracking_shot_range, tracking_data_path, device):
    general_data = {'observations': [], 'pre_actions': [], 'actions': [], 'next_observations': [], 
                    'terminals': [], 'time_step': [], 'hidden_states': []}
    tracking_data = {}
    
    # load the rnn model ensemble used for training
    ensemble = load_ensemble_from_parent_dir(parent_dir=model_dir)
    all_models = ensemble.members
    for memb in all_models:
        memb.to(device)
        memb.eval()
    
    # go over the training dataset, generate/collect the hidden states, time-costly!
    shot_num_list = list(offline_dst['ref_start_index'].keys())
    for cur_shot in tqdm(shot_num_list):
        t = offline_dst['ref_start_index'][cur_shot][0]
        cur_state = offline_dst['observations'][t]

        # initialize this target shot
        if abs(cur_shot - ref_shot) <= tracking_shot_range:
            tracking_data[cur_shot] = {'tracking_states': [], 'tracking_next_states': [], 'tracking_pre_actions': [], 'tracking_actions': []}
        
        while True:                

            # collect general data from this time step
            pre_action = offline_dst['pre_actions'][t]
            action_delta = offline_dst['action_deltas'][t]
            state_delta = offline_dst['observations_delta'][t]
            next_state = cur_state + state_delta
            cur_action = pre_action + action_delta

            general_data['observations'].append(cur_state.copy())
            general_data['pre_actions'].append(pre_action.copy())
            general_data['actions'].append(cur_action.copy())
            general_data['next_observations'].append(next_state.copy())
            general_data['terminals'].append(offline_dst['terminals'][t])
            general_data['time_step'].append(offline_dst['time_step'][t])

            # collect tracking data
            if cur_shot in tracking_data:
                tracking_data[cur_shot]['tracking_states'].append(cur_state.copy())
                tracking_data[cur_shot]['tracking_next_states'].append(next_state.copy())
                tracking_data[cur_shot]['tracking_pre_actions'].append(pre_action.copy())
                tracking_data[cur_shot]['tracking_actions'].append(cur_action.copy())

            # end of shot - time to get the hidden states
            if offline_dst['terminals'][t]:
                # prepare the input
                s_id = len(general_data['hidden_states'])
                shot_states = np.array(general_data['observations'][s_id:])
                shot_pre_actions = np.array(general_data['pre_actions'][s_id:])
                shot_cur_actions = np.array(general_data['actions'][s_id:])
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
                general_data['hidden_states'].extend(shot_hidden_states)

                break
            
            # otherwise, prepare for the next time step
            t += 1
            cur_state = next_state
    
    # post process
    for k in general_data:
        general_data[k] = np.array(general_data[k])
        print(k, general_data[k].shape) # for sanity check
    
    # applying the actuator bounds, optional
    general_data['action_lower_bounds'] = offline_dst['action_lower_bounds'].copy()
    general_data['action_upper_bounds'] = offline_dst['action_upper_bounds'].copy()
    general_data['pre_actions'] = np.clip(general_data['pre_actions'], general_data['action_lower_bounds'], general_data['action_upper_bounds'])
    general_data['actions'] = np.clip(general_data['actions'], general_data['action_lower_bounds'], general_data['action_upper_bounds'])

    # same for the tracking data
    for shot_id in tracking_data:
        for k in tracking_data[shot_id]:
            tracking_data[shot_id][k] = np.array(tracking_data[shot_id][k])

        tracking_data[shot_id]['tracking_pre_actions'] = np.clip(tracking_data[shot_id]['tracking_pre_actions'], 
                                                                 general_data['action_lower_bounds'], general_data['action_upper_bounds'])
        tracking_data[shot_id]['tracking_actions'] = np.clip(tracking_data[shot_id]['tracking_actions'], 
                                                             general_data['action_lower_bounds'], general_data['action_upper_bounds'])
    
    # save the general data
    with h5py.File(general_data_path, 'w') as hdf:
        for key, value in general_data.items():
            hdf.create_dataset(key, data=value)

    # save the tracking data
    with h5py.File(tracking_data_path, 'w') as hdf:
        for shot_id, shot_data in tracking_data.items():
            group = hdf.create_group(str(shot_id))
            for key, value in shot_data.items():
                group.create_dataset(key, data=value)


