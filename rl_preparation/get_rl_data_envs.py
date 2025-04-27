import os 
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
import pickle
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl_preparation.state_actuator_spaces import state_names_to_idxs, actuator_names_to_idxs, get_target_indices, acts_in_use
from rl_preparation.process_raw_data import raw_data_dir, rl_data_path, il_data_path, tracking_data_path, reference_shot, training_model_dir, evaluation_model_dir, change_every
from envs.utils.setup_targets import fixed_ref_shot_targets, step_function_targets

# load the offline dataset from the disk
def load_offline_data(env, tracking_target, is_il):
    # get general data 
    offline_data = {}

    if is_il:
        general_data_path = il_data_path
    else:
        general_data_path = rl_data_path
    
    hdf = h5py.File(general_data_path, 'r')
    offline_data['observations'] = hdf['observations'][:]
    offline_data['actions'] = hdf['actions'][:]
    offline_data['pre_actions'] = hdf['pre_actions'][:]
    offline_data['next_observations'] = hdf['next_observations'][:]
    offline_data['terminals'] = hdf['terminals'][:]
    offline_data['time_step'] = hdf['time_step'][:]
    offline_data['action_lower_bounds'] = hdf['action_lower_bounds'][:]
    offline_data['action_upper_bounds'] = hdf['action_upper_bounds'][:]
    if not is_il:
        offline_data['hidden_states'] = hdf['hidden_states'][:]
    hdf.close()
    
    offline_data['obs_dim'] = offline_data['observations'].shape[1]
    offline_data['act_dim'] = offline_data['actions'].shape[1]

    # if only a subset of the full state space is used, obs, act, and next_obs will go through a post process;
    # so we additionally store full versions of them.
    offline_data['full_observations'] = offline_data['observations'].copy()
    offline_data['full_actions'] = offline_data['actions'].copy()
    offline_data['full_next_observations'] = offline_data['next_observations'].copy()

    # get indices of the tracking target in the state space
    with open(raw_data_dir + '/info.pkl', 'rb') as file:
        data_info = pickle.load(file)

    offline_data['index_list'] = []
    offline_data['tracking_target_names'] = []
    keyword = tracking_target
    for i in range(offline_data['obs_dim']):
        if data_info['state_space'][i].startswith(keyword):
            offline_data['index_list'].append(i)
            offline_data['tracking_target_names'].append(data_info['state_space'][i])

    # get the tracking data
    tracking_data = {} 

    with h5py.File(tracking_data_path, 'r') as hdf:
        ref_shot = hdf[str(reference_shot)]['tracking_states'][:]
        ref_shot_next = hdf[str(reference_shot)]['tracking_next_states'][:]

        for shot_id in hdf:
            tracking_data[int(shot_id)] = {}
            shot = hdf[shot_id]
            for key in shot:
                tracking_data[int(shot_id)][key] = shot[key][:]

            # get targets for the tracking data (evaluation)
            if env == "base":
                tracking_data[int(shot_id)]['tracking_ref'] = fixed_ref_shot_targets(ref_shot_next, offline_data['index_list'], None)
            elif env == "profile_control": # TODO: use the first option (commented for now)
                # tracking_data[int(shot_id)]['tracking_ref'] = step_function_targets(ref_shot, offline_data['index_list'], None, change_every)
                tracking_data[int(shot_id)]['tracking_ref'] = step_function_targets(tracking_data[int(shot_id)]['tracking_states'], offline_data['index_list'], None, change_every)
            else:
                raise NotImplementedError

    # get targets for the general data (training) 
    if env == "base": # TODO: decouple the target setting menthod with the env type
        offline_data['tracking_ref'] = fixed_ref_shot_targets(ref_shot_next, offline_data['index_list'], offline_data['terminals'])
    elif env == "profile_control":
        offline_data['tracking_ref'] = step_function_targets(offline_data['observations'], offline_data['index_list'], offline_data['terminals'], change_every)
        # TODO: usig 'next_observations'
    else:
        raise NotImplementedError
    
    # if only using part of the full state space, we need to update some quantities.
    state_idxs = state_names_to_idxs(raw_data_dir)
    action_idxs = actuator_names_to_idxs(raw_data_dir)
    offline_data['state_idxs'] = state_idxs
    offline_data['action_idxs'] = action_idxs
    offline_data['action_names'] = acts_in_use

    offline_data['observations'] = offline_data['observations'][:, state_idxs]
    offline_data['next_observations'] = offline_data['next_observations'][:, state_idxs]
    offline_data['actions'] = offline_data['actions'][:, action_idxs]
    offline_data['action_lower_bounds'] = offline_data['action_lower_bounds'][action_idxs]
    offline_data['action_upper_bounds'] = offline_data['action_upper_bounds'][action_idxs]
    
    offline_data['obs_dim'] = offline_data['observations'].shape[1]
    offline_data['act_dim'] = offline_data['actions'].shape[1]

    offline_data['index_list'] = get_target_indices(tracking_target, offline_data['obs_dim'])

    return offline_data, tracking_data

# get the offline rl data (in d4rl format) and training
def get_rl_data_envs(env_id, task, device, is_il=False):
    offline_data, tracking_data = load_offline_data(env_id, task, is_il)

    if env_id == 'base':
        from envs.base_env import NFBaseEnv, SA_processor
        sa_processor = SA_processor(offline_data, tracking_data, device)
        env = NFBaseEnv(evaluation_model_dir, sa_processor, offline_data, tracking_data[reference_shot], reference_shot, device) # this is the env for evaluation

    elif env_id == 'profile_control':
        from envs.profile_control_env import ProfileControlEnv
        from envs.base_env import SA_processor
        sa_processor = SA_processor(offline_data, tracking_data, device)
        env = ProfileControlEnv(evaluation_model_dir, sa_processor, offline_data, tracking_data, reference_shot, device) # this is the env for evaluation
    
    else:
        raise NotImplementedError
    
    # collect the data for rl training: (s, a, r, s', d), where d denotes the termination signal
    offline_data['rewards'] = sa_processor.get_reward(offline_data['next_observations'], offline_data['time_step'])
    offline_data['actions'] = sa_processor.normalize_action(offline_data['actions'])
    offline_data['observations'] = sa_processor.get_rl_state(offline_data['observations'], batch_idx=np.arange(0, offline_data['observations'].shape[0]))
    offline_data['next_observations'] = sa_processor.get_rl_state(offline_data['next_observations'], batch_idx=np.arange(1, offline_data['observations'].shape[0]+1))
    # For next_obs where termination is True, the tracking targets for them might be problematic. 
    # However, this doesn't affect training since the bootstrapping from those next_obs will be masked out.
    
    return offline_data, sa_processor, env, training_model_dir
    

if __name__ == "__main__":
    # dummy test only
    # offline_data, tracking_data = load_offline_data(env="profile_control", tracking_target='betan_EFIT01')
    # print(offline_data['index_list'])
    # print(tracking_data.keys())
    # for k, v in tracking_data[reference_shot].items():
    #     print(k, v.shape)
    import torch
    offline_data, sa_processor, env, training_model_dir = get_rl_data_envs("base", "betan_EFIT01", torch.device("cuda"))
    print(offline_data['observations'].shape, offline_data['next_observations'].shape, offline_data['rewards'].shape)