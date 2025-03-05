import torch
import numpy as np
import pickle
import h5py
import random
import yaml
from collections import defaultdict
import pickle
import h5py
from tqdm import tqdm

from dynamics_toolbox.utils.storage.model_storage import load_ensemble_from_parent_dir

class SA_processor:
    def __init__(self, bounds, time_limit, device):
        self.time_limit = time_limit
        self.range = torch.FloatTensor((bounds[1] - bounds[0]) / 2.0).unsqueeze(0).to(device)
        self.mid = torch.FloatTensor((bounds[1] + bounds[0]) / 2.0).unsqueeze(0).to(device)
    
    def get_rl_state(self, state, time_steps):
        time_input = torch.FloatTensor(time_steps / float(self.time_limit)).unsqueeze(-1).to(state.device)
        rl_state = torch.cat([state, time_input], dim=-1)

        return rl_state
    
    def restore_time_steps(self, float_time_steps):
        # print(float_time_steps.shape, np.rint(float_time_steps * self.time_limit).shape)
        return np.rint(float_time_steps * self.time_limit).astype(int)
    
    def get_step_action(self, action):
        act_num = action.shape[0]
        return action * self.range.repeat(act_num, 1) + self.mid.repeat(act_num, 1)
    
    def normalize_action(self, action):
        act_num = action.shape[0]
        return (action - self.mid.repeat(act_num, 1).cpu().numpy()) / (self.range.repeat(act_num, 1).cpu().numpy()+1e-6)

def state_names_to_idxs(data_path, states_in_obs):
    with open(f'{data_path}/info.pkl', 'rb') as f:
        info = pickle.load(f)
        all_states = info['state_space']
        idxs = []
        for s in states_in_obs:
            idxs.append(all_states.index(s))
    return idxs

def actuator_names_to_idxs(data_path, acts_in_obs):
    with open(f'{data_path}/info.pkl', 'rb') as f:
        info = pickle.load(f)
        all_acts = info['actuator_space']
        idxs = []
        for s in acts_in_obs:
            idxs.append(all_acts.index(s))
    return idxs

def get_raw_data(offline_data_dir, tracking_target, reference_shot, action_bound_path, save_file, states_in_obs=None, acts_in_obs=None): 
    offline_data = {}
    # get main components
    with h5py.File(offline_data_dir + 'full.hdf5', 'r') as hdf:
        # print("Keys in the file:", list(hdf.keys()))
        full_observations = hdf['states'][:]
        # this sum is because 'next_actuators' are a_t - a_{t-1} for 'actuators' and 'actuators' are a_{t-1}
        offline_data['pre_actions'] = hdf['actuators'][:]
        offline_data['action_deltas'] = hdf['next_actuators'][:]
        full_next_observations = hdf['states'][:] + hdf['next_states'][:]
        offline_data['shotnum'] = hdf['shotnum'][:]
        # shape: (3793282, 27) (3793282, 14) (3793282, 27) (3793282,) 
        total_state_num = np.array(full_observations).shape[1]
        total_act_num = np.array(offline_data['pre_actions']).shape[1]

        if states_in_obs is not None:
            # Select only the specified columns/dimensions
            state_idxs = state_names_to_idxs(offline_data_dir, states_in_obs)
            offline_data['observations'] = full_observations[:, state_idxs]
            offline_data['next_observations'] = full_next_observations[:, state_idxs]
        else:
            # Use all dimensions if none specified
            offline_data['observations'] = full_observations
            offline_data['next_observations'] = full_next_observations
        
        if acts_in_obs is not None:
            # Select only the specified columns/dimensions
            acts_idxs = actuator_names_to_idxs(offline_data_dir, acts_in_obs)
            offline_data['pre_actions'] = offline_data['pre_actions'][:, acts_idxs]
            offline_data['action_deltas'] = offline_data['action_deltas'][:, acts_idxs]

    offline_data['obs_dim'] = np.array(offline_data['observations']).shape[1]
    offline_data['act_dim'] = np.array(offline_data['pre_actions']).shape[1]

    print('############# obs dim ############')
    print(f"selected {offline_data['obs_dim']}/{total_state_num} states")
    print('############# act dim ############')
    print(f"selected {offline_data['act_dim']}/{total_act_num} states")

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
    offline_data['actions'] = offline_data['pre_actions']
    offline_data['action_deltas'] = offline_data['action_deltas'][mask]
    # offline_data['next_observations'] = offline_data['next_observations'][mask]
    offline_data['shotnum'] = offline_data['shotnum'][mask]
    print(offline_data['shotnum'][:100])
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

    if acts_in_obs is not None:
            # Select only the specified columns/dimensions
            offline_data['action_lower_bounds'] = offline_data['action_lower_bounds'][:, acts_idxs]
            offline_data['action_upper_bounds'] = offline_data['action_upper_bounds'][:, acts_idxs]

    offline_data['pre_actions'] = np.clip(offline_data['pre_actions'], offline_data['action_lower_bounds'], offline_data['action_upper_bounds'])
    offline_data['action_deltas'] = np.clip(offline_data['action_deltas'], offline_data['action_lower_bounds'], offline_data['action_upper_bounds'])
    # print(offline_data['action_lower_bounds'], offline_data['action_upper_bounds'])
            
    offline_data['tracking_ref'] = []
    offline_data['tracking_states'] = []
    offline_data['tracking_actions'] = []
    found = False
    for i in range(tot_num):
        if int(offline_data['shotnum'][i]) == reference_shot:
            offline_data['tracking_ref'].append(offline_data['observations'][i][offline_data['index_list']])
            found = True
            
            offline_data['tracking_states'].append(offline_data['observations'][i])
            offline_data['tracking_actions'].append(offline_data['pre_actions'][i])
        else:
            if found:
                break
    offline_data['tracking_ref'] = np.array(offline_data['tracking_ref'])
    print(offline_data['tracking_ref'].shape)
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
        #if offline_data['shotnum'][i] == reference_shot: # TODO: double check if this makes sense from data side
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
    # print( offline_data['time_step'].shape, offline_data['next_observations'].shape, len(mask))
    # assert offline_data['time_step'].shape == offline_data['next_observations'].shape
    offline_data['terminals'] = np.array(offline_data['terminals'])
    # shape: (3793282,) (3793282,)

    with h5py.File(save_file, 'w') as hdf:
        for key, value in offline_data.items():
            hdf.create_dataset(key, data=value)

    return offline_data

def get_offline_data(data_path, tracking_target):
    offline_data = {}
    # get main components
    with h5py.File(data_path, 'r') as hdf:
        # print("Keys in the file:", list(hdf.keys()))
        full_observations = hdf['observations'][:]
        offline_data['actions'] = hdf['actions'][:]
        # actions wasn't processed in get_raw_data
        offline_data['next_observations'] = hdf['next_observations'][:]
        offline_data['terminals'] = hdf['terminals'][:]
        offline_data['time_step'] = hdf['time_step'][:]
        offline_data['tracking_states'] = hdf['tracking_states'][:] # states of the reference shot
        offline_data['tracking_actions'] = hdf['tracking_actions'][:] # actions of the reference shot

        offline_data['action_lower_bounds'] = hdf['action_lower_bounds'][:]
        offline_data['action_upper_bounds'] = hdf['action_upper_bounds'][:]
        offline_data['index_list'] = hdf['index_list'][:]
        offline_data['tracking_ref'] = hdf['tracking_ref'][:]

        offline_data['obs_dim'] = hdf['obs_dim'][()]
        offline_data['act_dim'] = hdf['act_dim'][()]


    # offline_data['obs_dim'] = offline_data['observations'].shape[1]
    # offline_data['act_dim'] = offline_data['actions'].shape[1]

    # get the start points for RL training
    # with open('/home/scratch/linmo/fusion_data/' + 'info.pkl', 'rb') as file: # TODO: change this path
    #     data_info = pickle.load(file)
        # print(data_info['state_space'], len(data_info['state_space']))
    
    
    # keyword = tracking_target # e.g., betan
    # for i in range(offline_data['obs_dim']):
    #     if data_info['state_space'][i].startswith(keyword):
    #         offline_data['index_list'].append(i) # the dimension of state that betan correponds to
    
    
    # for i in range(len(offline_data['tracking_states'])):
    #     offline_data['tracking_ref'].append(offline_data['tracking_states'][i][offline_data['index_list']])

    offline_data['tracking_ref'] = np.array(offline_data['tracking_ref'])

    # offline_data['ref_start_index'] = []
    # for i in range(len(offline_data['observations'])):
    #     if offline_data['time_step'][i] < offline_data['tracking_ref'].shape[0] - 1:
    #         offline_data['ref_start_index'].append(i)

    return offline_data


class NFEnv:
    def __init__(self, model_dir, device, tracking_ref, tracking_states, tracking_actions, index_list, sa_processor):
        self.cur_time = None
        self.time_limit = tracking_ref.shape[0]
        self.tracking_ref = np.array(tracking_ref)
        self.tracking_states = np.array(tracking_states)
        self.tracking_actions = np.array(tracking_actions)
        self.index_list = np.array(index_list) # the dimensions that the target quantity corresponds to
        self.track_coefficients = np.array([1.0 for _ in range(len(index_list))])
        # load the well-trained rnn model ensemble
        ensemble = load_ensemble_from_parent_dir(parent_dir=model_dir) # TODO: an ensemble of dynamics models
        self.all_models = ensemble.members
        for memb in self.all_models:
            memb.to(device)
            memb.eval()
        
        self.device = device
        self.sa_processor = sa_processor

    def seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)

    def reset(self):
        self.cur_time = random.randint(0, 9)
        self.cur_state = torch.FloatTensor(self.tracking_states[self.cur_time]).unsqueeze(0).to(self.device)
        self.pre_action = torch.FloatTensor(self.tracking_actions[self.cur_time]).unsqueeze(0).to(self.device)
        # reset the model
        for memb in self.all_models:
            memb.reset()
        
        return self.sa_processor.get_rl_state(self.cur_state, np.array([self.cur_time]))


    def step(self, cur_action):
        cur_action = torch.Tensor(cur_action).to(self.device)
        cur_action = self.sa_processor.get_step_action(cur_action)
        net_input = torch.cat([self.cur_state, self.pre_action, cur_action-self.pre_action], dim=-1) # TODO: agjust the input, ask Rohit what is the input/output of the current dynamics model

        ensemble_preds = 0.
        with torch.no_grad():
            for memb in self.all_models:
                net_input_n = memb.normalizer.normalize(net_input, 0)
                net_output_n, _ = memb.single_sample_output_from_torch(net_input_n) # torch.Size([1, 27])
                net_output = memb.normalizer.unnormalize(net_output_n, 1)
                ensemble_preds += net_output
        ensemble_preds = ensemble_preds / float(len(self.all_models)) # delta of the state

        self.cur_state = self.cur_state + ensemble_preds # the next state
        self.cur_time += 1

        return self.sa_processor.get_rl_state(self.cur_state, np.array([self.cur_time])), self.get_reward(self.cur_state.cpu().numpy(), self.cur_time)[0], self.is_done(self.cur_time), {}

    def get_reward(self, next_state, time_step): # return a numpy array
        # this time_step is for the next_state
        # time_step_indxs = np.where(time_step == 0)[0] #TODO: Ask about what this step does
        # print(time_step_indxs.shape)
        if np.isscalar(time_step):
            if time_step >= self.time_limit:
                time_step = self.time_limit - 1
            targets = self.tracking_ref[time_step]
            # print(next_state[:, self.index_list].shape, targets.shape, self.track_coefficients[np.newaxis, :].shape)
            return -1.0 * (np.square(next_state[:, self.index_list] - targets) * self.track_coefficients[np.newaxis, :]).mean(axis=1) / float(self.time_limit)
        else:
            for i in range(len(time_step)):
                if time_step[i] >= self.time_limit:
                    time_step[i] = self.time_limit - 1
            targets = self.tracking_ref[time_step]
            print("fusion env get reward debug dims", next_state[:, self.index_list].shape, targets.shape, self.track_coefficients[np.newaxis, :].shape)
            return -1.0 * (np.square(next_state[:, self.index_list] - targets) * self.track_coefficients[np.newaxis, :]).mean(axis=1) / float(self.time_limit)
            #TODO: 230 magic number fix
        # return -1.0 * (np.square(next_state[:, self.index_list] - targets) * self.track_coefficients[np.newaxis, :]).mean(axis=1) 

    def is_done(self, time_step):
        # print('fusion env', time_step, self.time_limit)
        return time_step >= self.time_limit - 1

    def get_normalized_score(self, ep_reward):
        return ep_reward / 100
