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

# which states/actuators are actuallu used
states_in_obs = ['betan_EFIT01',
                "temp_component1", 
                "temp_component2", 
                "temp_component3", 
                "temp_component4", 
                "itemp_component1", 
                "itemp_component2", 
                "itemp_component3", 
                "itemp_component4", 
                "dens_component1", 
                "dens_component2", 
                "dens_component3", 
                "dens_component4", 
                "rotation_component1", 
                "rotation_component2", 
                "rotation_component3", 
                "rotation_component4", 
                "pres_EFIT01_component1", 
                "pres_EFIT01_component2", 
                "q_EFIT01_component1", 
                "q_EFIT01_component2"]

acts_in_obs = ['pinj','tinj','gasA','bt_magnitude','bt_is_positive','ech_pwr_total']


def state_names_to_idxs(data_path):
    with open(data_path + 'info.pkl', 'rb') as f:
        info = pickle.load(f)
        all_states = info['state_space']
        idxs = []
        for s in states_in_obs:
            idxs.append(all_states.index(s))
    return idxs

def actuator_names_to_idxs(data_path):
    with open(data_path + 'info.pkl', 'rb') as f:
        info = pickle.load(f)
        all_acts = info['actuator_space']
        idxs = []
        for s in acts_in_obs:
            idxs.append(all_acts.index(s))
    return idxs

def get_raw_data(offline_data_dir, reference_shot, action_bound_path): 

    offline_data = {}
    # get main components
    with h5py.File(offline_data_dir + 'full.hdf5', 'r') as hdf:
        # print("Keys in the file:", list(hdf.keys()))
        offline_data['observations'] = hdf['states'][:]
        offline_data['observations_delta'] = hdf['next_states'][:]
        # this sum is because 'next_actuators' are a_t - a_{t-1} for 'actuators' and 'actuators' are a_{t-1}
        offline_data['pre_actions'] = hdf['actuators'][:]
        offline_data['action_deltas'] = hdf['next_actuators'][:]
        # offline_data['next_observations'] = offline_data['observations'] + hdf['next_states'][:]
        offline_data['shotnum'] = hdf['shotnum'][:]
        # shape: (3793282, 27) (3793282, 14) (3793282, 27) (3793282,) 
    offline_data['obs_dim'] = offline_data['observations'].shape[1]
    offline_data['act_dim'] = offline_data['pre_actions'].shape[1]

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
    offline_data['observations_delta'] = offline_data['observations_delta'][mask]
    offline_data['pre_actions'] = offline_data['pre_actions'][mask]
    offline_data['action_deltas'] = offline_data['action_deltas'][mask]
    # offline_data['next_observations'] = offline_data['next_observations'][mask]
    offline_data['shotnum'] = offline_data['shotnum'][mask]
    tot_num = offline_data['shotnum'].shape[0]
    # print(tot_num, offline_data['observations'][1001:1031] - offline_data['next_observations'][1000:1030]) # 3108605

    # get indexes of the the tracking target 
    with open(offline_data_dir + 'info.pkl', 'rb') as file:
        data_info = pickle.load(file)
        # print(data_info['state_space'], len(data_info['state_space']))
    
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

    # offline_data['pre_actions'] = np.clip(offline_data['pre_actions'], offline_data['action_lower_bounds'], offline_data['action_upper_bounds'])
    # offline_data['action_deltas'] = np.clip(offline_data['action_deltas'], offline_data['action_lower_bounds'], offline_data['action_upper_bounds'])
    # print(offline_data['action_lower_bounds'], offline_data['action_upper_bounds'])

    ref_start_index = defaultdict(list)
    for i in range(tot_num):
        shot_num = int(offline_data['shotnum'][i])
        if -50 <= shot_num - reference_shot <=50: # very important hyperparameter
            # if len(ref_start_index[shot_num]) < offline_data['tracking_ref'].shape[0] - 1:
            if len(ref_start_index[shot_num]) < 10:
                ref_start_index[shot_num].append(i)
    offline_data['ref_start_index'] = ref_start_index

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

    return offline_data

def store_offline_dataset(offline_dst, data_path, ref_shot, model_dir, device):
    data = {'observations': [], 'pre_actions': [], 'actions': [], 'next_observations': [], 
            'terminals': [], 'tracking_states': [], 'tracking_pre_actions': [], 'time_step': [],
            'tracking_actions': [], 'hidden_states': []}
    
    # load the well-trained rnn model ensemble
    ensemble = load_ensemble_from_parent_dir(parent_dir=model_dir)
    all_models = ensemble.members
    for memb in all_models:
        memb.to(device)
        memb.eval()
    
    shot_num_list = list(offline_dst['ref_start_index'].keys())
    for cur_shot in shot_num_list:
        t = offline_dst['ref_start_index'][cur_shot][0]
        cur_state = offline_dst['observations'][t]

        # reset the model
        for memb in all_models:
            memb.reset()
        
        while True:    
            pre_action = offline_dst['pre_actions'][t]
            action_delta = offline_dst['action_deltas'][t]
            state_delta = offline_dst['observations_delta'][t]

            # get the hidden state, very hacky
            hidden_states = []
            for memb in all_models:
                tmp_hidden_state = memb._hidden_state
                if tmp_hidden_state is None:
                    tmp_hidden_state = torch.zeros(memb._num_layers, 1, memb._hidden_size, device=device)
                hidden_states.append(tmp_hidden_state.cpu().numpy())
            hidden_states = np.array(hidden_states)
            hidden_states = np.squeeze(hidden_states, axis=2)

            # update the hidden state
            net_input = torch.cat([torch.FloatTensor(cur_state).unsqueeze(0).to(device), 
                                   torch.FloatTensor(pre_action).unsqueeze(0).to(device),
                                   torch.FloatTensor(action_delta).unsqueeze(0).to(device)], dim=-1)
            with torch.no_grad():
                for memb in all_models:
                    net_input_n = memb.normalizer.normalize(net_input, 0)
                    memb.single_sample_output_from_torch(net_input_n) # torch.Size([1, 27])

            next_state = cur_state + state_delta
            data['observations'].append(cur_state.copy())
            data['hidden_states'].append(hidden_states.copy())
            data['pre_actions'].append(pre_action.copy())
            data['actions'].append(pre_action.copy()+action_delta.copy())
            data['next_observations'].append(next_state.copy())
            data['terminals'].append(offline_dst['terminals'][t])
            data['time_step'].append(offline_dst['time_step'][t])
            if offline_dst['terminals'][t]:
                break

            t += 1
            cur_state = next_state
        
    rt = offline_dst['ref_start_index'][ref_shot][0]
    cur_state = offline_dst['observations'][rt]
    while True:
        pre_action = offline_dst['pre_actions'][rt]
        action_delta = offline_dst['action_deltas'][rt]

        state_delta = offline_dst['observations_delta'][t]
        next_state = cur_state + state_delta
        data['tracking_states'].append(cur_state.copy())
        data['tracking_pre_actions'].append(pre_action.copy())
        data['tracking_actions'].append(pre_action.copy()+action_delta.copy())
        if offline_dst['terminals'][rt]:
            break

        rt += 1
        cur_state = next_state
    
    for k in data:
        data[k] = np.array(data[k])
        print(k, data[k].shape)
    
    data['action_lower_bounds'] = offline_dst['action_lower_bounds'].copy()
    data['action_upper_bounds'] = offline_dst['action_upper_bounds'].copy()
    data['pre_actions'] = np.clip(data['pre_actions'], data['action_lower_bounds'], data['action_upper_bounds'])
    data['actions'] = np.clip(data['actions'], data['action_lower_bounds'], data['action_upper_bounds'])
    data['tracking_pre_actions'] = np.clip(data['tracking_pre_actions'], data['action_lower_bounds'], data['action_upper_bounds'])
    data['tracking_actions'] = np.clip(data['tracking_actions'], data['action_lower_bounds'], data['action_upper_bounds'])
    
    with h5py.File(data_path, 'w') as hdf:
        for key, value in data.items():
            hdf.create_dataset(key, data=value)

def load_offline_data(data_path, raw_data_dir, tracking_target, use_partial=False):
    offline_data = {}
    # get main components
    with h5py.File(data_path, 'r') as hdf:
        # print("Keys in the file:", list(hdf.keys()))
        offline_data['observations'] = hdf['observations'][:]
        offline_data['hidden_states'] = hdf['hidden_states'][:]
        offline_data['actions'] = hdf['actions'][:]
        offline_data['pre_actions'] = hdf['pre_actions'][:]
        offline_data['next_observations'] = hdf['next_observations'][:]
        offline_data['terminals'] = hdf['terminals'][:]
        offline_data['time_step'] = hdf['time_step'][:]
        offline_data['tracking_states'] = hdf['tracking_states'][:]
        offline_data['tracking_pre_actions'] = hdf['tracking_pre_actions'][:]
        offline_data['tracking_actions'] = hdf['tracking_actions'][:]

        offline_data['action_lower_bounds'] = hdf['action_lower_bounds'][:]
        offline_data['action_upper_bounds'] = hdf['action_upper_bounds'][:]
    
    offline_data['obs_dim'] = offline_data['observations'].shape[1]
    offline_data['act_dim'] = offline_data['actions'].shape[1]
    
    with open(raw_data_dir + 'info.pkl', 'rb') as file:
        data_info = pickle.load(file)
        # print(data_info['state_space'], len(data_info['state_space']))
    offline_data['index_list'] = []
    keyword = tracking_target
    for i in range(offline_data['obs_dim']):
        if data_info['state_space'][i].startswith(keyword):
            offline_data['index_list'].append(i)
    
    offline_data['tracking_ref'] = []
    for i in range(len(offline_data['tracking_states'])):
        offline_data['tracking_ref'].append(offline_data['tracking_states'][i][offline_data['index_list']])

    offline_data['tracking_ref'] = np.array(offline_data['tracking_ref'])

    offline_data['full_observations'] = offline_data['observations'].copy()
    offline_data['full_actions'] = offline_data['actions'].copy()
    offline_data['full_next_observations'] = offline_data['next_observations'].copy()
    # offline_data['ref_start_index'] = []
    # for i in range(len(offline_data['observations'])):
    #     if offline_data['time_step'][i] < offline_data['tracking_ref'].shape[0] - 1:
    #         offline_data['ref_start_index'].append(i)

    if use_partial:
        state_idxs = state_names_to_idxs(raw_data_dir)
        action_idxs = actuator_names_to_idxs(raw_data_dir)
        offline_data['state_idxs'] = state_idxs
        offline_data['action_idxs'] = action_idxs

        offline_data['observations'] = offline_data['observations'][:, state_idxs]
        offline_data['next_observations'] = offline_data['next_observations'][:, state_idxs]
        offline_data['actions'] = offline_data['actions'][:, action_idxs]
        offline_data['action_lower_bounds'] = offline_data['action_lower_bounds'][action_idxs]
        offline_data['action_upper_bounds'] = offline_data['action_upper_bounds'][action_idxs]
        
        offline_data['obs_dim'] = offline_data['observations'].shape[1]
        offline_data['act_dim'] = offline_data['actions'].shape[1]

        offline_data['index_list'] = []
        keyword = tracking_target
        for i in range(offline_data['obs_dim']):
            if states_in_obs[i].startswith(keyword):
                offline_data['index_list'].append(i)

    else:
        offline_data['state_idxs'] = list(range(0, offline_data['obs_dim']))
        offline_data['action_idxs'] = list(range(0, offline_data['act_dim']))

    return offline_data


class SA_processor:
    def __init__(self, bounds, time_limit, device):
        self.time_limit = time_limit
        self.range = torch.FloatTensor((bounds[1] - bounds[0]) / 2.0).unsqueeze(0).to(device)
        self.mid = torch.FloatTensor((bounds[1] + bounds[0]) / 2.0).unsqueeze(0).to(device)

        self.np_range = ((bounds[1] - bounds[0]) / 2.0)[np.newaxis, :]
        self.np_mid = ((bounds[1] + bounds[0]) / 2.0)[np.newaxis, :]
    
    def get_rl_state(self, state, time_steps):
        if type(state) == np.ndarray:
            time_input = time_steps / float(self.time_limit)
            rl_state = np.concatenate([state, time_input], axis=-1)
        else:
            time_input = torch.FloatTensor(time_steps / float(self.time_limit)).unsqueeze(-1).to(state.device)
            rl_state = torch.cat([state, time_input], dim=-1)

        return rl_state
    
    def restore_time_steps(self, float_time_steps):
        # print(float_time_steps.shape, np.rint(float_time_steps * self.time_limit).shape)
        return np.rint(float_time_steps * self.time_limit).astype(int)
    
    def get_step_action(self, action):
        act_num = action.shape[0]
        if type(action) == np.ndarray:
            return action * np.repeat(self.np_range, repeats=act_num, axis=0) + np.repeat(self.np_mid, repeats=act_num, axis=0)
        
        return action * self.range.repeat(act_num, 1) + self.mid.repeat(act_num, 1)
    
    def normalize_action(self, action):
        act_num = action.shape[0]
        return (action - self.mid.repeat(act_num, 1).cpu().numpy()) / (self.range.repeat(act_num, 1).cpu().numpy()+1e-6)


class NFEnv:
    def __init__(self, model_dir, device, tracking_ref, tracking_states, tracking_pre_actions, tracking_actions, index_list, sa_processor, state_idxs=None, action_idxs=None):
        self.cur_time = None
        self.time_limit = tracking_ref.shape[0]
        self.tracking_ref = np.array(tracking_ref)
        self.tracking_states = np.array(tracking_states)
        self.tracking_pre_actions = np.array(tracking_pre_actions)
        self.tracking_actions = np.array(tracking_actions)

        self.index_list = np.array(index_list) # the dimensions that the target quantity corresponds to

        self.track_coefficients = np.array([1.0 for _ in range(len(index_list))])
        coe = sum(self.track_coefficients)
        self.track_coefficients = self.track_coefficients / coe

        # load the well-trained rnn model ensemble
        ensemble = load_ensemble_from_parent_dir(parent_dir=model_dir) # TODO: an ensemble of dynamics models
        self.all_models = ensemble.members
        for memb in self.all_models:
            memb.to(device)
            memb.eval()
        
        self.device = device
        self.sa_processor = sa_processor

        self.state_idxs = state_idxs
        self.action_idxs = action_idxs

    def seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)

    def reset(self):
        self.cur_time = random.randint(0, 9)
        self.cur_state = torch.FloatTensor(self.tracking_states[self.cur_time]).unsqueeze(0).to(self.device)
        self.pre_action = torch.FloatTensor(self.tracking_pre_actions[self.cur_time]).unsqueeze(0).to(self.device)

        # reset the model
        for memb in self.all_models:
            memb.reset()
        
        if self.state_idxs is not None:
            return_state = self.cur_state[:, self.state_idxs]
        else:
            return_state = self.cur_state
        
        return self.sa_processor.get_rl_state(return_state, np.array([self.cur_time]))

    def step(self, cur_action):
        cur_action = torch.Tensor(cur_action).to(self.device)
        cur_action = self.sa_processor.get_step_action(cur_action)

        if self.action_idxs is not None:
            cur_action_pad = torch.FloatTensor(self.tracking_actions[self.cur_time]).unsqueeze(0).to(self.device)
            cur_action_pad[:, self.action_idxs] = cur_action
            cur_action = cur_action_pad

        net_input = torch.cat([self.cur_state, self.pre_action, cur_action-self.pre_action], dim=-1)

        ensemble_preds = 0.
        with torch.no_grad():
            for memb in self.all_models:
                net_input_n = memb.normalizer.normalize(net_input, 0)
                net_output_n, _ = memb.single_sample_output_from_torch(net_input_n) # torch.Size([1, 27])
                net_output = memb.normalizer.unnormalize(net_output_n, 1)
                ensemble_preds += net_output
        ensemble_preds = ensemble_preds / float(len(self.all_models)) # delta of the state

        self.cur_state = self.cur_state + ensemble_preds # the next state, TODO: use the true value for the unselected dimensions
        self.cur_time += 1
        # self.pre_action = torch.FloatTensor(self.tracking_pre_actions[self.cur_time]).unsqueeze(0).to(self.device)
        self.pre_action = cur_action.clone()

        if self.state_idxs is not None:
            return_state = self.cur_state[:, self.state_idxs]
        else:
            return_state = self.cur_state

        return self.sa_processor.get_rl_state(return_state, np.array([self.cur_time])), self.get_reward(return_state.cpu().numpy(), self.cur_time)[0], self.is_done(self.cur_time), {}

    def get_reward(self, next_state, time_step): 

        if np.isscalar(time_step):
            if time_step >= self.time_limit:
                time_step = self.time_limit - 1
            targets = self.tracking_ref[time_step]
        else:
            time_step[time_step >= self.time_limit] = self.time_limit - 1
            targets = self.tracking_ref[time_step]

        return -1.0 * (np.square(next_state[:, self.index_list] - targets) * self.track_coefficients[np.newaxis, :]).sum(axis=1) / float(self.time_limit)
        # return -1.0 * (np.square(next_state[:, self.index_list] - targets) * self.track_coefficients[np.newaxis, :]).sum(axis=1)

    def is_done(self, time_step):
        # print('fusion env', time_step, self.time_limit)
        return time_step >= self.time_limit - 1

    def get_normalized_score(self, ep_reward):
        return ep_reward / 100.0
