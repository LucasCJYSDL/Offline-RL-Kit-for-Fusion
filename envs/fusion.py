import torch
import numpy as np
import pickle
import h5py
import random

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

def get_offline_data(data_path, tracking_target):
    offline_data = {}
    # get main components
    with h5py.File(data_path, 'r') as hdf:
        # print("Keys in the file:", list(hdf.keys()))
        offline_data['observations'] = hdf['observations'][:]
        # offline_data['actions'] = hdf['actions'][:]
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
        targets = self.tracking_ref[time_step]
        return -1.0 * (np.square(next_state[:, self.index_list] - targets) * self.track_coefficients[np.newaxis, :]).mean(axis=1) / float(self.time_limit)
        # return -1.0 * (np.square(next_state[:, self.index_list] - targets) * self.track_coefficients[np.newaxis, :]).mean(axis=1) 

    def is_done(self, time_step):
        return time_step >= self.time_limit - 1
